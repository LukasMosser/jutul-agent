"""The session behind the MCP server, and how its tools are called.

One server process is one session: one Julia kernel, one trace, one workspace.
The MCP handshake must not wait for Julia, so the session starts in the
background when the server does, and listing or calling a tool waits for it.

A call goes through the session's own tools (``build_tools``, the list the
bundled agent is given), recorded in the trace as the agent's calls are, so a
transcript or report of an MCP session reads like any other.
"""

from __future__ import annotations

import asyncio
import re
import secrets
import sys
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from jutul_agent.agent.capabilities import Capability
    from jutul_agent.interfaces.mcp.loopback import LoopbackViewer
    from jutul_agent.session import Session
    from jutul_agent.session_core import SessionCore
    from jutul_agent.simulators.base import SimulatorAdapter

SURFACE = "mcp"

# How long a cancelled call waits for Julia to unwind before giving up on it. The
# kernel's own recovery is bounded (it restarts Julia if an interrupt is ignored).
_CANCEL_GRACE = 60.0


@dataclass(frozen=True)
class McpLaunch:
    """What the server runs: the folder's simulator and environment."""

    simulator: SimulatorAdapter
    workspace: Path
    julia_project: Path | None = None
    threads: str | None = None
    sysimage: bool | None = None
    prepare_env: bool = True
    virtual_display: bool = True
    viewer: bool = True
    viewer_port: int = 0
    capabilities: Sequence[Capability] = field(default_factory=tuple)


class StartupFailed(Exception):
    """The session could not start; the message says why, for the user."""


class UnknownTool(Exception):
    """A call named a tool this server does not have."""


@dataclass
class ToolOutcome:
    """A finished call: the tool's content, its status, and what it produced."""

    content: str | list[Any]
    status: str
    artifacts: list[dict[str, Any]]
    # Something the client missed: a cancelled call whose result it never saw.
    notice: str | None = None


# The tools that run Julia code, so cancelling them interrupts an eval.
_EVAL_TOOLS = frozenset({"run_julia", "plot_julia", "recapture_plot"})

_STATE_LOST = (
    "Note: a cancelled call's Julia code ignored the interrupt, so the session "
    "restarted. Variables and `using` imports from before are gone."
)


def capability_session_id() -> str:
    """A session id that doubles as the capability to view the session.

    Dated like every session id (so it sorts and lists the same), with 96 random
    bits after it: viewer links carry the id, so it must not be guessable.
    """
    from jutul_agent.session import default_session_id

    return default_session_id() + secrets.token_hex(12)


class McpRuntime:
    """The one session a server process runs, started in the background."""

    def __init__(self, launch: McpLaunch, *, session_id: str | None = None) -> None:
        self.launch = launch
        self.session_id = session_id or capability_session_id()
        self.core: SessionCore | None = None
        self.viewer: LoopbackViewer | None = None
        self._tools: dict[str, BaseTool] = {}
        self._startup: asyncio.Task[None] | None = None
        self._notice: str | None = None

    async def start(self) -> None:
        """Serve the viewer, and start the session behind it (without waiting)."""
        if self.launch.viewer:
            from jutul_agent.interfaces.mcp.loopback import LoopbackViewer

            self.viewer = LoopbackViewer(
                live_session=self._live_session,
                output_dir=self._output_dir,
                port=self.launch.viewer_port,
            )
            await self.viewer.start()
        self._startup = asyncio.create_task(self._start_session())

    async def _start_session(self) -> None:
        from jutul_agent.agent.builder import build_tools
        from jutul_agent.session_core import start_core

        launch = self.launch
        _note(f"starting Julia for {launch.simulator.display_name} in {launch.workspace}")
        core = await start_core(
            simulator=launch.simulator,
            surface=SURFACE,
            session_id=self.session_id,
            workspace=launch.workspace,
            julia_project=launch.julia_project,
            threads=launch.threads,
            open_windows=False,
            prepare_env=launch.prepare_env,
            discover=False,  # the launch already composed them
            virtual_display=launch.virtual_display,
            extensions=list(launch.capabilities),
            sysimage=launch.sysimage,
        )
        self.core = core  # from here on, ``aclose`` owns it
        self._tools = {
            tool.name: tool
            for tool in build_tools(core.session, surface=SURFACE, extensions=core.extensions)
        }
        core.start_warmup()
        _note("Julia is up" + (f"; viewer at {self.viewer.base_url}" if self.viewer else ""))

    async def ready(self) -> SessionCore:
        """The running session, once it is up; ``StartupFailed`` if it cannot start."""
        startup = self._startup
        if startup is None:
            raise StartupFailed("the jutul-agent session is not running")
        try:
            # Shielded: one caller giving up must not cancel the startup others share.
            await asyncio.shield(startup)
        except asyncio.CancelledError:
            if startup.cancelled():
                raise StartupFailed("the jutul-agent session was shut down") from None
            raise
        except Exception as exc:
            raise StartupFailed(f"jutul-agent could not start its Julia session: {exc}") from exc
        assert self.core is not None
        return self.core

    async def tools(self) -> list[BaseTool]:
        await self.ready()
        return list(self._tools.values())

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        on_output: Callable[[str], None] | None = None,
    ) -> ToolOutcome:
        """Run one tool call, recorded in the trace; ``on_output`` gets live output."""
        from jutul_agent.agent.tools import tool_output_writer
        from jutul_agent.trace.messages import content_to_str
        from jutul_agent.trace.schema import (
            ARTIFACT,
            TOOL_CALL,
            TOOL_RESULT,
            tool_call_payload,
            tool_result_payload,
        )

        core = await self.ready()
        tool = self._tools.get(name)
        if tool is None:
            raise UnknownTool(name)
        trace = core.session.trace
        call_id = f"mcp-{uuid.uuid4().hex[:12]}"
        mark = trace.max_id()
        trace.append(TOOL_CALL, tool_call_payload(id=call_id, name=name, args=arguments))

        # The task copies this context, so the tool's output reaches ``on_output``.
        token = tool_output_writer.set(on_output)
        try:
            task = asyncio.create_task(_invoke(tool, call_id, arguments))
        finally:
            tool_output_writer.reset(token)
        try:
            content, status = await _cancel_once(task)
        except asyncio.CancelledError:
            # The client never sees this call's result, so if the cancel cost the
            # session its state, the next result says so.
            lost = (
                name in _EVAL_TOOLS
                and task.done()
                and not getattr(core.session.julia, "cancel_preserved_state", True)
            )
            if lost:
                self._notice = _STATE_LOST
            trace.append(
                TOOL_RESULT,
                tool_result_payload(
                    tool_call_id=call_id,
                    name=name,
                    content=f"Cancelled. {_STATE_LOST}" if lost else "Cancelled.",
                    status="error",
                ),
            )
            raise
        trace.append(
            TOOL_RESULT,
            tool_result_payload(
                tool_call_id=call_id, name=name, content=content_to_str(content), status=status
            ),
        )
        # Calls can run concurrently (Julia serializes them), so an artifact that
        # names its call goes to that call; one that names none is in our window.
        artifacts = [
            event.payload
            for event in trace.events_after(mark)
            if event.kind == ARTIFACT and event.payload.get("tool_call_id") in (call_id, None)
        ]
        notice, self._notice = self._notice, None
        return ToolOutcome(content=content, status=status, artifacts=artifacts, notice=notice)

    def artifact_links(self, payloads: list[dict[str, Any]]) -> list[str]:
        """Where the user can see what a call produced: viewer URLs and file paths."""
        if self.core is None:
            return []
        return artifact_links(
            payloads,
            session_id=self.session_id,
            base_url=self.viewer.base_url if self.viewer is not None else None,
            output_dir=self.core.session.output_dir,
        )

    async def aclose(self) -> None:
        startup, self._startup = self._startup, None
        if startup is not None:
            if not startup.done():
                startup.cancel()
                await asyncio.wait({startup})
            if not startup.cancelled():
                startup.exception()  # retrieved; ``ready`` already reported it
        core, self.core = self.core, None
        if core is not None:
            await core.aclose()
        viewer, self.viewer = self.viewer, None
        if viewer is not None:
            await viewer.aclose()

    # The viewer serves this session and nothing else.
    def _live_session(self, session_id: str) -> Session | None:
        core = self.core
        if core is None or not secrets.compare_digest(session_id, self.session_id):
            return None
        return core.session

    def _output_dir(self, session_id: str) -> Path | None:
        session = self._live_session(session_id)
        return session.output_dir if session is not None else None


async def _invoke(tool: BaseTool, call_id: str, arguments: dict[str, Any]) -> tuple[Any, str]:
    """Call a tool as the agent would (a tool call, so injected ids resolve)."""
    try:
        message = await tool.ainvoke(
            {"type": "tool_call", "id": call_id, "name": tool.name, "args": arguments}
        )
    except Exception as exc:  # bad arguments, or a tool that raised
        return f"Error running tool `{tool.name}`: {type(exc).__name__}: {exc}", "error"
    return message.content, getattr(message, "status", None) or "success"


async def _cancel_once(task: asyncio.Task[Any]) -> Any:
    """Await ``task``; if the caller is cancelled, cancel it once and let it unwind.

    The kernel answers a cancelled eval by interrupting Julia and waiting for the
    interrupted eval to finish, so the REPL stays usable for the next call. That
    recovery wants one cancellation. An MCP cancel arrives through an anyio cancel
    scope, which keeps re-delivering it to whatever its task awaits, so the call
    runs as its own task, is cancelled exactly once, and is waited for shielded.
    """
    import anyio

    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        task.cancel()
        with anyio.CancelScope(shield=True):
            await asyncio.wait({task}, timeout=_CANCEL_GRACE)
        if task.done() and not task.cancelled():
            task.exception()  # retrieved; the caller is cancelled either way
        raise


def artifact_links(
    payloads: list[dict[str, Any]],
    *,
    session_id: str,
    base_url: str | None,
    output_dir: Path,
) -> list[str]:
    """One line per produced artifact: where to open it, and where it is on disk.

    A live plot opens on the popout page, which stages the figure at its own size
    in whatever window it gets (its widgets run in the session). Anything else
    (a static plot, a report, a snapshot) opens directly. Without a viewer, the
    path is all there is.
    """
    from jutul_agent.interfaces.server.app import artifact_wire_events

    lines: list[str] = []
    for payload, event in zip(
        payloads, artifact_wire_events(payloads, session_id, live=True), strict=True
    ):
        path = output_dir / str(payload.get("path") or "")
        title = event.get("title") or event.get("caption") or path.name
        if base_url is None:
            lines.append(f"{title}: {path}")
            continue
        if event["type"] == "viz" and event.get("live"):
            route = str(event["url"]).rsplit("/viz/", 1)[-1]
            width, height = _staged_size(event.get("width"), event.get("height"))
            url = f"{base_url}/popout/{quote(session_id)}?route={quote(route)}&w={width}&h={height}"
            lines.append(f"{title} (interactive): {url}")
        else:
            lines.append(f"{title}: {base_url}{event['url']}")
        lines.append(f"  saved as {path}")
    return lines


def _staged_size(width: Any, height: Any) -> tuple[int, int]:
    """The figure size a popout stages, within the bounds the page accepts."""

    def clamp(value: Any, default: int) -> int:
        try:
            return min(max(int(value), 100), 8192)
        except (TypeError, ValueError):
            return default

    return clamp(width, 1200), clamp(height, 800)


# ECMA-48 escapes: CSI (colours, cursor moves), OSC (titles, links), and the
# two-byte ones. A sequence cut off at a chunk boundary is never seen here: the
# relay only reads output up to the last line ending.
_ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")
_LINE_END_RE = re.compile(r"[\r\n]")


def status_line(text: str, limit: int = 200) -> str:
    """The last line in ``text`` that says something, without colour codes.

    Lines end at a newline or a carriage return (a progress meter redraws its line
    after one). A line with no letters or digits, such as a table border, says
    nothing.
    """
    for line in reversed(_LINE_END_RE.split(_ANSI_RE.sub("", text))):
        line = line.strip()
        if any(char.isalnum() for char in line):
            return line if len(line) <= limit else line[: limit - 1] + "…"
    return ""


def split_finished(text: str) -> tuple[str, str]:
    """``text`` split after its last line ending: the finished lines, and the rest."""
    end = max(text.rfind("\n"), text.rfind("\r")) + 1
    return text[:end], text[end:]


def _note(message: str) -> None:
    """A line for the server log (stderr: stdout is the protocol)."""
    print(f"jutul-agent mcp: {message}", file=sys.stderr, flush=True)
