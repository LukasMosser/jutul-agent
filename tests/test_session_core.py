"""``start_core``: the kernel half of the session bootstrap, usable without an agent.

``SessionHost.start`` builds the agent on top of a core (its own tests cover
that path); these pin what a front end that brings its own agent relies on: a
core stands up the session with no agent built, defers the warm-up until asked,
and tears down in the safe order.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from jutul_agent.agent.capabilities import Capability
from jutul_agent.lab.fakes import FakeJulia, make_fake_adapter


@pytest.fixture
def fake_kernel_world(monkeypatch) -> dict[str, Any]:
    """Patch out Julia, env preparation and package resolution; record the calls."""
    import jutul_agent.juliakernel as juliakernel_mod
    from jutul_agent.agent import builder as builder_mod
    from jutul_agent.julia import requirements as requirements_mod
    from jutul_agent.simulators import env_setup as env_setup_mod

    world: dict[str, Any] = {"kernel": FakeJulia(), "prepared": []}
    monkeypatch.setattr(requirements_mod, "require_julia", lambda *a, **kw: None)
    monkeypatch.setattr(juliakernel_mod, "JuliaKernel", lambda config: world["kernel"])
    monkeypatch.setattr(builder_mod, "resolve_package_sources", lambda project: ["pkg-src"])

    def _no_agent(*_a, **_kw):
        raise AssertionError("start_core must not build an agent")

    monkeypatch.setattr(builder_mod, "build_agent", _no_agent)

    def _prepare(simulator, *, workspace, julia_project, sim_name, dependencies):
        world["prepared"].append(dependencies)

    monkeypatch.setattr(env_setup_mod, "prepare_workspace_env", _prepare)
    return world


async def test_start_core_opens_a_session_without_an_agent(
    fake_kernel_world: dict[str, Any], tmp_path: Path
) -> None:
    from jutul_agent.session_core import start_core

    workspace = tmp_path / "ws"
    workspace.mkdir()
    dep = tmp_path / "Cap" / "Project.toml"
    cap = Capability(name="cap", dependencies=(dep,), warm_packages=("CapPkg",))

    core = await start_core(
        simulator=make_fake_adapter(tmp_path),
        surface="mcp",
        workspace=workspace,
        state_root=tmp_path,
        extensions=[cap],
        discover=False,
        virtual_display=False,
    )
    try:
        assert core.session.surface == "mcp"
        assert core.workspace == workspace
        assert core.extensions == [cap]
        assert core.package_sources == ["pkg-src"]
        # The capability's Julia dependencies went into the env it starts against.
        assert fake_kernel_world["prepared"] == [[dep]]
        # The tools' Julia is the yielding wrapper around the raw kernel.
        assert core.kernel is fake_kernel_world["kernel"]
        assert core.session.julia is not core.kernel
        # The warm-up waits for the caller.
        assert core.warmup_task is None
    finally:
        await core.aclose()


async def test_start_warmup_is_idempotent_and_aclose_cancels_it(
    fake_kernel_world: dict[str, Any], tmp_path: Path, monkeypatch
) -> None:
    from jutul_agent.session_core import start_core
    from jutul_agent.simulators import warmup as warmup_mod
    from jutul_agent.trace import TraceLog
    from jutul_agent.trace.schema import SESSION_END

    calls: list[tuple[Any, ...]] = []

    def _start_warmup(julia, warm_package, packages=(), code=(), abandonable=None):
        calls.append((julia, warm_package, list(packages)))
        return asyncio.create_task(asyncio.sleep(3600))

    monkeypatch.setattr(warmup_mod, "start_warmup", _start_warmup)
    cap = Capability(name="cap", warm_packages=("CapPkg",))
    core = await start_core(
        simulator=make_fake_adapter(tmp_path),
        surface="mcp",
        workspace=tmp_path,
        state_root=tmp_path,
        extensions=[cap],
        discover=False,
        virtual_display=False,
        prepare_env=False,
    )
    core.start_warmup()
    core.start_warmup()
    task = core.warmup_task
    assert task is not None
    # Once, against the raw kernel, with the capability's package included.
    assert calls == [(core.kernel, core.session.simulator.warm_package, ["CapPkg"])]

    await core.aclose()
    assert task.cancelled()
    with TraceLog.open_readonly(core.session.state_dir / "trace.sqlite") as log:
        assert [e.kind for e in log.iter_events()][-1] == SESSION_END
