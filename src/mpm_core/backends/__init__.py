"""Solver backend registry. Backends are imported lazily so importing this module stays cheap."""

from __future__ import annotations

from collections.abc import Callable

from mpm_core.solver import Solver

_factories: dict[str, Callable[..., Solver]] = {}


def _warp_factory(**kwargs) -> Solver:
    from mpm_core.backends.warp.solver import WarpSolver

    return WarpSolver(**kwargs)


_factories["warp"] = _warp_factory


def register_backend(name: str, factory: Callable[..., Solver]) -> None:
    _factories[name] = factory


def create_solver(backend: str, **kwargs) -> Solver:
    if backend not in _factories:
        raise KeyError(f"unknown solver backend {backend!r}, known: {sorted(_factories)}")
    return _factories[backend](**kwargs)
