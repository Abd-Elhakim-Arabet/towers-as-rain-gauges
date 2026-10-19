"""Model factory: returns the TCN for whichever framework configs/backend.yaml selects."""

from __future__ import annotations

from targ.backends import get_framework


def build_tcn(n_dynamic: int, n_static: int, **kw):
    if get_framework() == "mindspore":
        from .tcn_ms import TCN
    else:
        from .tcn_torch import TCN
    return TCN(n_dynamic, n_static, **kw)
