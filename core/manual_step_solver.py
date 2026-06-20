"""Factory for domain-specific manual step-by-step solvers.

The shared orchestration logic lives in
:class:`core.base_manual_step_solver.BaseManualStepSolver`. Each domain
(diff / integral / limit) provides its own subclass under ``domains/``,
and the :func:`ManualStepSolver` factory below selects the right one.

Existing callers that do ``from core.manual_step_solver import ManualStepSolver``
continue to work unchanged.
"""

from typing import Any

from core.base_manual_step_solver import BaseManualStepSolver

# Backwards-compatible re-export so callers can grab the base class from here.
__all__ = ['BaseManualStepSolver', 'ManualStepSolver']


def ManualStepSolver(domain: str, expression: str, variable: str = 'x',
                     point: Any = 0, direction: str = '+') -> BaseManualStepSolver:
    """Construct the domain-appropriate manual step solver.

    This is intentionally a factory function (not a class) so the call site
    reads the same as before while the actual work is delegated to the
    domain-specific subclass.
    """
    # Local imports: the domain subclasses import BaseManualStepSolver from
    # core, so importing them at module scope would create a cycle.
    from domains import (  # pylint: disable=import-outside-toplevel
        DiffManualStepSolver, IntegralManualStepSolver, LimitManualStepSolver,
    )

    solvers = {
        'diff': DiffManualStepSolver,
        'integral': IntegralManualStepSolver,
        'limit': LimitManualStepSolver,
    }
    if domain not in solvers:
        raise ValueError(
            f"Unsupported domain: {domain}. "
            f"Expected one of {list(solvers)}")

    return solvers[domain](
        expression=expression,
        variable=variable,
        point=point,
        direction=direction,
    )
