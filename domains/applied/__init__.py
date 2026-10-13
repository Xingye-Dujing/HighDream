"""Applied-calculus / series / ODE / complex / numerical operations package.

Organized like the other domains: one module per concern, handlers aggregated
into ``HANDLERS`` for ``compute_service``. Ops whose core is differentiation or
integration reuse ``DiffCalculator`` / ``IntegralCalculator`` so the working is
shown step by step, matching the main engines.
"""

from domains.applied import (applications, complex_numbers, integral_apps,
                             multivar, numerical, ode, series)

HANDLERS = {
    # applications of differentiation
    'tangent': applications.tangent,
    'linearization': applications.linearization,
    'critical_points': applications.critical_points,
    'newton': applications.newton,
    'mean_value': applications.mean_value,
    # integral applications
    'arc_length': integral_apps.arc_length,
    'volume_disk': integral_apps.volume_disk,
    'volume_shell': integral_apps.volume_shell,
    'surface_area': integral_apps.surface_area,
    # series / Taylor / improper
    'taylor': series.taylor,
    'taylor_error': series.taylor_error,
    'series_convergence': series.series_convergence,
    'radius_of_convergence': series.radius_of_convergence,
    'improper_integral': series.improper_integral,
    # implicit / parametric / polar
    'implicit_diff': multivar.implicit_diff,
    'parametric_diff': multivar.parametric_diff,
    'polar_area': multivar.polar_area,
    'polar_slope': multivar.polar_slope,
    # ODE
    'ode': ode.solve_ode,
    # complex numbers
    'complex_simplify': complex_numbers.complex_simplify,
    'complex_polar': complex_numbers.complex_polar,
    'complex_roots': complex_numbers.complex_roots,
    'de_moivre': complex_numbers.de_moivre_power,
    # numerical
    'numerical_integration': numerical.numerical_integration,
}
