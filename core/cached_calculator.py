from sympy import Derivative, Integral, Limit

from utils.db_cache import build_hash, get as cache_get, set_cache as cache_set


def _detect_operation_type(calc):
    op_type = getattr(calc, 'operation', None)
    if op_type is not None:
        mapping = {Derivative: 'diff', Integral: 'integral', Limit: 'limit'}
        return mapping.get(op_type, 'unknown')
    return 'unknown'


class DBCachedCalculator:
    # pylint: disable=too-few-public-methods
    """Mixin that adds database cache lookup before computation.

    Must be placed before the calculator class in MRO::

        class CachedDiff(DBCachedCalculator, DiffCalculator):
            pass
    """

    def compute_latex(self, expr, *args, **kwargs):
        # pylint: disable=no-member
        use_cache = kwargs.pop('use_cache', False)
        record_cache = kwargs.pop('record_cache', False)

        op_type = _detect_operation_type(self)

        context = dict(kwargs)
        if args:
            if op_type in ('diff', 'integral'):
                context['variable'] = str(args[0])
            elif op_type == 'limit':
                context['variable'] = str(args[0]) if len(args) > 0 else ''
                context['point'] = str(args[1]) if len(args) > 1 else ''
                context['direction'] = str(args[2]) if len(args) > 2 else ''

        key = build_hash(op_type, expr, **context)

        if use_cache:
            cached = cache_get(key)
            if cached is not None:
                return cached

        result = super().compute_latex(expr, *args, **kwargs)

        if record_cache:
            cache_set(key, op_type, expr, result, **context)

        return result
