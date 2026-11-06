"""Normalize user input and turn internal failures into user-facing messages.

The frontend can emit MathLive LaTeX, textbook text and matrix literals.  This
module is the single boundary that decides what those mean, so the dispatch
code never has to contain parser-specific regexes or exception strings.
"""

import re
from functools import lru_cache

from sympy import I, Symbol

from core.latex_syntax import _fold_matrix_ops, _iter_op_calls, _normalize_latex_ops
from utils.cache_utils import cached_sympify, has_unfilled_placeholders
from utils.expression_utils import strip_outer_delimiters
from utils.matrix_latex import latex_to_nested

_STRUCTURAL_OPERATOR_RE = re.compile(
    r'\\(?:int|iint|iiint|sum|prod|lim|partial|frac\s*\{\s*d\s*\})'
)


@lru_cache(maxsize=1024)
def looks_like_operator_input(expression: str) -> bool:
    """True when the user typed an explicit operator/template in the box.

    This is what makes the WYSIWYG palette work everywhere: a key inserts
    ``diff(x^2,x)``, ``\\int_0^1 ...``, ``\\det [matrix]`` or ``M^{-1}`` and the
    same dispatcher handles it no matter which operation selector is active.
    """
    if not isinstance(expression, str) or not expression.strip():
        return False
    text = latex_to_nested(expression)
    text = _fold_matrix_ops(text)
    text = _normalize_latex_ops(text)
    if _iter_op_calls(text):
        return True
    return bool(_STRUCTURAL_OPERATOR_RE.search(expression))


def should_route_as_operator(expression: str, operation_type: str = None) -> bool:
    """Single source of truth for preview (/api/parse) and compute (/api/compute).

    An explicit operator template wins over the selected operation, except for
    the ops that must parse the input box as data.  Both callers must use this
    same predicate: when they disagreed, an operator such as ``diff(x^2,x)``
    was previewed structurally but fed to ``Matrix(...)`` (or vice versa).
    """
    from core.operator_registry import _AUTO_ROUTE_SKIP  # pylint: disable=import-outside-toplevel

    return (operation_type not in (None, '', 'auto')
            and operation_type not in _AUTO_ROUTE_SKIP
            and looks_like_operator_input(expression))


def normalize_scalar_input(expression: str, complex_mode: bool = False):
    """Turn WYSIWYG LaTeX / textbook text into a SymPy-source string."""
    text = strip_outer_delimiters(expression)
    if not text:
        return False, '请输入表达式'
    if has_unfilled_placeholders(text):
        return False, '请先填写输入框中的占位（□）后再计算'
    try:
        obj = cached_sympify(text)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, str(exc)
    if complex_mode and hasattr(obj, 'subs'):
        obj = obj.subs(Symbol('i'), I)
    if isinstance(obj, (list, tuple, dict)):
        return False, '请输入单个数学表达式（矩阵请使用矩阵运算入口）'
    return True, str(obj)


def normalize_matrix_input(expression: str, validate: bool = True):
    """Convert MathLive bmatrix blocks and validate the resulting matrix text."""
    text = strip_outer_delimiters(expression)
    if has_unfilled_placeholders(text):
        return False, '请先填写输入框中的占位（□）后再计算'
    try:
        text = latex_to_nested(text)
    except ValueError as exc:
        return False, str(exc)
    if validate:
        from core.common_matrix_calculator import CommonMatrixCalculator  # pylint: disable=import-outside-toplevel
        for line in text.split('\n'):
            if not line.strip():
                continue
            try:
                CommonMatrixCalculator.parse_matrix_input(line)
            except Exception:  # pylint: disable=broad-exception-caught
                return False, (
                    '矩阵格式不正确：请使用 [[1,2],[3,4]] 或 bmatrix 形式，'
                    '并保证每行元素个数相同'
                )
    return True, text


def as_bool_flag(value, default: bool = False) -> bool:
    """Interpret a UI flag that may arrive as a real bool or as ``'True'``/``'False'``.

    The notebooks send strings, but API clients (and the tests) may send JSON
    booleans; comparing with ``== 'True'`` silently ignored the latter.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('true', '1', 'yes')


def user_error(exc, operation_type: str) -> str:
    """Map internal parser/engine exceptions to a readable user-facing message."""
    msg = str(exc).strip()
    low = msg.lower()
    if 'placeholder' in low or '占位' in msg or '空白单元格' in msg or '请先填写' in msg:
        return '请先填写输入框中的占位（□）后再计算'
    if 'cannot convert' in low or 'could not parse' in low or 'sympify' in low \
            or 'syntaxerror' in low or 'unexpected character' in low:
        return f'无法解析输入，请检查表达式写法: {msg}'
    if 'exceeds the limit' in low or 'int too large to convert' in low \
            or '4300 digits' in low:
        # A huge power stays unevaluated while parsing, but evaluating/printing it
        # later trips Python's int<->str digit limit. Say so in plain language
        # instead of surfacing the CPython internal message.
        return '表达式中的幂次或整数过大, 无法展开或显示; 请改用较小的指数'
    if 'list index out of range' in low or 'not enough values' in low \
            or 'pop from empty' in low:
        return '矩阵输入格式不正确，请检查每行元素个数和运算参数'
    if 'non-square' in low or 'not a square' in low or 'square matrix' in low:
        return '该运算要求方阵，请检查矩阵行列数'
    if 'shape' in low or 'dimension' in low or 'size' in low:
        return f'矩阵维度不匹配: {msg}'
    if 'singular' in low or 'not invertible' in low:
        return '该矩阵不可逆（行列式为 0）'
    return f'计算失败（{operation_type}）: {msg}'
