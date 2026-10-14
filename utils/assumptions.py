"""Parse user-supplied simplification assumptions and apply them soundly.

An assumptions string such as ``x>0, y real`` is turned into per-symbol SymPy
assumptions and applied to a *finished* expression by a confined symbol round-trip:
swap the constrained symbols for assumed variants, let ``expand_log`` / ``powdenest``
/ ``simplify`` use those symbol flags, then swap back. This yields ``sqrt(x^2) -> x``
and ``log(x^2)/2 -> log(x)`` WITHOUT changing the solving symbols' identity, touching
the substitution BFS, or applying an unsound ``force=True`` to unconstrained symbols.

The functions here are pure and dependency-free so they can be unit-tested directly.
"""

import re

from sympy import Symbol, expand_log, powdenest, simplify

# operator-form clause -> symbol flags (relationals are NOT valid refine assumptions,
# so ``x>0`` is stored as the ``positive``/``real`` flags rather than a Relational).
_OP_FLAGS = {
    ('>', '0'): {'positive': True, 'real': True},
    ('<', '0'): {'negative': True, 'real': True},
    ('>=', '0'): {'nonnegative': True, 'real': True},
    ('<=', '0'): {'nonpositive': True, 'real': True},
    ('!=', '0'): {'nonzero': True},
    ('==', '0'): {'zero': True},
}
_WORD_FLAGS = {
    'real': {'real': True},
    'integer': {'integer': True},
    'rational': {'rational': True},
    'positive': {'positive': True, 'real': True},
    'negative': {'negative': True, 'real': True},
    'nonnegative': {'nonnegative': True, 'real': True},
    'nonpositive': {'nonpositive': True, 'real': True},
    'nonzero': {'nonzero': True},
    'finite': {'finite': True},
    'odd': {'odd': True},
    'even': {'even': True},
}

# clause separators: ascii/unicode comma & semicolon, newline, or the word 且
_SPLIT_RE = re.compile(r'[,;，；\n]|且')
_OP_RE = re.compile(r'^\s*([A-Za-z_]\w*)\s*(>=|<=|!=|==|>|<)\s*0\s*$')
_WORD_RE = re.compile(r'^\s*([A-Za-z_]\w*)\s+([A-Za-z]+)\s*$')


def parse_assumption_flags(text: str) -> dict:
    """Parse ``text`` into ``{symbol_name: {assumption_kwargs}}``.

    Unrecognized clauses are silently dropped (degrade to "no assumption" rather than
    raising), so a malformed hypotheses field can never 500 the computation task."""
    flags: dict = {}
    for clause in _SPLIT_RE.split(text or ''):
        if not clause.strip():
            continue
        m = _OP_RE.match(clause)
        if m and (m.group(2), '0') in _OP_FLAGS:
            flags.setdefault(m.group(1), {}).update(_OP_FLAGS[(m.group(2), '0')])
            continue
        w = _WORD_RE.match(clause)
        if w and w.group(2).lower() in _WORD_FLAGS:
            flags.setdefault(w.group(1), {}).update(_WORD_FLAGS[w.group(2).lower()])
    return flags


def refine_with_flags(expr, flags: dict):
    """Return ``expr`` simplified under ``flags`` via a confined symbol round-trip.

    Leaves the expression untouched when there are no matching free symbols. The
    returned expression reuses the ORIGINAL symbol objects (identity preserved)."""
    if not flags or expr is None:
        return expr
    free = getattr(expr, 'free_symbols', None)
    if not free:
        return expr
    swap_in: dict = {}
    swap_out: dict = {}
    for sym in free:
        if isinstance(sym, Symbol) and sym.name in flags:
            tmp = Symbol(sym.name, **flags[sym.name])
            swap_in[sym] = tmp
            swap_out[tmp] = sym
    if not swap_in:
        return expr
    refined = expr.xreplace(swap_in)
    refined = expand_log(refined)
    refined = powdenest(refined)
    refined = simplify(refined)
    return refined.xreplace(swap_out)
