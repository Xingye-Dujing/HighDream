# NOTE List

## Multiplication Operator Usage
- **中文**: 乘法运算符 `*` 绝对不能省略！ `'ab'` 会被解析为一个单独的符号，而不是 `a*b`。
- **English**: Multiplication operator `*` must never be omitted! `'ab'` will be parsed as a single symbol, not as `a*b`.


## Fraction Creation Method
- **中文**: 分数应该使用 SymPy 的 `sympify()` 或 `Rational()` 创建.  不要使用 `/`，否则可能被解释为不精确的小数。
- **English**: Fractions should be created using SymPy's `sympify()` or `Rational()`
  Do not use `/`, otherwise it may be interpreted as an imprecise decimal.
