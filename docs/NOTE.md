# NOTE List

## Multiplication Operator Usage

- **中文**: 乘法运算符 `*` 绝对不能省略！ `'ab'` 会被解析为一个单独的符号，而不是 `a*b`。
- **English**: Multiplication operator `*` must never be omitted! `'ab'` will be parsed as a single symbol, not as `a*b`.

## Fraction Creation Method

- **中文**: 分数应该使用 SymPy 的 `sympify(string)` 或 `Rational(a, b)` 创建. 不要使用 `/`，否则可能被解释为不精确的小数。
- **English**: Fractions should be created using SymPy's `sympify(string)` or `Rational(a, b)`
  Do not use `/`, otherwise it may be interpreted as an imprecise decimal.

## Inverse Trigonometric Function Notation

- **中文**: 在输入表达式时，反三角函数是 `asin()`, `acos()`, `atan()`，而不是 arcsin(), arccos(), arctan()。
- **English**: When entering expressions, inverse trigonometric functions are asin(), acos(), atan(), not arcsin(), arccos(), arctan().

## Expression Validation

- **中文**: 将所有的验证表达式 `a == b` 替换为 `a.equals(b)`。后者能够处理具有多种等效形式的表达式情况。
- **English**: Replace all validation expressions `a == b` with `a.equals(b)`. The latter can handle cases where expressions have multiple equivalent forms.

## Parentheses Formatting

- **中文**: 将所有括号替换为 `\left(` 和 `\right)` 以获得更好的视觉效果。这些括号会自动调整以匹配表达式的高度。
- **English**: Replace all parentheses with `\left(` and `\right)` for better visual appeal. These automatically adjust to match the height of the expression.
