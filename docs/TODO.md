# TODO List

## Expression Validation

- **中文**: 将所有的验证表达式 `a == b` 替换为 `a.equals(b)`。后者能够处理具有多种等效形式的表达式情况。
- **English**: Replace all validation expressions `a == b` with `a.equals(b)`. The latter can handle cases where expressions have multiple equivalent forms.

## Parentheses Formatting

- **中文**: 将所有括号替换为 `\left(` 和 `\right)` 以获得更好的视觉效果。这些括号会自动调整以匹配表达式的高度。
- **English**: Replace all parentheses with `\left(` and `\right)` for better visual appeal. These automatically adjust to match the height of the expression.

## Power Calculation Optimization

- **中文**: SymPy 显式计算所有幂运算，这会导致指数较大时出现性能问题。我们需要特殊处理来避免直接计算幂值。
- **English**: SymPy evaluates all powers explicitly, which causes performance issues when exponents are large. We need special handling to avoid computing power values directly.

## Auxiliary Feature for Calculation Process

- **中文**: 添加辅助功能：每个计算过程部分应包含链接区域。点击链接区域应该快速生成一个或多个相关的计算单元格，允许用户验证那些直接给出而未显示过程的结果。
- **English**: Add an auxiliary feature: each calculation process section should include a link area. Clicking the link area should quickly generate one or more related calculation cells, allowing users to verify results that are directly given without showing the process.

## Manual Rule Selection Feature

- **中文**: 实现人工选择每一步使用何种规则的功能，以更好地满足用户需求
- **English**: Implement a feature to manually select which rule to apply at each step, to better meet user needs

## Asynchronous Processing Implementation

- **中文**: 后端应返回 `task_id` 并在前端实现`轮询`：实现异步处理以防止前端阻塞。
- **English**: Backend should return `task_id` and implement polling on frontend: Achieve asynchronous processing to prevent frontend blocking.

---

## Script Module (`script.py`)

- **中文**: 外部新建 HTML 文件，进行导入
- **English**: External new HTML file, perform import

---

## Differential Calculator (`diff_calculator.py`)

- **中文**: 支持对含有不确定量但不确定量是常数的函数求导，结果用不确定量表示（偏导的一种特例）；支持求偏导
- **English**: Support differentiation of functions containing uncertain quantities where the uncertain quantity is a constant, with results expressed using the uncertain quantity (a special case of partial derivatives); support partial derivatives

## Integral Calculator (`integral_calculator.py`)

- **中文**: 支持定积分计算（增加上下限参数）；支持多重积分
- **English**: Support definite integral calculation (add upper and lower limit parameters); support multiple integrals

## Limit Calculator (`limit_calculator.py`)

- **中文**: 有些规则的使用具有前提条件，但现在并非所有规则的前提条件都已全面判断；增加等价无穷小和泰勒展开的解法；阶乘极限、数列极限；想办法优化步骤，有些步骤太过冗余
- **English**: Some rules have prerequisite conditions for usage, but not all rule prerequisites are comprehensively judged now; add equivalent infinitesimal and Taylor expansion solutions; factorial limits, sequence limits; find ways to optimize steps, some steps are too redundant

### Base Rules (`limit/rules/base_rules.py`)

- **中文**: `mul_split_rule()` 中的相似代码很多，后续还需进一步重构
- **English**: There is much similar code in `mul_split_rule()`, further refactoring is needed subsequently

## Matrix Symbol Expression Parser (`matrix/symbol_expression_parser.py`)

- **中文**: 实现矩阵符号表达式的化简功能
- **English**: Implement simplification functionality for matrix symbolic expressions

## Expression Parser Enhancement (`expression_parser.py`)

- **中文**: 完善 `expression_parser.py`，现在好像有一些问题
- **English**: Improve the functionality of `epression_parser.py`, as there are currently some issues that need to be addressed

## Integral Calculator Rules and Matcher Enhancement

- **中文**: 完善积分部分的规则和匹配器，当前只能处理简单的积分计算，需要扩展更多复杂的积分场景支持
- **English**: Improve the rules and matcher, as it currently can only handle simple integral calculations and needs to support more complex integration scenarios
