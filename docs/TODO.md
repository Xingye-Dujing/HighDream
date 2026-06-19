# TODO List

1. SymPy 显式计算所有幂运算，这会导致指数较大时出现性能问题。我们需要特殊处理来避免直接计算幂值。

SymPy evaluates all powers explicitly, which causes performance issues when exponents are large. We need special handling to avoid computing power values directly.

2. 每个计算过程部分应包含链接区域。点击链接区域应该快速生成一个或多个相关的计算单元格，允许用户验证那些直接给出而未显示过程的结果。

Each calculation process section should include a link area. Clicking the link area should quickly generate one or more related calculation cells, allowing users to verify results that are directly given without showing the process.

4. 将 `script.js` 中的 HTML 标签提出到外部文件，`script.js` 进行导入而不是内嵌。这样便于网页的样式设计。

Extract the HTML tags from `script.js` into external files, and have `script.js` import them instead of embedding them inline. This facilitates web page styling design.

5. 求导相关：支持对含有不确定量但不确定量是常数的函数求导，结果用不确定量表示（偏导的一种特例）；支持求偏导。

Support differentiation of functions containing uncertain quantities where the uncertain quantity is a constant, with results expressed using the uncertain quantity (a special case of partial derivatives); support partial derivatives.

6. 积分相关：支持定积分计算（增加上下限参数）；支持多重积分。

Support definite integral calculation (add upper and lower limit parameters); support multiple integrals.

7. 极限相关：有些规则的使用具有前提条件，但现在并非所有规则的前提条件都已全面判断；增加等价无穷小和泰勒展开的解法；阶乘极限、数列极限；想办法优化步骤，有些步骤太过冗余。

Some rules have prerequisite conditions for usage, but not all rule prerequisites are comprehensively judged now; add equivalent infinitesimal and Taylor expansion solutions; factorial limits, sequence limits; find ways to optimize steps, some steps are too redundant

8. `mul_split_rule()` 中的相似代码很多，后续还需进一步重构。

There is much similar code in `mul_split_rule()`, further refactoring is needed subsequently.

9. 实现矩阵符号表达式的化简功能。

Implement simplification functionality for matrix symbolic expressions.

10. 完善 `expression_parser.py`，现在好像有一些问题。

Improve the functionality of `epression_parser.py`, as there are currently some issues that need to be addressed.

11. 完善积分部分的规则和匹配器，当前只能处理简单的积分计算，需要扩展更多复杂的积分场景支持。

Improve the rules and matcher, as it currently can only handle simple integral calculations and needs to support more complex integration scenarios.

12. 所有 CSS 和 JS 文件都需要进行重构，以提高代码质量和维护性。

All CSS and JS files need to be refactored to improve code quality and maintainability.

13. 积分规则增加虚调子的双元法。

参考资料 https://zhuanlan.zhihu.com/p/443599480 和 https://zhuanlan.zhihu.com/p/1961384432788314036