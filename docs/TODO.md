# TODO List

> 状态图例 Legend: ✅ 已完成 done · 🔶 部分完成 partial · ⬜ 待办 todo
> 最近一次代码核对 (code audit): 2026-10-17

1. ✅ SymPy 显式计算所有幂运算，这会导致指数较大时出现性能问题。我们需要特殊处理来避免直接计算幂值。

SymPy evaluates all powers explicitly, which causes performance issues when exponents are large. We need special handling to avoid computing power values directly.

> 已实现: `utils/cache_utils.cached_sympify` 对含大整数幂的字面量在 `evaluate(False)` 下解析, 由 `config.LARGE_POWER_EXPONENT_DIGITS` 触发, 巨值不再被物化。
> Done: huge-power literals are parsed under `evaluate(False)` (guard in `cached_sympify`, threshold in `config`).

2. ✅ 每个计算过程部分应包含链接区域。点击链接区域应该快速生成一个或多个相关的计算单元格，允许用户验证那些直接给出而未显示过程的结果。

Each calculation process section should include a link area. Clicking the link area should quickly generate one or more related calculation cells, allowing users to verify results that are directly given without showing the process.

> 已实现: `static/script.js` 的 `attachStepLinks` + `/api/steps` (`core/step_tree.structured_solve`) 为 diff/integral/limit 结果生成可点击的分步链接与规则卡片。
> Done: `attachStepLinks` / `/api/steps` produce clickable per-step links and rule cards.

4. ⬜ 将 `script.js` 中的 HTML 标签提出到外部文件，`script.js` 进行导入而不是内嵌。这样便于网页的样式设计。

Extract the HTML tags from `script.js` into external files, and have `script.js` import them instead of embedding them inline. This facilitates web page styling design.

> 待办(重构): 属结构性重构, 牵涉模板加载顺序与大量内联模板字符串, 风险较高, 未做。
> Pending (refactor): touches template load order and many inline strings; not attempted to avoid regressions.

5. ✅ 求导相关：支持对含有不确定量但不确定量是常数的函数求导，结果用不确定量表示（偏导的一种特例）；支持求偏导。

Support differentiation of functions containing uncertain quantities where the uncertain quantity is a constant, with results expressed using the uncertain quantity (a special case of partial derivatives); support partial derivatives.

> 已实现: `DiffCalculator` 设 `_allow_multi_symbols = True`, 固定 `variable`、其余自由符号按常数处理即得偏导; 另有 `implicit_diff`/`parametric_diff` 应用算子。
> Done: `_allow_multi_symbols` keeps the requested variable and treats other symbols as constants (partial derivatives).

6. ✅ 积分相关：支持定积分计算（增加上下限参数）；支持多重积分。

Support definite integral calculation (add upper and limit parameters); support multiple integrals.

> 已实现: 前端定积分上下限入参 + `IntegralCalculator` 上下限; 二重(矩形)积分见 `domains/applied/integral_apps.iterated_integral` (算子 `iterated_integral`)。
> Done: definite integrals (bounds) + `iterated_integral` for rectangular double integrals.

7. 🔶 极限相关：有些规则的使用具有前提条件，但现在并非所有规则的前提条件都已全面判断；增加等价无穷小和泰勒展开的解法；阶乘极限、数列极限；想办法优化步骤，有些步骤太过冗余。

Some rules have prerequisite conditions for usage, but not all rule prerequisites are comprehensively judged now; add equivalent infinitesimal and Taylor expansion solutions; factorial limits, sequence limits; find ways to optimize steps, some steps are too redundant.

> 部分完成: 等价无穷小标准极限规则已就位 (`sin_over_x`, `ln_one_plus_x_over_x`, `exp_minus_one_over_x`, `g_over_sin`, `g_over_ln_one_plus`, `g_over_exp_minus_one`); 阶乘/数列极限与高阶 (如 `(arctan x - sin x)/x^3`) 经现有引擎可正确求解 (实测通过)。仍待办: 逐规则前提条件的全面判定、步骤冗余优化、显式泰勒展开解法。
> Partial: equivalent-infinitesimal rules present; factorial/sequence/higher-order limits verified to solve. Remaining: full per-rule prerequisite checks, step-redundancy cleanup, explicit Taylor-expansion method.

8. ⬜ `mul_split_rule()` 中的相似代码很多，后续还需进一步重构。

There is much similar code in `mul_split_rule()`, further refactoring is needed subsequently.

> 待办(重构): `domains/limit/rules/base_rules.mul_split_rule` 仍有重复分支; 纯重构, 未做。
> Pending (refactor): duplicated branches remain; not attempted.

9. ✅ 实现矩阵符号表达式的化简功能。

Implement simplification functionality for matrix symbolic expressions.

> 已实现: `domains/matrix/matrix_symbolic.simplify_matrix_symbolic`, 算子 `matrix_symbolic` (别名 `msimplify`/`matrixsimplify`/`symmatrix`)。
> Done: `simplify_matrix_symbolic` exposed as the `matrix_symbolic` op.

10. 🔶 完善 `expression_parser.py`，现在好像有一些问题。

Improve the functionality of `expression_parser.py`, as there are currently some issues that need to be addressed.

> 部分完成: 已修复 `expr`(等价表达式) 对 MathLive LaTeX 输入直接 `SympifyError` 崩溃的问题 (改用 `cached_sympify`, 见提交 "等价表达式(expr)对所见即所得输入直接报错"), 并补回归测试。仍待查: 积化和差仅覆盖 sin/cos 两因子、`xreplace` 对重复子式的边界等潜在问题。
> Partial: fixed the expr-op crash on LaTeX input (now via `cached_sympify`, with a regression test). Remaining: product-to-sum covers only two sin/cos factors; possible `xreplace` edge cases.

11. ⬜ 完善积分部分的规则和匹配器，当前只能处理简单的积分计算，需要扩展更多复杂的积分场景支持。

Improve the rules and matcher, as it currently can only handle simple integral calculations and needs to support more complex integration scenarios.

> 待办(功能扩展, 开放): 引擎已能解大量换元/分部/有理/三角积分 (见 docs 笔记本与测试), 但"更多复杂场景"是持续性扩展, 无固定边界, 未做。
> Pending (open-ended feature expansion).

12. ⬜ 所有 CSS 和 JS 文件都需要进行重构，以提高代码质量和维护性。

All CSS and JS files need to be refactored to improve code quality and maintainability.

> 待办(重构): 大范围重构, 风险高, 未做。已就地的安全改动: 各页面 JS 的 innerHTML 转义 (见提交 "各独立页面 JS 同类 innerHTML XSS")。
> Pending (large refactor). A safe in-place change already done: innerHTML escaping across page scripts.

13. ⬜ 积分规则增加虚调子的双元法。

参考资料 https://zhuanlan.zhihu.com/p/443599480 和 https://zhuanlan.zhihu.com/p/1961384432788314036

> 待办(新功能): 需实现特定积分技法并接入规则引擎, 属较大功能开发, 未做。
> Pending (new solver technique + engine integration).

14. ⬜ 手动修改完善 2026 年 6 月 19 日及之后的帮助页面的内容，现在是纯 AI 生成。

Manually modify and improve the content of the help pages for June 19, 2026, and beyond, which is currently generated entirely by AI.

> 待办(内容): 需人工撰写, 非代码任务。
> Pending (manual content authoring).

15. ⬜ 手动重构优化 2026 年 6 月 19 日及之后 AI 生成的代码。

Manually refactor and optimize the code generated by AI for June 19, 2026, and beyond.

> 待办(重构): 大范围人工重构, 未做。
> Pending (manual refactor).
