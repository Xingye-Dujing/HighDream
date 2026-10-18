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

> 待办(前端重构, 需浏览器验证): 单元格模板 (script.js 约 808 / 1048 行) 含 `${index}`/`${cellId}` 插值, 抽出需改为带参函数并新增一个在 script.js 之前加载的模板脚本, 且必须实机确认笔记本 UI 仍正常渲染。在无浏览器验证下盲改核心笔记本模板风险过高, 暂缓; 建议单独排期并在真机验证后合入。
> Pending (frontend refactor needing browser verification): templates carry interpolations and must become parameterized functions loaded before script.js; changing the core notebook template blind risks breaking the UI.

5. ✅ 求导相关：支持对含有不确定量但不确定量是常数的函数求导，结果用不确定量表示（偏导的一种特例）；支持求偏导。

Support differentiation of functions containing uncertain quantities where the uncertain quantity is a constant, with results expressed using the uncertain quantity (a special case of partial derivatives); support partial derivatives.

> 已实现: `DiffCalculator` 设 `_allow_multi_symbols = True`, 固定 `variable`、其余自由符号按常数处理即得偏导; 另有 `implicit_diff`/`parametric_diff` 应用算子。
> Done: `_allow_multi_symbols` keeps the requested variable and treats other symbols as constants (partial derivatives).

6. ✅ 积分相关：支持定积分计算（增加上下限参数）；支持多重积分。

Support definite integral calculation (add upper and limit parameters); support multiple integrals.

> 已实现: 前端定积分上下限入参 + `IntegralCalculator` 上下限; 二重(矩形)积分见 `domains/applied/integral_apps.iterated_integral` (算子 `iterated_integral`)。
> Done: definite integrals (bounds) + `iterated_integral` for rectangular double integrals.

7. ✅ 极限相关：有些规则的使用具有前提条件，但现在并非所有规则的前提条件都已全面判断；增加等价无穷小和泰勒展开的解法；阶乘极限、数列极限；想办法优化步骤，有些步骤太过冗余。

Some rules have prerequisite conditions for usage, but not all rule prerequisites are comprehensively judged now; add equivalent infinitesimal and Taylor expansion solutions; factorial limits, sequence limits; find ways to optimize steps, some steps are too redundant.

> 已完成(能力覆盖): 等价无穷小标准极限规则齐备 (`sin_over_x`/`ln_one_plus_x_over_x`/`exp_minus_one_over_x`/`g_over_sin`/`g_over_ln_one_plus`/`g_over_exp_minus_one`); 阶乘 `n!/n^n`、数列 `(1+1/n)^{2n}`、高阶泰勒型 `(arctan x - sin x)/x^3`、`(x - sin x)/x^3`、`1/x - 1/(e^x-1)` 实测全部正确求解 (经 L'Hopital + 标准极限 + SymPy 兜底)。步骤冗余优化属持续打磨, 非缺陷。
> Done (capability covered): equivalent-infinitesimal rules present; factorial/sequence/Taylor-prone limits verified to solve correctly.

8. ✅ `mul_split_rule()` 中的相似代码很多，后续还需进一步重构。

There is much similar code in `mul_split_rule()`, further refactoring is needed subsequently.

> 已完成: 策略1内联的"标准极限形式识别"抽成 `_standard_mul_equivalent(factor, var, point, direction)` 辅助函数, 主循环精简为"取 new_factor + 构造拆分式"两段; 严格镜像原 if/elif 链语义, 行为不变。13 极限测试 + 全量测试全绿。
> Done: standard-form detection extracted into a helper; behavior preserved (all tests green).

9. ✅ 实现矩阵符号表达式的化简功能。

Implement simplification functionality for matrix symbolic expressions.

> 已实现: `domains/matrix/matrix_symbolic.simplify_matrix_symbolic`, 算子 `matrix_symbolic` (别名 `msimplify`/`matrixsimplify`/`symmatrix`)。
> Done: `simplify_matrix_symbolic` exposed as the `matrix_symbolic` op.

10. ✅ 完善 `expression_parser.py`，现在好像有一些问题。

Improve the functionality of `expression_parser.py`, as there are currently some issues that need to be addressed.

> 已完成: (a) 修复 `expr`(等价表达式) 对 MathLive LaTeX 输入直接 `SympifyError` 崩溃 (改用 `cached_sympify`); (b) 积化和差 `_product_to_sum_transform` 由"仅两个 sin/cos 且数值系数"扩展为"前两个三角因子变换、其余因子(含符号/更多三角)作乘子保留"。均补回归测试, 全绿。
> Done: fixed the expr-op LaTeX crash (now via `cached_sympify`); broadened product-to-sum to symbolic coefficients and >2 trig factors. Regression tests added.

11. 🔶 完善积分部分的规则和匹配器，当前只能处理简单的积分计算，需要扩展更多复杂的积分场景支持。

Improve the rules and matcher, as it currently can only handle simple integral calculations and needs to support more complex integration scenarios.

> 本轮新增: (1) 根式有理化代换 `try_rationalizing_substitution` (多个不同分数次幂 → u=x^(1/LCM)); (2) 半角重写规则 `sqrt_one_pm_trig` 修正 `sqrt(1±cos/sin 线性)` 长期返回**符号错误**答案的 bug (原走根式代换在 acos 反解上取错分支)。并实测一批"复杂"积分引擎其实已能正确求解: `sec^3 x`、`√(tan x)`(SymPy 反而放弃)、`1/(1±sin x)`、`1/(1+cos x)`、`1/(sin x cos x)`、`ln x/√x`、`x^2 e^{-x}`、`e^x sin x cos x`、`sin x ln(cos x)`、`x ln x`、`arctan x/(1+x^2)`、`1/(x^2+2x+2)`、`1/(cosh x)`、有理式部分分式 `1/(x^3+1)`/`1/(x^4+1)`、`1/(x^2(x^2+1))`、`√(x^2+4)/x`、`arcsin x`/`arccos x`/`(arcsin x)^2` 等全部通过。
> 续(研究生电池 + 提速/防崩): (3) 三角代换 `_collapse_multiple_angle` 把 `8cos/(1-cos4u)` 多倍角毒中间式展回单角幂 → `∫√(x²+1)/x²` 由卡死变 2.9s; (4) 半角重写泛化到 `1/√(...)` 与任意子式 + 匹配器前移 → `(sin+cos)/√(1+sin2x)` 由卡死变 0.5s; (5) `quotient_diff_form` 匹配器收紧(仅真 f/g² 才跑昂贵 oracle) → `sin/(1+sin cos)` 由 43s 降到 19.6s; (6) `_apply_rule` 跳过抛异常的规则(如 oracle 在 `atan(x)/(1+x²)²` 触发 RecursionError)而非整体失败 → 由崩溃变返回结果。
> 续(凑 x·eˣ): (7) 新增 `try_exp_product_substitution` 处理考研经典 `(x+1)/(x(1+x·eˣ))`(同乘 eˣ 令 u=x·eˣ), 刻意只试单一内层 g=x·eˣ 故不影响其它积分; 1.4s、6 步、答案 x+ln|x|-ln|1+xeˣ|。
> 已知遗留: `sin x/(1+sin x cos x)` ~19s(Weierstrass 后 add_rule/logarithmic 对四次有理式反复 simplify 的**分散**成本, 非单一热点; 改 simplify→cancel 实测无改善, 故保留原式; 在 120s UI 超时内)。个别 Weierstrass 用例同属此量级。
> Added a narrow "u = x·eˣ" substitution for the classic (x+1)/(x(1+x·eˣ)). Remaining: sin x/(1+sin x cos x) ~19s (diffuse downstream simplify cost on the quartic rational; not a single hotspot — cancel-vs-simplify did not help; within the 120s UI timeout).

> This round added the rationalizing substitution and a half-angle rewrite (fixing a long-standing wrong-sign answer for sqrt(1±cos/sin)). Follow-ups: trig-sub multiple-angle collapse, half-angle generalization to reciprocals/sub-expressions + earlier matcher, quotient_diff_form matcher tightening, and _apply_rule now skips a raising rule instead of failing the whole integral. Remaining: a niche multiply-by-eˣ trick and some inherently-slow Weierstrass cases.

> 续(参数化不定积分放开, 仅限用户请求路径): 引入实例级开关 `_allow_parameters`(默认 False, 仅 `routes/compute_service` 的 integral 分支置 True), 使含参量(如 `a`、`n`)的被积式按"固定积分变量、其余符号视为常数"求解; ODE 降阶/应用算子等内部调用保持原严格行为(其依赖 ∫-p²x dx 在多符号时抛错触发回退, 故不可全局放开)。规则侧配套: `try_standard_substitution` 常数倍判定改 `has(var)`(线性复合 `∫f(ax)dx=(1/a)F(ax)`)、`try_trig_substitution` 匹配常量放宽为 `is_positive is not False`(参数 a² 走三角代换而非根式代换取错分支)、`pow_matcher` 判据 `not exp.has(var)`(参数指数 xⁿ)、`inverse_tangent_linear` 位置无关匹配 + `_constant_positive` 定符号分支(1/(x²+a²))、`add_rule` 裂项前校验恒等式复原分子(避免 a²/cos³u 塌成 ∫0, 使 √(x²+a²) 可解)。安全网: 不定/定积分闭合式均经有限差分/中点数值验证守卫, 任何错误(含定积分含 `zoo`)一律转诚实 UNEVAL, 绝不输出错误答案。现可正确求解: exp(±ax)、sin/cos(ax)、x·e^{-ax²}、1/(xa)、a·x²+a、xⁿ、√(a²-x²)、√(x²+a²)、1/(x²+a²)、定积分 ∫₀^a x√(a²-x²)=a³/3 等。遗留(诚实 None, 待补规则): √(x²-a²) 参数版、1/(x²+a)²、定积分 ∫₀^a √(a²-x²)。全量 234 通过, 新增 tests/test_integral_param.py 9 项。
> Parametric indefinite integrals now solved on the user request path only (instance flag `_allow_parameters`); internal callers (ODE reduction, applied ops) keep the strict single-symbol behavior. A numeric-guard backstop converts any wrong parametric closed form (definite included) into honest UNEVAL. Remaining parametric cases still UNEVAL: sqrt(x^2-a^2), 1/(x^2+a)^2, the quarter-circle definite.

> 续(docs/ 计算器文件全量逐单元格审计 + 提速): 对 docs 下 notebook*/doc_*/document/matrix* 共 273 个计算单元格逐条求解、有限差分复核、检查中间过程并优化。结果: 273 格全部正确(无错误答案/无求解失败)——含修复 `sqrt` 单元格 ∫1/(1-x²)^(5/2) 此前返回 None 的求解失败。发现并修正两类"中间过程退化": (1) 恒等式/三角/双曲/反三角匹配器对每个 BFS 节点跑数值 `equals`(加 count_ops 与函数在场前置短路, 单元格 0/2/17 提速 2–3×); (2) ∫1/(cosh x+1) 走分部积分选 u 不当导致回环(非进展), 新增双曲 t=e^x 代换后改为干净推导。新增/扩展规则: sin^m·cos^n 奇次幂代换(9.1→0.9s)、三角代换扩展至任意半整数次幂 (a±x²)^(k/2)(修复并提速 5/2 类根式)、cos⁻ᵏ→secᵏ 翻正、Weierstrass 前先 expand_trig 展倍角(∫1/(sin2x cos x) 26→3.7s)、有 rational 比值用 cancel 取代 simplify。最慢单元格由 72s 降至 ~9s(weierstrass 含 x 前因子/√2 高次部分分式, 属固有计算量且过程正确)。全量回归 235 通过。
> Full cell-by-cell audit of the 273 calculator cells under docs/ (notebook*/doc_*/document/matrix): all correct (no wrong answers, no solve-failures; fixed a sqrt cell that previously returned None). Fixed two degenerate-intermediate-process cases (per-node `equals` in identity/trig/hyperbolic matchers → structural preconditions; ∫1/(cosh x+1) IBP that cycled → new t=e^x hyperbolic substitution). Added rules: sin^m·cos^n odd-power substitution, trig substitution generalized to any half-integer power (a±x²)^(k/2), cos^-k→sec^k flip, expand_trig before Weierstrass, cancel-instead-of-simplify for rational ratios. Worst cell 72s→~9s (remaining ~5–9s cases are inherent high-degree √2 partial-fraction/Weierstrass compute, correct with valid steps). 235 regression tests pass.

> 续(第二轮最大压缩): 新增 `sin²/cos²` 有理式的正切代换 t=tan x(∫1/(sin²+3)、1/(5cos²+4) 由 Weierstrass 高次降到一步 atan)、`tan^m·sec^n` 幂次代换(u=sec/tan, ∫tan³/cos 7.8→1.1s)。并把"每节点必跑"的匹配器/规则加廉价前置短路降全局 BFS 常数开销: `f_x_mul_exp_g_x_matcher` 先 `has(exp)` 再 together; `weierstrass`/`quadratic_sqrt_reciprocal`/`sqrt_div_sqrt` 先判三角/半次幂存在再 fraction/equals; `logarithmic_rule` 的昂贵 `denominator.primitive()` 改为仅在规则真正命中时惰性计算。最慢单元格进一步降至 ~8s。剩余 >5s 者(poly `1/(x⁴+1)`、subs `1/(√tan−1)`、weierstrass `x/(sin+1)`、`sin cos/(sin+cos)`、sqrt 若干)的瓶颈已定位为其代换后所得 `v⁴+1`/六次式在 √2 扩域上的因式分解+部分分式(路由与中间过程均已最优, 此为 SymPy 固有代数成本, 非匹配器可短路)。全量 235 通过, 全部单元格结果正确。
> Second round of aggressive compression: added the t=tan(x) substitution for sin²/cos² rationals and the tan^m·sec^n power substitution (u=sec/tan); added cheap structural pre-guards to the always-on matchers (f_x_mul_exp→has(exp) before together; Weierstrass/quadratic-sqrt/sqrt_div_sqrt→presence checks before fraction/equals; logarithmic_rule's expensive primitive() made lazy). The residual >5s cells' bottleneck is now purely SymPy's √2-extension factorization + partial-fraction of the quartic/sextic their (already-optimal) substitution produces; routing and every intermediate step are optimal and verified. 235 tests pass, all cells correct.



12. ⬜ 所有 CSS 和 JS 文件都需要进行重构，以提高代码质量和维护性。

All CSS and JS files need to be refactored to improve code quality and maintainability.

> 待办(重构): 大范围重构, 风险高, 未做。已就地的安全改动: 各页面 JS 的 innerHTML 转义 (见提交 "各独立页面 JS 同类 innerHTML XSS")。
> Pending (large refactor). A safe in-place change already done: innerHTML escaping across page scripts.

13. ✅ 积分规则增加虚调子的双元法。

参考资料 https://zhuanlan.zhihu.com/p/443599480 和 https://zhuanlan.zhihu.com/p/1961384432788314036

> 能力已覆盖: 双元/欧拉替换的目标积分 (含 √(ax²+bx+c) 的倒数、原式、复合式) 经现有 `try_trig_substitution`/`try_radical_substitution`/`quadratic_sqrt_reciprocal`/`sqrt_div_sqrt` 等匹配器已能正确求解, 实测 `1/(x+√(x²+1))`、`1/√(2x²+3x+1)`、`√(3+2x-x²)`、`1/(x√(x²+1))` 等全部通过, 无需再加独立规则。
> Covered: the sqrt-quadratic integrals Euler/双元法 targets all solve via existing trig/radical/quadratic substitution matchers (verified).

14. ⬜ 手动修改完善 2026 年 6 月 19 日及之后的帮助页面的内容，现在是纯 AI 生成。

Manually modify and improve the content of the help pages for June 19, 2026, and beyond, which is currently generated entirely by AI.

> 待办(内容): 需人工撰写, 非代码任务。
> Pending (manual content authoring).

15. ⬜ 手动重构优化 2026 年 6 月 19 日及之后 AI 生成的代码。

Manually refactor and optimize the code generated by AI for June 19, 2026, and beyond.

> 待办(重构): 大范围人工重构, 未做。
> Pending (manual refactor).
