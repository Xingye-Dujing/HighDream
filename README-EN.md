# HighDream ([中文描述](./README.md))

## Project Overview

### What is this project?

HighDream is a symbolic computation tool based on **Python** and **SymPy**. It adopts a modular design and uses **breadth-first search (BFS) and rule-driven** approaches to automatically solve mathematical problems including differentiation, integration, limits, matrices, and equivalent transformations, while **providing step-by-step solutions**.

### Why was this project created?

When I was in high school, I struggled with the calculations in solid geometry and conic sections, and I kept thinking it would be great if a computer could help me compute automatically and show the process. I didn't have time to work on it in high school, and I forgot about it in my first year of university. So it was delayed until September this year, when I started building it at the beginning of my sophomore year. After nearly three months of work on and off, I finally finished the basic framework.

### Why is it named HighDream?

As mentioned above, this idea originated from my high school days `(The Dream of High School)`. I looked it up, and `high dream` means an intoxicating dream. Although NetEase Youdao Dictionary suggests it might imply being disconnected from reality, I only take the meaning of intoxication. **I hope that in the future, computers will have strong intelligence and promote the development of human theoretical knowledge.** Therefore, naming it `HighDream` represents both my high school idea and my vision for the future.

## Image Demos (running on Firefox in Debian)

1. Homepage

![Homepage](./docs/img/01.png)

2. Beautiful rendering of mathematical symbols

![Beautiful rendering of mathematical symbols](./docs/img/02.png)

3. Example of intermediate steps for integration

![Example of intermediate steps for integration](./docs/img/03.png)
![Example of intermediate steps for integration](./docs/img/10.png)
![Example of intermediate steps for integration](./docs/img/11.png)
![Example of intermediate steps for integration](./docs/img/12.png)
![Example of intermediate steps for integration](./docs/img/13.png)
![Example of intermediate steps for integration](./docs/img/14.png)

4. Example of intermediate steps for differentiation

![Example of intermediate steps for differentiation](./docs/img/04.png)

5. Example of intermediate steps for limits

![Example of intermediate steps for limits](./docs/img/05.png)
![Example of intermediate steps for limits](./docs/img/06.png)

6. Example of intermediate steps for matrices

![Example of intermediate steps for matrices](./docs/img/07.png)
![Example of intermediate steps for matrices](./docs/img/08.png)
![Example of intermediate steps for matrices](./docs/img/09.png)

7. Example of intermediate steps for equivalent transformations

![Example of intermediate steps for equivalent transformations](./docs/img/15.png)
![Example of intermediate steps for equivalent transformations](./docs/img/16.png)

8. UE-like blueprint usage

![UE-like blueprint usage](./docs/img/17.png)

9. Matrix analysis

![Matrix analysis](./docs/img/18.png)
![Matrix analysis](./docs/img/19.png)
![Matrix analysis](./docs/img/20.png)

10. Simple LaTeX editor and renderer with auto-completion

![LaTeX editor and renderer](./docs/img/21.png)
![LaTeX editor and renderer](./docs/img/22.png)
![LaTeX editor and renderer](./docs/img/23.png)

11. Command line display of function calls for current computation (facilitates problem localization and future extension)

![Command line](./docs/img/24.png)

12. Manual rule selection page (`/rule_select`)(running on Chrome in Windows)

![Manual rule selection page](./docs/img/01.gif)
![Manual rule selection page](./docs/img/29.png)

13. Blueprint-based manual rule selection / exhaustive rule enumeration (`/method_tree`)(running on Chrome in Windows)

    - Right-click drag to pan the canvas
    - Left-click drag a node to move it
    - Drag a node edge to resize it
    - Left-click double-click a node to disconnect its edges
    - Step-by-step confirmation of whether each rule should be applied, with live progress display
    - Supports fully-automatic exhaustive traversal of the whole method tree (async, cancellable)

![Manual rule selection page with blueprint usage](./docs/img/25.png)
![Manual rule selection page with blueprint usage](./docs/img/26.png)
![Manual rule selection page with blueprint usage](./docs/img/27.png)
![Manual rule selection page with blueprint usage](./docs/img/28.png)

## Features

- Step-by-step solutions: displays the complete calculation process, facilitating teaching and understanding
- Rule-based pattern matching: rule application based on pattern matching, no AI involvement, high stability of output results
- Multi-domain support: differentiation, integration, limits, matrix operations, equivalent transformations
- Definite integrals: enter an upper bound (lower bound optional, defaults to 0) in an integral cell to compute a definite integral, showing "find the antiderivative, then apply the Newton-Leibniz formula"; indefinite integrals still append the constant C
- Improper-integral input: integral bounds accept any infinity spelling — oo, ∞, infinity, 负无穷/正无穷/无穷大 — which are normalized to SymPy's ∞ before parsing, so an infinite-endpoint integral (e.g. ∫₁^∞ 1/x²=1, ∫₀^∞ e⁻ˣ=1) now returns its full step-by-step derivation and result; also fixed two rendering glitches — the improper-limit step no longer leaks a literal None, and an integrand like 1/x² no longer shows a spurious leading 1 (the parse_latex unit-coefficient artifact)
- Substitution back-substitution fix (important): a trigonometric substitution introduces a temporary variable carrying real/positive assumptions, but the engine previously rebuilt the parent integral from an assumption-free symbol during write-back, so the substitution silently no-opped, the correct answer was discarded, and a malformed ∫…(u,asin x) shipped to students. It now reuses the operation the engine actually stored (preserving the dummy's identity) and takes the integration variable from the integral's own limits, closing a large class of substitution integrals: ∫1/√(1−x²)=asin x, ∫√(1−x²), ∫1/√(x²±1)→ln|x+√(x²±1)|, ∫1/(x²√(x²−1)), ∫x²/√(1−x²), ∫1/√(x²+2x+3), and more. Both the web path and the manual step-by-step path (/rule_select) are fixed so no unfinished integral leaks, and the trig-substitution note now renders dθ with the actual variable (du/dv).
- Two definite-integral fixes: (1) the interior-pole scan hit periodic denominators (sin/cos/tan) where solveset over Reals returns an infinite solution set, hanging definite integrals (e.g. ∫₀^{π/2} sin/(sin+cos) returned nothing); the candidate search is now confined to Interval.open(a,b) with an iteration cap for unbounded intervals, and symbolic (non-comparable) bounds are skipped safely — so it solves instead of hanging (∫₀^{π/2} sin/(sin+cos)=π/4, ∫₀^{π/2} sin/(1+cos)=ln 2). (2) finite-endpoint improper integrals (e.g. ∫₀¹ ln x) previously returned nan because direct substitution gives S.NaN; the endpoint test now detects any non-finite value and takes a one-sided limit, correctly yielding −1.
- Radical-substitution branch fix: inverting a quadratic radical (e.g. ∫√(x²+2x)) has two x(u) branches; it previously chose the x≤−2 branch, so on the textbook “variable positive” region (x>0) the antiderivative’s derivative was the negative of the integrand (flipped sign). It now takes the branch that grows positive as u→+∞ per the assume-positive convention, giving the correctly-signed antiderivative for √(x²+2x), √(x²−2x), √(x²+1) etc. on x>0.
- Definite-integral symmetry techniques (king/interval-recurrence and parity): two new step-by-step shortcuts that fire only for finite numeric bounds with no interior poles (so they never mask divergence) and are disabled on the manual /rule_select path. (1) Mirror/king property I=∫ₐᵇf(x)dx=∫ₐᵇf(a+b−x)dx; adding gives 2I=∫ₐᵇ[f(x)+f(a+b−x)]dx, and when that sum is a constant c, I=c(b−a)/2, shown step by step (e.g. ∫₀^{π/2} sin/(sin+cos)=π/4 and ∫₀^{π/2} 1/(1+tan³x)=π/4 — the latter previously took ~21 s and is now instant). (2) Parity: an odd integrand on [−A,A] gives 0 directly (e.g. ∫₋₁¹ x³cos x, ∫₋₂² x/(1+x²)).
- Rational-denominator split-field extension: for biquadratic denominators ax⁴+bx²+c (c/a>0) it applies the closed-form real factorization (x²+√(c/a))²−(2√(c/a)−b/a)x² to auto-derive the splitting field √(c/a), √(2√(c/a)−b/a) and partial-fraction over it, folding quartics that are irreducible over ℚ but split over ℝ into two real quadratics (e.g. 1/(x⁴−2x²+2), 1/(x⁴+1), 1/(x⁴+4)) into the existing complete-square/atan/log flow step by step; such denominators previously failed to split and the BFS spun to timeout.
- Partial derivatives: pick any single letter variable as the differentiation variable in a derivative cell; all other symbols are treated as constants
- Extended integral rules: direct antiderivatives for sec/csc/cot squares, power-reduction formulas, and the tan·sec and cot·csc products, avoiding the bloated forms produced by the Weierstrass substitution; even mixed sin^m·cos^n products are power-reduced via multiple-angle identities before termwise integration
- Symbolic matrix simplification: a new linear-algebra mode reducing A·A⁻¹=I, I·A=A, (A⁻¹)⁻¹=A, (AB)⁻¹=B⁻¹A⁻¹, (AB)ᵀ=BᵀAᵀ, A·(B+C)=A·B+A·C step by step (uppercase symbols are treated as n×n matrices)
- Higher-order derivatives: diff(expr, x, n) / dⁿ/dxⁿ solved order-by-order — each order runs the rule engine fully and feeds the previous result back, never a black-box .diff(n).
- New matrix operations: trace tr(A) (usable as a scalar in compositions), null/column/row space (RREF + pivot columns, then a basis and dimension), Moore–Penrose pseudoinverse A⁺ (pinv), Jordan normal form (jordan: A = PJP⁻¹ with J and the transition matrix P), quadratic-form definiteness (definite: Sylvester's leading-principal-minor test → positive/negative/indefinite/semi-definite), and characteristic polynomial (charpoly: p(λ) = det(A − λI), expanded and factored). Matrix-valued results are not silently forced into scalar arithmetic — they yield a friendly note instead.
- Numerical IVP solvers: Euler (euler), improved Euler / Heun (heun), and 4th-order Runge–Kutta (rk4) advance y' = f(x, y) step by step, listing x_k and y_k at every step and the numeric approximation of y(x_end).
- Numerical root-finding: bisection (bisection, needs a sign change on the bracket), secant (secant), and fixed-point iteration (fixedpoint), alongside Newton (newton) — each lists every iterate until convergence (e.g. bisection(x^2-2, x, 1, 2, 8) → 1.414…).
- Complex n-th roots: nthroots(w, n) returns all n roots of w by De Moivre (modulus, argument and the formula z_k = |w|^(1/n)·e^{i(θ+2πk)/n}, then each root for k = 0..n-1).
- Audit fixes: the quotient rule was silently dead (a wrong `is_constant` test) — division now uses it; finite sums Σ_{k=1}^{n} were misrouted to infinite-series convergence and reported "diverges" — they now use closed-form summation (Faulhaber/geometric/telescoping), falling back to convergence only when unbounded; the radius of convergence now honors the summation variable and falls back to the Cauchy–Hadamard root test for oscillating coefficients instead of crashing; a standalone simplify(...) no longer returns an empty result.
- Multivariable & vector calculus: partial derivatives / gradient (gradient), Hessian matrix with the second-partial discriminant D (hessian), Jacobian matrix and determinant (jacobian), directional derivative (directional), curl (curl) and divergence (divergence), constrained extrema via Lagrange multipliers (lagrange); triple integrals over a box (layer by layer), change of integration order for linear-bounded regions (changeorder, auto-reversed to dx-then-dy and split into slabs, verified exactly equal to the original before display), and cylindrical / spherical triple integrals (Jacobian r and ρ²sinφ applied automatically); line integrals — first kind ∫_C f ds (lineint, curve given as '|'-separated parametric components, arc element |r'(t)| rule-derived) and second kind / work ∫ P dx + Q dy (lineint2 / work); surface integrals — first kind ∬_S f dS (surfint, graph z = g(x,y), dS = √(1+g_x²+g_y²)) and second kind / flux ∬_S F·dS (flux, upward orientation −P·g_x − Q·g_y + R). Every partial comes from DiffCalculator and every single/double integral from IntegralCalculator (genuine step-by-step); line/surface results are additionally cross-checked against composite-Simpson quadrature and honestly refused on mismatch rather than emitting a wrong value. Examples: lineint(x^2+y^2, cos(t)|sin(t), t, 0, 2*pi) = 2π, lineint2(y|-x, cos(t)|sin(t), t, 0, 2*pi) = −2π, surfint(1, x+y, x, y, 0, 1, 0, 1) = √3, flux(x|y|z, 1-x-y, x, y, 0, 1, 0, 1) = 1
- Limit-engine enhancements: equivalent-infinitesimal substitution (equivalent_infinitesimal) — for independent factors of a product/quotient with u→0, replace sin u~u, tan u~u, arcsin u~u, arctan u~u, 1−cos u~u²/2, e^u−1~u, ln(1+u)~u, each step labelled "等价无穷小代换". Safety: substitution NEVER happens inside a sum (the sin x−x→0 pitfall), so such differences automatically fall through to L'Hopital (e.g. (sin x−x)/x³ still gives −1/6). The substitution is ranked after the isolated "重要极限" standard-limit matchers (sin x/x, (e^x−1)/x, …) but before the generic split rules, so they never overshadow each other. Separately, the L'Hopital chain now uses a cost-bounded `_simp` (identity cancel/together/expand_trig, adopted only when it does not enlarge the expression, else full simplify) that skips the trigsimp cost repeatedly triggered by the BFS — the slowest notebook-limit cell drops from ~2.5 s to ~1.7 s with every result and intermediate step verified unchanged. Added tests/test_limit_equiv_infinitesimal.py and docs/notebook-equivalent-infinitesimal.json
- Advanced calculus: tangent line / linearization / critical points (extrema) / Newton's method, arc length / solid-of-revolution volume (disk & shell) / surface area, iterated (double) integrals over rectangles (both layers integrated step by step), implicit and parametric differentiation, polar area & slope, Taylor/Maclaurin series (each order's derivative derived by the rule engine), series convergence (n-th-term / ratio / root / Leibniz / p-series tests applied by name), improper integrals (endpoint infinite/singular values are decided by one-sided limits; an interior pole splits the integral into improper sub-integrals tested one by one, and a divergent one is reported honestly instead of blindly applying Newton-Leibniz), limits carrying a constant parameter (the limit variable stays fixed and any other symbol is a constant, e.g. sin(a*x)/x→a, (1-cos(a*x))/x²→a²/2, (1+a/x)^x→e^a), and one-sided limits of absolute-value / floor / sign functions (the |·| is stripped once its argument's sign is fixed on the approach side, e.g. |x|/x → +1 from the right, −1 from the left), ODEs (first-order: separable, homogeneous y'=f(y/x), linear via integrating factor, Bernoulli, exact equations via a potential function, and non-exact equations via a mu(x)/mu(y) integrating factor; second-order and higher: constant-coefficient homogeneous via the characteristic equation — repeated and complex roots included — non-homogeneous via variation of parameters, Cauchy-Euler variable-coefficient homogeneous, and higher-order constant-coefficient homogeneous by factoring the characteristic polynomial over the rationals and pairing each factor's basis, and second-order equations missing y (y''=f(x,y')) reduced to first order via p=y' then re-integrated; all solved step by step, only other forms fall back to dsolve), numerical integration (trapezoid / Simpson), mean value & Rolle's theorem, power-series radius and interval of convergence (endpoint tests) with term-by-term differentiation/integration, and complex numbers (a+bi simplification, polar/exponential form, roots of complex equations), and De Moivre powers
- Universal input box: a new default "Auto-detect (calculus + linear algebra)" mode where you type the whole expression in one box and the program dispatches to every feature by operator/structure — structural forms (d/dx, ∫, ∮, lim, Σ, matrix literals A+B / A*B / lone-matrix row-reduce, bare \det) and function forms (det/rank/inverse/rref/eigenvals/svd/lu/diagonalize/orthogonalize/dependence/project/transpose, diff/int/limit/sum/taylor/tangent/linearization/critical/newton/meanvalue/arclength/volume/shell/surface/implicit/parametric/ode/numerical/complex/demoivre/msimplify/simplify, and matrix arithmetic add/subtract/multiply(A,B)); a plain expression is simplified — no need to pick the type first (Maple-style)
- Universal input box · recursive composition: operators may be embedded in any elementary expression, and the solver reduces them innermost-first — each inner operator is solved by its own step engine, its result value is substituted back, and the process repeats until only elementary arithmetic remains (which is also shown step by step). The whole derivation renders as one align block with nested sub-blocks (outer steps one line each; an operator's own steps are indented under the line that uses it). No step is ever produced by a SymPy black-box shortcut. Examples: 3*diff(x,x), sin(x)+int(x^2,x), diff(int(x^2,x),x), det([[1,2],[3,4]])+rank([[1,0],[0,1]]), volume(x,x,0,1)*3, taylor(e^x,x,0,2)+1. Matrix-valued results (inverse/rref) are not dropped into scalar arithmetic, and pedagogical objects (tangent line, critical points, convergence verdicts) that don't participate in arithmetic yield a friendly note instead of a wrong answer; a lone top-level operator or a symbolic-matrix form (e.g. msimplify(A·B·B⁻¹)) still uses the original fast path, unchanged
- WYSIWYG input: the cell input is an interactive, live-rendered LaTeX editor (MathLive, bundled locally for offline use) — the input IS the rendered math, no separate preview; a template keyboard and symbol palette let you insert fractions, radicals, sub/superscripts, derivatives, definite/indefinite integrals, sums, limits, partial derivatives, trig functions, constants and matrices with one click, each leaving editable placeholders; per-operation templates and default variables are preset for calculus/series; single-matrix linear-algebra problems (determinant/rank/inverse/eigenvalues/RREF/LU/diagonalization/SVD/orthogonalization) render as a real matrix and are pre-seeded with a fillable 2×2 matrix skeleton; two-input problems (matrix operations, solving Ax=b, vector projection, linear-transform/basis-change) provide two matrix editors; equation-list entry for linear systems and sympy-syntax symbolic matrix expressions keep plain-text entry
- Implicit cells (Maple document mode): by default cells have no border, no "Calculator:" prompt and no visible control row, so the page reads as a continuous input→result document; the control row (run / operation type / per-operation parameters / variable / cache / keyboard) floats in above the input only on hover or focus, and the focused cell is marked by a thin left accent bar; pressing Enter in the math editor evaluates the cell and auto-appends a fresh implicit input below (Maple-style), while Ctrl/Cmd+Enter evaluates the plain textarea (preserving newlines); a loaded notebook always ends with an editable empty input
- Notebook export: the top "Actions" menu gains "Export as Markdown / Export as LaTeX / Export as PDF" buttons that walk the cells in document order and write each cell's input plus its full step-by-step derivation into a downloadable .md or .tex file (the LaTeX uses the ctexart class for Chinese and needs XeLaTeX to compile), while PDF opens the browser print dialog (print styles hide the menu, toolbar and rail); every result also carries its own "Export this cell" button that downloads just that cell's input + derivation as Markdown. The raw derivation is stashed at render time and cleared on re-run/error, so stale results are never exported
- Assumptions input: each cell's control row gains an "假设/assumptions" text field (persisted with the notebook). Enter e.g. `x>0, y real` (comma/semicolon/且 separated; supports >0 <0 >=0 <=0 !=0 ==0 and the real/positive/integer/... word forms). Differentiation and (in/def)inite integration append an explicit "依假设化简最终结果: …" step so sqrt(x²)→x, log(x²)/2→log(x), ∫1/x→log(x) match textbook hypotheses. It uses a confined symbol round-trip (swap only the constrained symbols, simplify, swap back), so unconstrained symbols are left alone (sqrt(y²) is NOT wrongly collapsed) and solving-symbol identity / the substitution flow are untouched; a run with assumptions automatically bypasses the result cache (whose key has no assumptions field) to avoid cross-contamination, and malformed input degrades safely to "no assumptions" instead of erroring
- Per-step rule cards: below the result of differentiation / integration / limits a collapsed "分步规则卡片" panel reuses the existing /api/steps structured-steps endpoint to split the whole derivation into individual cards (each card's header states the rule/explanation used at that step, its body is that step's math) for step-by-step focus. Purely front-end — the original full align block above stays unchanged (so export is unaffected) and no backend change is needed
- Function plotting (SVG): for results whose expression is a single-variable scalar function (differentiation / integration / limits / tangent / linearization / arc length / solids of revolution, …) a collapsed "绘制函数图像 y=f(x)" region appears below the output; opening it calls the new /api/plot endpoint, which samples y=f(x) on a fixed grid and returns plain numbers for an inline SVG polyline chart (axes / zero line / ticks), overlaying the tangent line + point for tangent results. The chart supports wheel zoom, drag panning and double-click reset, plus a one-click show/hide grid button; sampling also locates approximate real roots (sign-change bisection + local |f| minima) and labels their coordinates, with tolerances relative to the curve's own magnitude so a uniformly tiny but non-zero function is not flooded with false roots and asymptotes are not mistaken for roots. Sampling is server-side with sympify + subs only — never eval/exec/lambdify — with hard caps on sample count and |y|; asymptotes and out-of-domain points come back as null breaks (e.g. 1/x at 0, log(x) on the negative axis) so the client breaks the polyline. The SVG lives inside the scrollable .math-output, so it neither regresses existing rendering nor blocks the fullscreen view
- History snapshots and shortcuts: auto-save is gone; the top "Actions" menu gains "历史快照" (history snapshots), which stores the current notebook in a server-side SQLite database (cache/snapshots.db) on demand and lets the user manually restore any earlier snapshot (the current notebook is saved as a safety snapshot first); Ctrl/⌘+S still saves the notebook as a JSON file
- Symbol / template palette (formerly the "virtual keyboard"): it has no separate title bar and is no longer a floating side rail — it is a column inside the page body. The left content and the right palette share one body container and scroll independently, and the palette never grows taller than the body; keys are separated by hairlines, every key and input follows the global corner-radius setting, and the palette shares the paper background with the content. Besides the basic tokens (numbers / operators / variables & constants / structure templates / functions) it is organized by capability to cover every computation feature — expression simplification, single-variable calculus (derivative / indefinite & definite integral / limit / improper integral / Taylor & remainder / tangent / linearization / extrema / Newton / mean value / arc length / volumes & surface), multi-variable calculus (partial / double integral / implicit / parametric / polar area & slope — kept separate from single-variable), series, ODEs, numerical methods, complex numbers, and linear algebra (single-matrix operations plus matrix add/subtract/multiply); each key renders and inserts its real operator template with a visible □ box at every fillable slot (e.g. derivative d/d□(□), definite integral ∫(□)d□, limit lim_□→□(□), function sin(□), sub/superscript □ⁿ/□); after a key inserts a template the caret auto-positions on the first □ and typing replaces that box (likewise, clicking right after any □ then typing replaces it), guiding step-by-step filling; the multi-input problem types (solve Ax=b / system-to-matrix / basis & coordinate transform / linear-transform change of basis) switch the cell to the matching operation type and pre-seed its matrix editors; the matrix panel, matrix lab and matrix analysis pages share the same body layout: the panel palette offers matrix input tokens and structure templates, the lab palette offers executable matrix operations (transpose / inverse / rref / determinant / rank / trace / row-column ops / add-subtract-multiply) rendered as math, and the analysis page uses the same WYSIWYG matrix editor. All matrix inputs accept both bmatrix and [[...]] forms, and empty cells, unfilled slots and dimension mismatches are rejected with a clear message instead of being sent to the engine
- Input symmetry: structural LaTeX from the WYSIWYG editor (d/d□, ∫, lim, Σ, Π, det, inverse/transpose, …) is normalized back to the matching operation before solving. Whether the selector is "auto" or a concrete "derivative / integral / limit", a template is applied exactly once — no silent second derivative, double integral, or a limit parsed as a symbol. Unrecognized LaTeX, unfilled slots, non-square matrices and dimension mismatches return a clear error instead of a confident wrong answer
- Theme & unified dropdowns: the whole product uses the ZEN-OMM theme (light/dark, with corner radius and UI font size adjustable from the top menu), fronted by a slide-out product menu bar (View / Tools / Reference groups, with each page's action buttons folded into an "Actions" group); every dropdown (operation type, limit direction, operation kind, determinant type, basis/coordinate transform, …) is replaced by a custom omm-style popup panel — the native `<select>` stays as the data model while the popup is mounted on `<body>` and positioned with `fixed`, auto-flipping vertically and clamping horizontally to stay on-screen, escaping container `overflow` clipping, and stacking above the virtual keyboard / fullscreen overlays (z-index 4000) so it is never covered
- LaTeX output: beautiful rendering of mathematical formulas
- Extensible architecture: easy to add new calculation rules and operation types
- Two ways to use: Python code calls in Jupyter Notebook; graphical web interface
- Manual rule selection: via `/rule_select` page, users can manually choose which rule to apply at each derivation step for differentiation, integration, and limits; a new "Undo" button rolls back the last chosen rule (undo is per whole user action via a snapshot stack, since one rule application can produce several derivation steps; the history is bounded)
- Exhaustive rule-enumeration method tree: via `/method_tree` page, all applicable rule branches at each step can be enumerated and displayed as a blueprint canvas, making the full solving space easy to inspect; double-clicking any node expands THAT branch (node-to-root) into a single linear step-by-step derivation in a docked panel, reusing the exact notebook align rendering and without re-running the whole search

## Development Tech Stack

- Programming language: Python 3.11.6
- Core libraries: SymPy (symbolic computation library), Flask (web visualization)
- Output format: LaTeX (rendered by MathJax)

## Architecture Design

This project mainly consists of **four** architectures:

1. Architecture for differentiation, integration, limits, and determinant calculations (`BaseCalculator` + `BaseStepGenerator` + `RuleRegistry`)
2. Architecture for matrix-related operations (`CommonMatrixCalculator` + `MatrixStepGenerator`)
3. Architecture for frontend to call backend for computation (Flask blueprints + `task_manager` async tasks)
4. Architecture for manual rule selection and exhaustive rule-enumeration method tree (`BaseManualStepSolver` with its per-domain subclasses, and `MethodTreeEnumerator`)

## Core Classes

- [**BaseCalculator**](https://high-dream.vercel.app/help/base_calculator) (`core/base_calculator.py`)
- [**BaseStepGenerator**](https://high-dream.vercel.app/help/base_step_generator) (`core/base_step_generator.py`)
- [**RuleRegistry**](https://high-dream.vercel.app/help/rule_registry) (`core/rule_registry.py`)
- [**CommonMatrixCalculator**](https://high-dream.vercel.app/help/common_matrix_calculator) (`core/common_matrix_calculator.py`)
- [**MatrixStepGenerator**](https://high-dream.vercel.app/help/matrix_step_generator) (`core/matrix_step_generator.py`)
- [**BaseManualStepSolver**](https://high-dream.vercel.app/help/base_manual_step_solver) — Base orchestrator for manual rule selection, encapsulating shared selection/rollback/state-management logic across domains
- [**MethodTreeEnumerator**](https://high-dream.vercel.app/help/method_tree_enumerator) — Core enumerator for the exhaustive method tree, recursively expanding every applicable rule branch

## Usage

- Python code calls: please refer to the examples in `main.ipynb` first. (Remember to install third-party libraries)
- Visual website:

  (1) Run `python app_local.py` (requires Python installation)

  (2) Run `high-dream.exe` (download from Releases page)

  (3) Visit `https://high-dream.vercel.app/` (for experience only, computation speed is slow)

## Notes on Usage

0. Do not close the command line (black window) while using.
1. The multiplication operator `*` must never be omitted! `ab` will be parsed as a single symbol, not `a*b`.
2. Inverse trigonometric functions are `asin()`, `acos()`, `atan()`, not `arcsin()`, `arccos()`, `arctan()`.
3. Do not use `e**` or `e^`; use `exp()` instead. `e` will be parsed as a letter variable like `x`. `exp()` is the properly defined natural exponential function.

## For More Details, See the [Help Documentation](https://high-dream.vercel.app/help)

## Special Thanks

1. Heavily referenced [MathDF](https://mathdf.com/int/cn/). It is a god that can provide intermediate steps. This project has made some extensions based on the types of integrals that it cannot solve (such as the `f(x)*e^g(x)` type).
2. **AI** participated throughout the process. Sincere thanks to various free AI tools in China (such as Qianwen and DeepSeek). Postscript: Since Qoder offers 200 free Q&A sessions per day, I started using the Qoder CLI on June 19, 2026. When it has no free quota, that's when I will no longer use Qoder CLI. The usage of Qoder CLI ends on June 22.
