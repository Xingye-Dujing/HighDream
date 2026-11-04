# AGENTS.md

Guidance for AI agents working in this repository.

## Running the app

- **Local dev (opens http://127.0.0.1:5000/):** `python app_local.py` — uses Werkzeug's threaded WSGI server (one thread per request), opens browser automatically. *(gevent's greenlet server froze every request while one handler blocked, e.g. the SSE endpoint waiting on a task event — do not go back to it.)*
- **Vercel entrypoint:** `python app.py` — plain `Flask.run()` (for local testing of deploy path).
- **Build executable:** `pyinstaller app_local.spec` → `dist/high-dream.exe`. Do NOT raise `optimize` above `1` (spec pins it); `2` breaks SymPy.
- **Dependencies:** `requirements_local.txt` for dev, `requirements_deployment.txt` for Vercel (just `sympy`, `flask`).

## SymPy gotchas (enforce in generated code/examples)

- `*` is never implicit: `ab` parses as one symbol, not `a*b`.
- Inverse trig is `asin`, `acos`, `atan` — NOT `arcsin`/`arccos`/`arctan`.
- Natural exponential is `exp()`. `e` alone is just a symbol; `e**x` / `e^x` ≠ `exp(x)`.

## Architecture

- `core/` — BFS + rule-driven solver engine: `BaseCalculator`, `BaseStepGenerator`, `RuleRegistry`, `CommonMatrixCalculator`, `MatrixStepGenerator`, `BaseManualStepSolver`, `MethodTreeEnumerator`. `ManualStepSolver` is a **factory function** (`core/manual_step_solver.py`), not a class.
- `domains/` — per-domain rule sets: `differentiation/`, `integral/`, `limit/`, `matrix/`, plus `expression_parser.py`.
- `routes/` — Flask blueprints: `main.py` (pages), `api.py` (a thin aggregator importing focused route modules: `compute_api.py`, `matrix_api.py`, `sympy_api.py`, `manual_api.py`, `method_tree_api.py`, `snapshot_api.py`, all registering on `api_blueprint.py`). Async tasks via `task_manager.py` (in-memory `ThreadPoolExecutor`). Compute dispatch in `compute_service.py`: `start_compute()` routes via an if/elif chain (the real dispatcher); the `_calculators` / `_cached_calculators` dicts are just operation-name → calculator-class lookups. Runtime op strings (`det`, `transform-1`, …) and some dict keys (`det_1`, `transform_1`) intentionally differ — don't assume the dict drives routing.
- `core/operator_registry.py` — declarative operation tables (`_AUTO_FUNCS`, matrix op sets, text-input ops, square-only ops). `core/latex_syntax.py` — structural LaTeX → operator-call normalization and operator-call scanning. `core/input_validation.py` — WYSIWYG/text/matrix normalization and user-facing error mapping. `utils/expression_utils.py` — generic balanced-parenthesis / top-level-split helpers shared by parsers and matrix code.
- `utils/` — LaTeX rendering (`latex_formatter.py`) and expression helpers.
- `config.py` — central tunables: parser depth, L'Hôpital count, method-tree cutoffs (defaults and hard limits), task timeouts.
- `app_local.py` / `app.py` both call `create_app()` which registers `main` and `api` blueprints.

## Lint & auto-format

- **Pylint** only (`.pylintrc` disables many pedantic rules). Run `.venv/Scripts/pylint <file>`.
- A PostToolUse hook in `.qoder/settings.json` auto-runs `.venv/Scripts/pylint.exe` on every `.py` file after editing (a pylint failure blocks the edit).
- **`.pylintrc` is gitignored** — a fresh clone won't have it; recreate it or the hook re-flags rules the project intentionally allows.
- No auto-formatter. Match existing 4-space PEP-8-ish style.
- Rule subpackages under `domains/*/rules/` carry an `__init__.py`; without one pylint treats that dir as the source root and false-flags top-level `core`/`utils` as import-errors, blocking the edit hook.

## Help documentation (`/help/*`)

- **Author in:** `help_docs/src/*.md`
- **Generate HTML:** `python help_docs/build_help.py html` (requires pandoc ≥3 on `PATH`)
- **Never hand-edit** `templates/help/*.html` — changes are clobbered on next build.
- Both `.md` and generated `.html` are git-tracked (Vercel serves the HTML directly); commit the pair together.
- Existing `.qoder/skills/help-build` and `help-new` contain the Qoder-specific workflows for this.

## Docs

Update **both** `README.md` (Chinese, primary) and `README-EN.md` (English) on user-facing changes.

## Testing

- Regression tests live in `tests/` (pure-logic: `utils/expr_type.py`, `ExpressionParser` product-to-sum, the large-power guard, and the documented SymPy gotchas). Run all: `.venv/Scripts/python.exe -m pytest` or `/test`. Run single test: `.venv/Scripts/python.exe -m pytest tests/test_xxx.py -q`. `pytest` is a dev-only dep (see `requirements_local.txt`). Add a `test_*.py` (with `tests/__init__.py` so pylint resolves top-level imports) for new deterministic logic.
- **Contract suites** (fail until a new feature is added to their maps): `test_operation_contract.py` covers every operation type (valid, empty/unfilled, malformed, matrix shape, LaTeX invariants); `test_keyboard_contract.py` covers every palette key; `test_api_contract.py` covers every Flask page/API route, task lifecycle, manual/method-tree/snapshot flows; `test_matrix_operations.py` covers every matrix operation; `test_frontend_static_contract.py` checks JS syntax, layout landmarks, unique IDs and local assets; `test_cache_contract.py` covers the SQLite cache. Add a new operation/key/API/route to the relevant contract set when implementing it.

- End-to-end smoke check: `/verify` boots `app_local.py` and curls `/api/parse`; or run `main.ipynb` / the web UI manually.

## Commit messages

Short **Chinese**, matching existing log (e.g. `更新文档`, `Fix md`). Commit once per separate type of change (e.g. backend, styles, page JS, docs) instead of batching everything together.

## Code style

- **Reuse** existing functions and patterns; add new ones only when existing utilities cannot be adapted.
- Prefer **loose coupling** — modules should depend on abstractions (`BaseCalculator`, `BaseStepGenerator`), not on concrete implementations.
- **Keep extension points:** when adding a new domain or operation, follow the existing registration pattern (`_calculators` dict, rule registry, step generator).
- Avoid redundant helper layers; a function with a single caller in the same module is usually unnecessary.

## Keys an agent would miss

- **WYSIWYG / operator symmetry:** `routes/compute_service.py::_dispatch_compute` auto-routes explicit operator templates (`diff(...)`, `\int`, `\lim`, `\det M`, `M^{-1}`) to `_dispatch_auto` regardless of the selected operation, so a derivative template is never differentiated twice. `utils/cache_utils.cached_sympify` normalizes MathLive LaTeX (`\operatorname{re}`, `\left|z\right|`, `\pi` → sympy constant, implicit `x y`), and rejects unfilled templates instead of falling back to a wrong `sympify`. Matrix inputs accept both `bmatrix` and `[[...]]`; `utils/matrix_latex.latex_to_nested` rejects empty/ragged cells.
- **Matrix page split:** `/matrix_panel` accepts a matrix for analysis/experiments; `/matrix_lab` applies executable operations (`rref`, `transpose`, `inverse`, `\mathrm{swap\_rows}(0,1)`, …) to the matrix loaded by its top form; `/matrix_analysis` analyzes one matrix. Keep each page's palette matching that role.
- **Manual step solver sessions** (`/rule_select` page) are in-memory dicts in `api.py` (`_manual_sessions`) — lost on server restart.
- **Step LaTeX rendering:** `core/base_step_generator.py::format_note` turns an explanation's ``$...$`` into live LaTeX and wraps prose in ``\text{}``. Do NOT wrap whole explanations in `\text{}` (that shows literal `$`). Keep math in `$...$` inside rule-explanation strings.
- **Applied ops** (`core/applied_calc.py`, dispatched by `compute_service`): each returns a `\begin{align}` block; integral-based ops reuse `IntegralCalculator` for step-by-step. All output must be LaTeX (no unicode math symbols in labels).
- **Method tree API** is async: POST to `/api/method_tree_start`, poll `/api/method_tree_status`, cancel via `/api/method_tree_cancel`, interactive decisions via `/api/method_tree_respond`.
- **Task results** can stream over SSE at `/api/stream` (`text/event-stream`) as an alternative to polling `/api/task_status`.
- **`/api/sympy_execute` is RCE by design** — it `exec()`/`eval()`s user-submitted code (`exec-used`/`eval-used` are pylint-disabled). Treat it as a sensitive area: don't casually expand, refactor, or copy its pattern into other endpoints.
- **Adding an operation:** register a class in `compute_service.py` (`_calculators` / `_cached_calculators`), add a branch to `start_compute()`'s if/elif chain, and expose a route in `api.py` — the dict alone does nothing.
- **`core/base_calculator.py::_apply_rule` prints `rule: <name>` to stdout** on every applied rule. This is a documented feature (README #11, "命令行显示当前调用的函数"), not leftover debug — do not remove it.
- **Manual/select path is intentionally uncapped:** `SelectRuleCalculator._apply_rule` bypasses `_check_rule_is_can_apply`, so the limit calculator's L'Hôpital count cap (`domains/limit/limit_calculator.py`) does NOT apply to the `/rule_select` manual path. By design (a human picks each step) — don't "fix" it.
- **Huge-power guard:** `utils/cache_utils.cached_sympify` parses inputs whose integer-power literal has ≥ `config.LARGE_POWER_EXPONENT_DIGITS` digits (e.g. `2**1000000`) under `sympy.evaluate(False)`, so the giant value is never materialized (avoids memory blowup + Python's 4300-digit int→str crash). The `expr` op (`ExpressionParser.parse`) still uses raw `sympify`; adopt the guard there if needed.
- **Partial derivatives:** `DiffCalculator` sets `_allow_multi_symbols = True`, so `_do_compute` keeps the requested `variable` fixed across all sub-derivatives and treats other free symbols as constants (the `var_rule` yields 0 for them). Integral/limit keep the single-symbol restriction because their BFS re-derives the variable after substitution. **Integral honors the UI's variable selector too** when the caller set `_allow_parameters` (the user path): `∫2 dt`, not `∫2 dx` — `const_matcher` treats anything independent of the working variable as a constant.
- **Result cache DB:** SQLite cache at `cache/computation.db`, managed by `utils/db_cache.py`. Hash key via SHA256 of (op_type + expression + variable/point/direction + calculator `options`: integral `include_special_matchers`, limit `max_lhopital`). Per-cell toggles ("缓存"/"记录") in the frontend send `use_cache` / `record_cache` flags through the compute payload; parse them with `core.input_validation.as_bool_flag` (the JS sends `'True'`/`'False'`, API clients may send booleans). Cached calculator variants (`CachedDiffCalculator`, `CachedIntegralCalculator`, `CachedLimitCalculator`) use the `DBCachedCalculator` mixin in `core/cached_calculator.py`; switching is handled in `compute_service.py` based on the flags. **Definite integrals (with bounds) bypass this cache** because its hash key (`db_cache.build_hash`) carries no bound fields; **unresolved derivations are never cached** (`sudden_end` suppresses the write). The table is pruned to `db_cache.CACHE_MAX_ROWS`.
- **TaskManager worker slots:** a task whose future exceeds its timeout is marked failed immediately, but its semaphore slot stays held (by a reaper thread) until the abandoned SymPy computation really finishes. A new task that cannot get a slot within its own timeout fails with "排队超时". Don't "fix" this by releasing the slot on timeout — that reintroduces unbounded executor queueing.
- **Frontend escaping rule:** anything server-authored (SymPy LaTeX, rule explanations, task results) must be wrapped in `escapeHtml(...)` (or set via `textContent`) before `innerHTML`; `sympify` accepts `Symbol('<img ...>')`, so latex alone is not safe. `tests/test_frontend_static_contract.py::test_server_values_are_escaped_before_innerhtml` lists the guarded fields — extend that list for new insertions.
- **Operator input preview:** `utils/latex_formatter.str_to_latex` renders `diff(...)`-style input as `\operatorname{...}(...)` instead of feeding it to `sympify` (whose implicit-multiplication fallback split `diff` into `d*i*f*f`). Keep the operator branches in `compute_service` and the preview in sync.
- **No CI/CD** — no `.github/`, no `vercel.json`. Deployment is manual via Vercel.
