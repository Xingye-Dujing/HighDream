# AGENTS.md

Guidance for AI agents working in this repository.

## Running the app

- **Local dev (opens http://127.0.0.1:5000/):** `python app_local.py` — uses `gevent.pywsgi.WSGIServer`, opens browser automatically.
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
- `routes/` — Flask blueprints: `main.py` (pages), `api.py` (compute endpoints). Async tasks via `task_manager.py` (in-memory `ThreadPoolExecutor`). Compute dispatch in `compute_service.py`: `start_compute()` routes via an if/elif chain (the real dispatcher); the `_calculators` / `_cached_calculators` dicts are just operation-name → calculator-class lookups. Runtime op strings (`det`, `transform-1`, …) and some dict keys (`det_1`, `transform_1`) intentionally differ — don't assume the dict drives routing.
- `utils/` — LaTeX rendering (`latex_formatter.py`) and expression helpers.
- `config.py` — central tunables: parser depth, L'Hôpital count, method-tree cutoffs (defaults and hard limits), task timeouts.
- `app_local.py` / `app.py` both call `create_app()` which registers `main` and `api` blueprints.

## Lint & auto-format

- **Pylint** only (`.pylintrc` disables many pedantic rules). Run `.venv/Scripts/pylint <file>`.
- A PostToolUse hook in `.qoder/settings.json` auto-runs `.venv/Scripts/pylint.exe` on every `.py` file after editing (a pylint failure blocks the edit).
- **`.pylintrc` is gitignored** — a fresh clone won't have it; recreate it or the hook re-flags rules the project intentionally allows.
- No auto-formatter. Match existing 4-space PEP-8-ish style.

## Help documentation (`/help/*`)

- **Author in:** `help_docs/src/*.md`
- **Generate HTML:** `python help_docs/build_help.py html` (requires pandoc ≥3 on `PATH`)
- **Never hand-edit** `templates/help/*.html` — changes are clobbered on next build.
- Both `.md` and generated `.html` are git-tracked (Vercel serves the HTML directly); commit the pair together.
- Existing `.qoder/skills/help-build` and `help-new` contain the Qoder-specific workflows for this.

## Docs

Update **both** `README.md` (Chinese, primary) and `README-EN.md` (English) on user-facing changes.

## Testing

- Regression tests live in `tests/` (pure-logic: `utils/expr_type.py`, `ExpressionParser` product-to-sum, the large-power guard, and the documented SymPy gotchas). Run `.venv/Scripts/python.exe -m pytest` or `/test`; `pytest` is a dev-only dep (see `requirements_local.txt`). Add a `test_*.py` (with `tests/__init__.py` so pylint resolves top-level imports) for new deterministic logic.
- End-to-end smoke check: `/verify` boots `app_local.py` and curls `/api/parse`; or run `main.ipynb` / the web UI manually.

## Commit messages

Short **Chinese**, matching existing log (e.g. `更新文档`, `Fix md`).

## Code style

- **Reuse** existing functions and patterns; add new ones only when existing utilities cannot be adapted.
- Prefer **loose coupling** — modules should depend on abstractions (`BaseCalculator`, `BaseStepGenerator`), not on concrete implementations.
- **Keep extension points:** when adding a new domain or operation, follow the existing registration pattern (`_calculators` dict, rule registry, step generator).
- Avoid redundant helper layers; a function with a single caller in the same module is usually unnecessary.

## Keys an agent would miss

- **Manual step solver sessions** (`/rule_select` page) are in-memory dicts in `api.py` (`_manual_sessions`) — lost on server restart.
- **Method tree API** is async: POST to `/api/method_tree_start`, poll `/api/method_tree_status`, cancel via `/api/method_tree_cancel`, interactive decisions via `/api/method_tree_respond`.
- **Task results** can stream over SSE at `/api/stream` (`text/event-stream`) as an alternative to polling `/api/task_status`.
- **`/api/sympy_execute` is RCE by design** — it `exec()`/`eval()`s user-submitted code (`exec-used`/`eval-used` are pylint-disabled). Treat it as a sensitive area: don't casually expand, refactor, or copy its pattern into other endpoints.
- **Adding an operation:** register a class in `compute_service.py` (`_calculators` / `_cached_calculators`), add a branch to `start_compute()`'s if/elif chain, and expose a route in `api.py` — the dict alone does nothing.
- **`core/base_calculator.py::_apply_rule` prints `rule: <name>` to stdout** on every applied rule. This is a documented feature (README #11, "命令行显示当前调用的函数"), not leftover debug — do not remove it.
- **Manual/select path is intentionally uncapped:** `SelectRuleCalculator._apply_rule` bypasses `_check_rule_is_can_apply`, so the limit calculator's L'Hôpital count cap (`domains/limit/limit_calculator.py`) does NOT apply to the `/rule_select` manual path. By design (a human picks each step) — don't "fix" it.
- **Huge-power guard:** `utils/cache_utils.cached_sympify` parses inputs whose integer-power literal has ≥ `config.LARGE_POWER_EXPONENT_DIGITS` digits (e.g. `2**1000000`) under `sympy.evaluate(False)`, so the giant value is never materialized (avoids memory blowup + Python's 4300-digit int→str crash). The `expr` op (`ExpressionParser.parse`) still uses raw `sympify`; adopt the guard there if needed.
- **Partial derivatives:** `DiffCalculator` sets `_allow_multi_symbols = True`, so `_do_compute` keeps the requested `variable` fixed across all sub-derivatives and treats other free symbols as constants (the `var_rule` yields 0 for them). Integral/limit keep the single-symbol restriction because their BFS re-derives the variable after substitution.
- **Result cache DB:** SQLite cache at `cache/computation.db`, managed by `utils/db_cache.py`. Hash key via SHA256 of (op_type + expression + variable/point/direction). Per-cell toggles ("缓存"/"记录") in the frontend send `use_cache` / `record_cache` flags through the compute payload. Cached calculator variants (`CachedDiffCalculator`, `CachedIntegralCalculator`, `CachedLimitCalculator`) use the `DBCachedCalculator` mixin in `core/cached_calculator.py`; switching is handled in `compute_service.py` based on the flags. **Definite integrals (with bounds) bypass this cache** because its hash key (`db_cache.build_hash`) carries no bound fields.
- **No CI/CD** — no `.github/`, no `vercel.json`. Deployment is manual via Vercel.
