---
title: "LimitManualStepSolver - 极限手动规则选择器文档"
output: "limit_manual_step_solver.html"
---

# **LimitManualStepSolver** (domains/limit/limit_manual_step_solver.py)

极限领域的手动规则选择器，继承自 `BaseManualStepSolver`。它在基类基础上做了三件事：（1）把 `point` 与 `direction` 注入到每次传给 Calculator 的上下文；（2）保护 `_lhopital_count`，避免「仅仅列举规则」就消耗洛必达使用次数；（3）把 `SelectLimitCalculator` 包装成 Web 客户端可调用的 API。

<div class="warning">

**注意:**   与微分 / 积分子类不同，本类必须覆盖 `_extend_context`、`_snapshot_solver_state`、`_restore_solver_state`，因为极限的每条规则都需要知道 `x → a^+` / `x → a^-` 这类信息，且 l'Hôpital 规则内部有使用次数上限。

</div>

## 一.   类概述

`LimitManualStepSolver` 的全部公开行为都继承自 `BaseManualStepSolver`：

- `state()` / `applicable_rules()` / `apply_rule(rule_name)` / `fallback()` / `finish()`
- `steps` / `explanations` / `pending` / `current_expr` / `done` / `error`

子类本身声明：

```
_domain = 'limit'
rule_display_names = _RULE_DISPLAY_NAMES

def _create_calculator(self):     return SelectLimitCalculator()
def _init_calculator(self):       self.calculator._lhopital_count = 0
def _extend_context(self, ctx, _expr):
    ctx['point'] = self.point
    ctx['direction'] = self.direction
def _snapshot_solver_state(self): return {'lhopital_count': ...}
def _restore_solver_state(self, snapshot): ...
```

## 二.   已注册的规则键

极限领域既有「重要极限」类规则，也有「拆分 / 合并」类规则，以及「洛必达法则」家族（按未定式类型细分）：

<div class="parameter-table">

| rule_key | 中文显示名 |
|----|----|
| sin_over_x | 重要极限 `sin(x)/x` |
| one_plus_one_over_x_pow_x | 重要极限 `(1+1/x)^x` |
| ln_one_plus_x_over_x | 重要极限 `ln(1+x)/x` |
| exp_minus_one_over_x | 重要极限 `(e^x-1)/x` |
| g_over_sin / g_over_ln_one_plus / g_over_exp_minus_one | `g(x)/...` 型 |
| mul_split / add_split / div_split | 乘法 / 加法 / 除法拆分 |
| direct_substitution | 直接代入 |
| conjugate_rationalize | 共轭有理化 |
| small_o_add | 小 o 加法 |
| const_inf_add / const_inf_mul / const_inf_div / const_zero_div | 常数与无穷的组合 |
| lhopital_direct | 洛必达（0/0, ∞/∞） |
| lhopital_zero_times_inf | 洛必达（0·∞） |
| lhopital_inf_minus_inf | 洛必达（∞-∞） |
| lhopital_power | 洛必达（幂指型） |
| pow / exp / log | 幂指 / 指数 / 对数函数极限 |
| sin / cos / tan / sec / csc / cot | 三角函数极限 |
| asin / acos / atan | 反三角函数极限 |
| sinh / cosh / tanh | 双曲函数极限 |

</div>

<div class="warning">

**注意:**   洛必达规则内部会递增 `_lhopital_count`，超过 `SelectLimitCalculator` 配置的上限（默认 3 次）后，规则仍然会出现在 `applicable_rules()` 列表里，但 `apply_rule('lhopital_*')` 会返回错误并让 `self.error` 指示「已被限制」。`applicable_rules()` 的预览不会消耗计数——它使用 `_snapshot_solver_state` / `_restore_solver_state` 保护了 `_lhopital_count`。

</div>

## 三.   使用方法

```
solver = ManualStepSolver(
    domain='limit',
    expression='sin(x)/x',
    variable='x',
    point=0,
    direction='+',
)
state = solver.state()
rules = solver.applicable_rules()        # 通常包含 'sin_over_x'
state = solver.apply_rule('sin_over_x')  # 应用重要极限
state = solver.finish()
```

## 四.   参见

- [BaseManualStepSolver](/help/base_manual_step_solver) — 全部公开 API 的详细说明
- [DiffManualStepSolver](/help/diff_manual_step_solver) — 微分子类
- [IntegralManualStepSolver](/help/integral_manual_step_solver) — 积分子类
