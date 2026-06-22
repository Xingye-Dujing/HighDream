---
title: "IntegralManualStepSolver - 积分手动规则选择器文档"
output: "integral_manual_step_solver.html"
---

# **IntegralManualStepSolver** (domains/integral/integral_manual_step_solver.py)

积分领域的手动规则选择器，继承自 `BaseManualStepSolver`。它把 `SelectIntegralCalculator` 包装成一个 Web 客户端可调用的 API，前端在每一步从候选规则列表里选一条，本类驱动 `SelectIntegralCalculator` 把规则应用到当前表达式，并把新产生的子表达式放回 BFS 队列。

<div class="warning">

**注意:**   本类**不**重新实现任何积分算法；所有规则注册、匹配、实际计算都由 `SelectIntegralCalculator` 完成。本类只负责：（1）把领域标识 `_domain = 'integral'` 注入到基类；（2）提供 `rule_display_names` 把规则键翻译成中文显示名；（3）覆盖 `_create_calculator()` 返回正确的 Calculator。

</div>

## 一.   类概述

`IntegralManualStepSolver` 的全部公开行为都继承自 `BaseManualStepSolver`：

- `state()` / `applicable_rules()` / `apply_rule(rule_name)` / `fallback()` / `finish()`
- `steps` / `explanations` / `pending` / `current_expr` / `done` / `error`

子类本身只声明三件事：

```
_domain = 'integral'
rule_display_names = _RULE_DISPLAY_NAMES

def _create_calculator(self):
    return SelectIntegralCalculator()
```

## 二.   已注册的规则键

积分领域的规则覆盖较广，除基础函数外还包含换元法、分部积分法、万能公式代换等复杂规则。

<div class="parameter-table">

| rule_key | 中文显示名 |
|----|----|
| add | 加法展开 |
| const / var | 常数积分 / 变量积分 |
| mul_const | 常数乘法 |
| pow | 幂函数积分 |
| exp / log | 指数 / 对数函数积分 |
| sin / cos / tan / sec / csc / cot | 三角函数积分 |
| sinh / cosh / tanh / sech / csch / coth | 双曲函数积分 |
| inverse_trig | 反三角函数积分 |
| inverse_tangent_linear | 线性反正切积分 |
| sin_power / cos_power / tan_power | 三角函数幂积分 |
| logarithmic | 对数函数积分 |
| parts | 分部积分法 |
| substitution | 换元积分法 |
| f_x_mul_exp_g_x | `f(x) e^{g(x)}` 型积分 |
| quotient_diff_form | 商微分形式 |
| quadratic_sqrt_reciprocal | 二次根式倒数积分 |
| sqrt_div_sqrt | 根式相除积分 |
| weierstrass_substitution | 万能公式代换 |

</div>

<div class="warning">

**注意:**   `substitution` 类规则触发后，`finish()` 会自动做回代（把中间变量替换回原变量）。在 `MethodTreeEnumerator` 的遍历里，回代会被作为一个额外的子节点显示。

</div>

## 三.   使用方法

```
solver = ManualStepSolver(domain='integral', expression='x*exp(x)', variable='x')
state = solver.state()
rules = solver.applicable_rules()   # 通常会包含 'parts'
state = solver.apply_rule('parts')  # 分部积分
state = solver.finish()             # 回代换元变量 + 化简
```

## 四.   参见

- [BaseManualStepSolver](/help/base_manual_step_solver) — 全部公开 API 的详细说明
- [DiffManualStepSolver](/help/diff_manual_step_solver) — 微分子类
- [LimitManualStepSolver](/help/limit_manual_step_solver) — 极限子类
