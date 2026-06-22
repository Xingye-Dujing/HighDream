---
title: "DiffManualStepSolver - 微分手动规则选择器文档"
output: "diff_manual_step_solver.html"
---

# **DiffManualStepSolver** (domains/differentiation/diff_manual_step_solver.py)

微分领域的手动规则选择器，继承自 `BaseManualStepSolver`。它把 `SelectDiffCalculator` 包装成一个 Web 客户端可调用的 API：前端在每一步从候选规则列表里选一条，本类驱动 `SelectDiffCalculator` 把规则应用到当前表达式，并把新产生的子表达式放回 BFS 队列。

<div class="warning">

**注意:**   本类**不**重新实现任何微分算法；所有规则注册、匹配、实际计算都由 `SelectDiffCalculator` 完成。本类只负责：（1）把领域标识 `_domain = 'diff'` 注入到基类；（2）提供 `rule_display_names` 把规则键翻译成中文显示名；（3）覆盖 `_create_calculator()` 返回正确的 Calculator。

</div>

## 一.   类概述

`DiffManualStepSolver` 的全部公开行为都继承自 `BaseManualStepSolver`：

- `state()` / `applicable_rules()` / `apply_rule(rule_name)` / `fallback()` / `finish()`
- `steps` / `explanations` / `pending` / `current_expr` / `done` / `error`

子类本身只声明三件事：

```
_domain = 'diff'
rule_display_names = _RULE_DISPLAY_NAMES

def _create_calculator(self):
    return SelectDiffCalculator()
```

## 二.   已注册的规则键

<div class="parameter-table">

| rule_key | 中文显示名 |
|----|----|
| add | 加法法则 |
| mul | 乘法法则 |
| div | 除法法则 |
| chain | 链式法则 |
| const | 常数求导 |
| var | 变量求导 |
| pow | 幂函数求导 |
| sin / cos / tan / sec / csc / cot | 对应三角函数求导 |
| asin / acos / atan | 对应反三角函数求导 |
| exp | 指数求导 |
| log | 对数求导 |
| sinh / cosh / tanh | 对应双曲函数求导 |

</div>

<div class="warning">

**注意:**   若 `SelectDiffCalculator` 新增了规则但此处没有同步加中文名，`applicable_rules()` 返回的 `display_name` 会回退到 `name` 本身（前端会看到英文键名）。

</div>

## 三.   使用方法

通过统一的工厂入口（`core/manual_step_solver.py` 的 `ManualStepSolver` 类）创建：

```
solver = ManualStepSolver(domain='diff', expression='sin(x)*exp(x)', variable='x')
state = solver.state()             # 初始快照
rules = solver.applicable_rules()  # 当前可应用规则列表
state = solver.apply_rule('mul')   # 选乘法法则
state = solver.fallback()          # 或让 SymPy 兜底
state = solver.finish()            # 结束会话
```

## 四.   参见

- [BaseManualStepSolver](/help/base_manual_step_solver) — 全部公开 API 的详细说明
- [IntegralManualStepSolver](/help/integral_manual_step_solver) — 积分子类
- [LimitManualStepSolver](/help/limit_manual_step_solver) — 极限子类
