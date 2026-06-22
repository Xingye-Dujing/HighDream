---
title: "BaseManualStepSolver - 手动规则选择基础编排器文档"
output: "base_manual_step_solver.html"
---

# **BaseManualStepSolver** (core/base_manual_step_solver.py)

Web 界面手动规则选择的基础编排器。它把一个领域相关的 `Select*Calculator` 包在身后，对外暴露一个**非阻塞**的 API：前端可以每一步都挑一条规则来应用，也可以在某一步回退到 SymPy 兜底。领域特定的子类（微分 / 积分 / 极限）放在 `domains/` 目录下，只覆盖配置项（使用的 Calculator、规则中文名、领域相关钩子），不重新实现核心符号计算。

<div class="warning">

**注意:**   该类不直接实例化。生产代码通过 `DiffManualStepSolver` / `IntegralManualStepSolver` / `LimitManualStepSolver` 间接使用。

</div>

<div class="warning">

**注意:**   出现的所有图中，箭头的方向无实际意义，上面文字所指可能是前对后，也可能是后对前

</div>

## 一.   类概述

`BaseManualStepSolver` 把规则驱动计算拆解成「前端挑规则 → 后端应用规则 → 前端看下一步」的循环，主要组件：

- **Select\*Calculator**：负责规则注册、表达式缓存、实际操作对象构造（Derivative / Integral / Limit）。本类**不**重新实现核心符号计算，只是驱动它。
- **BFS 队列 `pending`**：存储等待处理的子表达式；每一步用户选定规则后，把结果产生的新子表达式追加回队列。
- **步骤记录 `steps` / `explanations`**：记录到当前为止的推导链，与 Calculator 内置的 `step_generator` 保持同步。
- **顶层表达式字典 `expr_to_operation`**：把最外层表达式中的子表达式替换为最新计算结果，用于实时生成「全局视图」的 LaTeX。

它与其它类的关系如下：

```
routes/api.py  ──►  BaseManualStepSolver  ──►  Select*Calculator
                           │                         │
                           │                         ▼
                           │                  RuleRegistry
                           ▼
                    BaseStepGenerator
```

## 二.   设计原则

- **领域解耦**：核心算法（BFS、规则预览、提交步骤、兜底回退）放在基类；领域相关的「用哪个 Calculator」「规则中文名」「上下文是否需要 point/direction」交给子类。
- **幂等预览**：`applicable_rules()` 在列举适用规则时，会为每条规则生成 LaTeX 预览，但不会污染 `steps`——内部用「快照 + 还原」把 `step_generator` 和求解器状态恢复到预览前的样子。
- **非阻塞 API**：每个公开方法都返回一份 JSON 友好的 `state()`，前端轮询或 WebSocket 都能用。
- **失败友好**：规则抛异常、规则被限流（例如 l'Hôpital 次数超限）、规则对当前表达式不匹配——都不会让求解器进入非法状态，错误信息写到 `self.error` 里，由前端展示。

## 三.   属性

<div class="parameter-table">

| 属性名 | 类型 | 描述 | 访问级别 |
|----|----|----|----|
| calculator | Select\*Calculator | 被包装的领域计算器，负责规则注册、缓存与操作对象构造 | 公共 |
| variable | Symbol | 主变量符号 | 公共 |
| point | Expr | 极限趋近点（仅 limit 领域使用，其余领域为默认值 0） | 公共 |
| direction | str | 极限趋近方向（`'+'` 或 `'-'`，仅 limit 领域使用） | 公共 |
| steps | List\[Expr\] | 到目前为止的推导链表达式 | 公共 |
| explanations | List\[str\] | 与 `steps` 对齐的解释文本 | 公共 |
| top_expr | Expr | 初始（最外层）表达式 | 公共 |
| expr_to_operation | Dict\[Expr, Operation\] | 子表达式 → 当前最新 Operation 的映射，用于全局视图 | 公共 |
| pending | Deque\[Expr\] | BFS 等待队列，存储尚未被用户处理的子表达式 | 公共 |
| current_expr | Optional\[Expr\] | 当前正在等待用户挑选规则的表达式 | 公共 |
| done | bool | 推导是否已结束 | 公共 |
| error | Optional\[str\] | 上一次 `apply_rule` / `fallback` 失败的错误说明 | 公共 |
| rule_display_names | Dict\[str, str\] | 类属性：`rule_key -> 中文显示名`，子类必须覆盖 | 类变量 |
| \_domain | str | 类属性：领域标识，子类覆盖为 `'diff'` / `'integral'` / `'limit'` | 类变量 |

</div>

## 四.   方法

### 1.   初始化

```
__init__(self, expression: str, variable: str = 'x',
         point: Any = 0, direction: str = '+') -> None
```

构造一个手动求解会话：

1. 调用 `_create_calculator()` 拿到一个领域相关的 `Select*Calculator`。
2. 把输入表达式 sympify 并做一次性化简；若化简发生变化，把「简化表达式」作为额外步骤记入 `step_generator`。
3. 构造初始操作对象（`Derivative` / `Integral` / `Limit`）并存入 `expr_to_operation`。
4. 初始化 `steps` / `explanations` / `pending` / `current_expr`。

<div class="warning">

**注意:**   子类**必须**覆盖 `_create_calculator()`；否则构造时会抛出 `NotImplementedError`。

</div>

### 2.   公开 API

```
state(self) -> Dict[str, Any]
```

返回当前会话的 JSON 友好快照。字段：

<div class="parameter-table">

| 字段 | 含义 |
|----|----|
| `done` | 推导是否已结束 |
| `error` | 上一次失败的错误说明 |
| `domain` | 领域标识 |
| `top_level_latex` | 最外层表达式的当前 LaTeX（子表达式已替换为最新结果） |
| `current_expr_latex` | 正在等待挑选规则的子表达式 LaTeX |
| `pending` | 等待队列中所有子表达式的 LaTeX |
| `steps` | `[{latex, explanation}, ...]`，已提交步骤的列表 |
| `applicable_rules` | 当前表达式上可应用规则的预览（若已结束则为空） |

</div>

```
applicable_rules(self) -> List[Dict[str, str]]
```

列出当前表达式上所有可应用的规则。每项：

<div class="parameter-table">

| 字段 | 含义 |
|----|----|
| `name` | 规则键（例如 `'chain'`、`'parts'`），即前端后续调用 `apply_rule` 用的名字 |
| `display_name` | 中文显示名（来自 `rule_display_names`，缺失则回退到 `name`） |
| `latex_preview` | 若该规则被应用，结果表达式的 LaTeX；预览失败时为 `None` |

</div>

<div class="warning">

**注意:**   预览会临时修改 `step_generator` 与求解器状态，但 `applicable_rules` 内部用 `_snapshot_solver_state` / `_restore_solver_state` / `_backup_step_generator` / `_restore_step_generator` 四件套把所有副作用都还原，对后续调用无影响。

</div>

```
apply_rule(self, rule_name: str) -> Dict[str, Any]
```

应用指定规则到 `current_expr`，提交步骤，并推进 BFS 队列。返回 `state()`。

失败场景（不会让求解器进入非法状态）：

- 会话已结束 → 抛 `RuntimeError`
- 规则未注册 → 抛 `KeyError`
- 规则被限流（例如 l'Hôpital 次数超限）→ `self.error` 被设置，`state()` 返回
- 规则返回 `None` / 抛异常 → `self.error` 被设置，`step_generator` 回滚到调用前

```
fallback(self) -> Dict[str, Any]
```

对 `current_expr` 调用 SymPy 的 `.doit()` 兜底计算。解释文本形如 `手动计算（SymPy 回退）: $...$`。返回 `state()`。

```
finish(self) -> Dict[str, Any]
```

运行最终后处理（`calculator.final_postprocess`，包含回代换元变量 + 保守化简），把 `done` 置为 `True`。返回 `state()`。幂等：若已 `done` 直接返回当前状态。

### 3.   领域钩子（子类可覆盖）

```
_create_calculator(self)
```

**抽象方法**：返回一个构造完毕的 `Select*Calculator`。三个子类必须覆盖。

```
_init_calculator(self) -> None
```

构造 Calculator 之后的领域初始化钩子。默认什么都不做；`LimitManualStepSolver` 用它重置 `_lhopital_count`。

```
_extend_context(self, ctx: Dict[str, Any], expr: Expr) -> None
```

往「传给 Calculator 的上下文」里追加领域字段。默认什么都不做；`LimitManualStepSolver` 用它注入 `point` 和 `direction`。

```
_snapshot_solver_state(self) -> Dict[str, Any]
_restore_solver_state(self, snapshot: Dict[str, Any]) -> None
```

用于 `applicable_rules()` 做预览前后保存/还原领域内部计数器。默认空实现；`LimitManualStepSolver` 用它保护 `_lhopital_count`，避免「仅仅列举规则」就消耗洛必达使用次数。

## 五.   子类化指南

创建一个新领域（例如「行列式」）的子类需要：

1. 在 `domains/<新领域>/<新领域>_manual_step_solver.py` 里写一个 `class XxxManualStepSolver(BaseManualStepSolver)`。
2. 覆盖 `_domain` 与 `rule_display_names`。
3. 覆盖 `_create_calculator()`，返回对应领域的 `Select*Calculator`。
4. 若该领域有特殊上下文（例如 `point`、`direction`、`subs_dict`），覆盖 `_extend_context` 与 `_snapshot_solver_state` / `_restore_solver_state`。
5. 在 `routes/api.py` 的 `_manual_sessions` 工厂里把新领域映射到新子类，并（可选）在 `ManualStepSolver`（`core/manual_step_solver.py`）里暴露一个统一的工厂入口。

<div class="warning">

**注意:**   可参照已实现的 `DiffManualStepSolver`、`IntegralManualStepSolver`、`LimitManualStepSolver`。

</div>
