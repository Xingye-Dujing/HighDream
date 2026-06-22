---
title: "MethodTreeEnumerator - 全规则遍历方法树文档"
output: "method_tree_enumerator.html"
---

# **MethodTreeEnumerator** (core/method_tree_enumerator.py)

给定一个求解问题（领域 + 表达式 + 变量 / 极限点 / 方向），`MethodTreeEnumerator` 用 BFS 把 `BaseManualStepSolver` **所有**可走的规则应用路径全部展开成一棵树。内置硬截断（深度、总节点数、挂钟时间、外部取消）保证单次请求不会卡住工作线程。它被设计为由 `routes/task_manager.py` 在后台线程驱动，HTTP 层通过 `snapshot()` 拉取增量进度。

<div class="warning">

**注意:**   出现的所有图中，箭头的方向无实际意义，上面文字所指可能是前对后，也可能是后对前

</div>

## 一.   类概述

`MethodTreeEnumerator` 解决的核心问题是「给定一个数学问题，所有可能的求解路径长什么样」：

- 每个节点是一个「已经应用了若干规则的求解器快照」。
- 每条边是一次规则应用（边上标注规则键与中文名）。
- 当一个节点 `done == True` 时，`final_latex` 保存该条路径的最终答案。
- 当一次规则触发了换元 + 回代（例如 `substitution`），会在「回代前」和「回代后」之间插入一个额外的子节点，让用户能看清回代这一步。

它依赖：

- `ManualStepSolver`（`core/manual_step_solver.py`）：统一的领域工厂，根据 `domain` 挑出对应 `BaseManualStepSolver` 子类。
- `config.py` 里的 6 个截断常量：`METHOD_TREE_DEFAULT_MAX_DEPTH` / `METHOD_TREE_HARD_MAX_DEPTH` 等。
- `routes/task_manager.py`：把它当 `task_manager.create_task('method_tree', ...)` 的 worker 跑。

## 二.   设计原则

- **合作式取消**：在每次出队、每条子规则分支、每个 `_should_stop()` 调用点都检查截断条件；外部调用 `cancel()` 后，`run()` 在下一个安全点返回。
- **增量可见**：`snapshot()` 加锁复制当前 `_nodes` / `_children`，HTTP 层可以在 `run()` 还没结束时就开始渲染前端画布。
- **回放式一致性**：每个节点的「当前求解器」不是从父节点继承来的，而是从根节点**重新构造**并把规则路径从头回放到该节点。这避免了并发下共享可变状态，代价是 O(路径长度) 的重放成本——可接受，因为单次规则应用是毫秒级。
- **双模式**：`interactive=True` 时进入「逐步确认」协议，每展开一条规则边都通过 `_decision_condition` 暂停并等待 `respond_decision()`；`interactive=False` 时一口气跑完，受 `time_limit_seconds` 约束。

## 三.   构造参数

```
__init__(self,
         domain: str,
         expression: str,
         variable: str = 'x',
         point: Any = 0,
         direction: str = '+',
         max_depth: Any = METHOD_TREE_DEFAULT_MAX_DEPTH,
         max_nodes: Any = METHOD_TREE_DEFAULT_MAX_NODES,
         time_limit_seconds: Any = METHOD_TREE_DEFAULT_TIME_SECONDS,
         interactive: bool = False) -> None
```

<div class="parameter-table">

| 参数 | 描述 | 默认 / 上限 |
|----|----|----|
| domain | 领域标识：`'diff'` / `'integral'` / `'limit'` | 必填 |
| expression | 待求解的表达式字符串 | 必填 |
| variable | 主变量名 | `'x'` |
| point | 极限趋近点（仅 limit 领域使用） | `0` |
| direction | 极限趋近方向 `'+'` / `'-'` | `'+'` |
| max_depth | 根到叶的最大深度 | `METHOD_TREE_DEFAULT_MAX_DEPTH`；硬上限 `METHOD_TREE_HARD_MAX_DEPTH` |
| max_nodes | 树的最大节点总数 | `METHOD_TREE_DEFAULT_MAX_NODES`；硬上限 `METHOD_TREE_HARD_MAX_NODES` |
| time_limit_seconds | 挂钟预算（仅 `interactive=False` 生效） | `METHOD_TREE_DEFAULT_TIME_SECONDS`；硬上限 `METHOD_TREE_HARD_MAX_TIME_SECONDS` |
| interactive | 是否进入逐步确认协议 | `False` |

</div>

<div class="warning">

**注意:**   超过硬上限的值会被 `_clamp` 静默裁剪回硬上限；非法类型或 ≤0 的值会被替换为默认值。仅当最终值仍 `< 1` 时才抛 `ValueError`。

</div>

## 四.   公开方法

### 1.   run

```
run(self) -> Dict[str, Any]
```

启动 BFS 并返回完整的方法树 payload。payload 字段：

<div class="parameter-table">

| 字段 | 含义 |
|----|----|
| `root_id` | 根节点的 ID（字符串，形如 `'n0'`） |
| `nodes` | `id -> node_dict`，节点字典 |
| `children` | `id -> [child_ids]`，便于前端按父子关系布局 |
| `pending_decision` | 仅 `interactive=True`：等待用户确认的边 |
| `stats` | 节点数、已用时间、最深节点深度等统计 |
| `truncated` | 是否被任一截断条件打断 |
| `reason` | 结束原因：`'completed'` / `'depth_limit'` / `'node_limit'` / `'time_limit'` / `'cancelled'` / `'error'` |
| `max_depth_seen` | 实际达到的最大深度 |
| `error` | 异常信息（仅 `reason == 'error'` 时非空） |

</div>

节点字典字段：

<div class="parameter-table">

| 字段 | 含义 |
|----|----|
| `id` / `parent` / `depth` | 节点 ID / 父 ID / 深度 |
| `latex` | 当前工作表达式的 LaTeX（回代前为回代前，回代后为回代后） |
| `top_latex` | 顶层表达式的当前 LaTeX（便于显示「全局视图」） |
| `rule_applied` / `rule_display` | 到达该节点所用的规则键与中文名；回代节点的 `rule_applied` 是 `'back_subs'` |
| `explanation` | 规则返回的解释文本 |
| `done` | 该节点是否已完成推导 |
| `truncated` | 该节点是否被深度截断 |
| `final_latex` | 当 `done == True` 时，最终答案的 LaTeX |
| `children` | 子节点 ID 列表 |

</div>

### 2.   cancel

```
cancel(self) -> None
```

请求合作式取消。`run()` 会在下一个 `_should_stop()` 调用点返回。`reason` 会被设为 `'cancelled'`。若在交互模式下且正等待 `respond_decision`，该方法同时会把响应置为 `False` 并唤醒等待线程。

### 3.   respond_decision

```
respond_decision(self, accepted: bool) -> bool
```

仅 `interactive=True` 时生效。回应 `snapshot()['pending_decision']` 所描述的「是否应用这条规则」的询问。返回 `True` 表示当时确实在等待响应；`False` 表示没有等待（例如已取消或已结束）。

### 4.   snapshot

```
snapshot(self) -> Dict[str, Any]
```

返回当前已构建部分的方法树快照（线程安全）。HTTP 层通常在 `run()` 异步跑的同时轮询本方法以增量渲染前端画布。结构与 `run()` 返回的 payload 一致，只是可能还不完整。

## 五.   内部机制（阅读源码时参考）

### 1.   回放式求解器

每个节点并不持有自己的 `ManualStepSolver` 实例。当需要「当前节点上的可用规则」时，`_replay_solver(rule_path)` 会从根节点重新构造一个新 solver，按 `rule_path` 把规则顺序 `apply_rule` 一遍，得到和该节点一致的求解器状态。这避免了共享可变状态，但让每个节点的代价是 O(路径长度)。

### 2.   截断检查

`_should_stop()` 依次检查：

1. 节点数 ≥ `max_nodes` → `reason = 'node_limit'`
2. 非交互模式 + 已用时间 ≥ `time_limit` → `reason = 'time_limit'`
3. `_stop` 被置位（`cancel()` 触发）→ `reason = 'cancelled'`

任一条件命中都会设置 `truncated = True` 并返回 `True`。深度截断（`depth >= max_depth`）在出队时单独判断并标记对应节点 `truncated = True`。

### 3.   回代检测

`_build_tree()` 在每次 `apply_rule` 后检查 `solver.calculator.step_generator.subs_dict` 是否非空；若非空，说明发生了换元 + 回代，会在「回代前」与「回代后」之间插入一个 `rule_applied = 'back_subs'`、`rule_display = '回代换元变量'` 的额外子节点，让前端能把回代这一步单独可视化。

## 六.   与 HTTP 层的配合

`routes/api.py` 暴露：

- `POST /api/method_tree_start` — 启动一次遍历，返回 `task_id`。
- `POST /api/method_tree_status` — 拉取 `task_manager` 状态；若仍在跑，调用 `snapshot()` 返回增量树；若已结束，从 `task.result` 拿完整树。
- `POST /api/method_tree_cancel` — 调用 `cancel()`。

`routes/method_tree_service.py` 封装了具体的调用与快照逻辑。

## 七.   使用示例

```
enumerator = MethodTreeEnumerator(
    domain='integral',
    expression='x*exp(x)',
    variable='x',
    max_depth=6,
    max_nodes=200,
    time_limit_seconds=30,
)
payload = enumerator.run()
print(payload['stats'])
for nid, node in payload['nodes'].items():
    if node['done']:
        print(nid, '=>', node['final_latex'])
```

## 八.   参见

- [BaseManualStepSolver](/help/base_manual_step_solver) — 单条路径上的手动求解器
- [RuleRegistry](/help/rule_registry) — 规则匹配与调度
