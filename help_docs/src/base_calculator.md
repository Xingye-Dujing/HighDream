---
title: "BaseCalculator - 符号计算器基类文档"
output: "base_calculator.html"
---

# **BaseCalculator** (core/base_calculator.py)

符号计算器的抽象基类，为微分、积分、极限和行列式计算器提供统一的框架和基础功能。它实现了规则驱动的逐步计算。

<div class="warning">

**注意:**   该类只能继承使用，不能直接使用。

</div>

<div class="warning">

**注意:**   出现的所有图中，箭头的方向无实际意义，上面的文字所指可能是前对后，也可能是后对前

</div>

## 一.   类概述

BaseCalculator 是一个抽象基类，定义了符号表达式逐步计算器的核心架构。它采用以下关键技术：

- **规则注册系统**：通过 RuleRegistry 管理表达式变换规则
- **步骤生成器**：记录和展示计算过程的每一步
- **缓存机制**：优化重复计算，提高性能
- **广度优先搜索(BFS)**：使用队列存储每步待处理的表达式，确保计算的逻辑顺序

它与其它类的关系如下：

![架构一](/static/images/system_architecture_1.svg)

## 二.   设计原则

- **可扩展性**：通过继承和重写方法支持不同类型的计算器
- **模块化**：将规则管理、步骤生成、缓存等功能分离为独立组件
- **性能优化**：使用 LRU 缓存和表达式缓存减少重复计算
- **灵活性**：支持多种输出格式（列表、LaTeX）

## 三.   属性

<div class="parameter-table">

| 属性名 | 类型 | 描述 | 访问级别 |
|----|----|----|----|
| \_rule_registry | RuleRegistry | 规则注册器实例，管理与规则和匹配器相关的操作 | 私有 |
| step_generator | BaseStepGenerator | 步骤生成器实例，负责记录和管理计算步骤 | 公共 |
| processed | set | 记录已处理的表达式，防止在 BFS 过程中陷入死循环 | 公共 |
| cache | dict | 缓存表达式到对应操作对象（Derivative/Integral等）的转换结果 | 公共 |
| operation | Operation | 计算操作类型（如 Derivative, Integral），由子类在 init_key_property 中初始化 | 受保护 |
| rule_dict | RuleDict | 规则字典，包含所有可用的表达式变换规则 | 受保护 |
| matcher_list | MatcherList | 匹配器列表，用于确定应用哪些规则及应用顺序 | 受保护 |

</div>

## 四.   方法

### 1.   初始化与状态管理

```
__init__(self) -> None
```

初始化计算器实例，设置规则注册器、步骤生成器、缓存等核心组件。

**执行流程:**

1.  创建 RuleRegistry 和 BaseStepGenerator 实例
2.  初始化 processed 集合和 cache 字典
3.  调用 init_key_property 初始化关键属性
4.  验证属性是否正确初始化
5.  注册所有规则和匹配器

<img src="/static/images/base_calculator/bc_init.svg" style="max-width: 100%" alt="bc_init" />

```
@abstractmethod init_key_property(self) -> None
```

**抽象方法** - 必须在子类中实现。用于初始化计算器的关键属性：

<img src="/static/images/base_calculator/bc_init_key.svg" style="max-width: 100%" alt="bc_init_key_property" />

```
_validate_properties(self) -> None
```

验证所有必需属性是否已在 init_key_property 中正确初始化。

<div class="warning">

**注意:**   如果必需属性未初始化，将抛出 ValueError 异常。

</div>

```
_initialize_rules(self) -> None
```

通过 RuleRegistry 注册所有规则和匹配器函数。

```
reset_process(self) -> None
```

重置计算器状态(清除已处理表达式集合并重置步骤生成器)，为新的计算做准备。

### 2.   核心计算方法

```
_do_compute(self, expr: str, operation: Operation, **context: Context) -> None
```

广度优先搜索(BFS)实现逐步的符号计算。

**参数:**

- `expr`: 要计算的表达式字符串
- `operation`: 要执行的操作类型
- `**context`: 计算上下文（如变量、极限方向等）

**算法流程:**

1.  重置计算器状态
2.  将输入字符串转换为 SymPy 表达式
3.  尝试简化表达式
4.  使用 BFS 遍历表达式树
5.  对每个子表达式应用规则
6.  记录每一步的计算结果
7.  最终化简和后处理

![bc_do_compute](/static/images/base_calculator/bc_do_compute.svg)

```
_compute(self, expr: str, **context: Context) -> None
```

逐步计算表达式的包装方法，会使用实例的 operation 属性。LimitCalculator 将大幅度重构它。

### 3.   规则应用方法

```
_apply_rule(self, expr: Expr, operation: Operation, **context: Context) -> Tuple[Expr, str]
```

应用最适合的规则来变换表达式，并返回结果和解释说明。

**返回值:** 包含变换后的表达式和解释说明的元组

**执行流程:**

1.  获取适用于当前表达式的规则列表
2.  检查规则是否可以应用
3.  按优先级尝试应用规则
4.  如果没有规则匹配，回退到 SymPy 的内置计算方法

```
_check_rule_is_can_apply(self, _rule: RuleFunction) -> bool
```

检查特定规则是否可以应用。默认实现总是返回 True，子类可以重写此方法以根据特定条件过滤规则。

### 4.   缓存与性能优化方法

```
_get_cached_result(self, expr: Expr, operation: Operation, **context: Context) -> Operation
```

获取表达式对应的操作对象（Derivative/Integral等），使用缓存避免重复计算。

<img src="/static/images/base_calculator/bc_gcr.svg" style="max-width: 100%" alt="bc_gcr" />

```
@lru_cache(maxsize=128) _cached_simplify(self, expr: Expr) -> Expr
```

获得表达式的化简结果，使用 LRU 缓存机制优化性能。

### 5.   上下文处理方法

```
_context_split(self, **context: Context) -> Symbol
```

从上下文中分离出构建操作对象所需的参数。默认实现仅适合微分、积分计算器，极限和行列式计算器需要重写此方法。

```
_get_context_dict(self, **context: Context) -> RuleContext
```

将传入的上下文参数转换为规则上下文字典，供规则函数使用。

### 6.   表达式处理方法

```
_perform_operation(self, expr: Expr, operation: Operation, **context: Context) -> Operation
```

将表达式转换成对应的操作对象（Derivative/Integral等）。仅被 \_get_cached_result() 调用。

```
_update_expression(self, current_expr: Expr, operation: Operation, expr_to_operation: Dict[Expr, Operation], **context: Context) -> Tuple[Expr, str, Dict[Expr, Operation]]
```

在计算过程中更新表达式。应用规则到当前表达式，然后使用 subs 方法更新 expr_to_operation 字典中所有相关条目。

### 7.   后处理方法

```
@staticmethod _step_expr_postprocess(step_expr: Expr) -> Expr
```

在将步骤表达式添加到步骤生成器之前对其进行后处理。默认情况下直接返回输入表达式，子类可以根据需要重写此方法。

**示例:** 在积分计算器中用于添加积分常数。

```
_final_postprocess(self, final_expr: Expr) -> None
```

对最终计算结果进行后处理。通过假设所有自由符号都是正实数来进行域感知简化。

**示例:** 这有助于化简像 ln(1/x) 到 -ln(x) 这样的表达式。

<div class="warning">

**注意:**   Sympy 的 simplify() 采用保守策略，如果不假设 x 为正实数，它不会化简形如 ln(1/x) 的表达式。

</div>

### 8.   输出方法

```
@abstractmethod compute_list(self, expr: str, **context: Context) -> Tuple[List[Expr], List[str]]
```

计算表达式的逐步求解过程并以元组形式返回结果。

**抽象方法** - 必须在子类中设定计算所需的上下文。

**返回值:** 包含 SymPy 表达式列表和步骤说明字符串列表的元组

**适用场景:** 需要程序化访问计算结果的场景

```
@abstractmethod compute_latex(self, expr: str, **context: Context) -> str
```

计算表达式的逐步求解过程并以 LaTeX 字符串形式返回结果。

**抽象方法** - 必须在子类中设定计算所需的上下文。

**适用场景:** 在 Jupyter Notebook 或网页中渲染数学表达式

## 五.   子类化指南

创建 BaseCalculator 的子类需要实现以下关键步骤：

- 在子类的 init_key_property 方法中务必初始化所有必需属性
- 在子类的 compute_list 和 compute_latex 方法中务必设定计算的上下文
- 合理设计规则和匹配器，确保计算过程的正确性和效率
- 考虑重写一些方法以实现特定领域所需的操作

<div class="warning">

**注意:**   可参照已实现的 DiffCalculator, IntegralCalculator, LimitCalculator, DetCalculator 类。

</div>
