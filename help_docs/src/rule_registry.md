---
title: "RuleRegistry - 规则注册器文档"
output: "rule_registry.html"
---

# **RuleRegistry** (core/rule_registry.py)

符号表达式的规则系统，管理规则的注册、匹配和调度。它实现了灵活的规则驱动架构，支持各种数学运算的自动变换。

<div class="warning">

**注意:**   出现的所有图中，箭头的方向无实际意义，上面的文字所指可能是前对后，也可能是后对前

</div>

## 一.   类概述

RuleRegistry 实现了基于规则的模式匹配和表达式转换系统，主要功能包括：

- **规则管理**：注册、组织和调度表达式转换规则
- **匹配系统**：通过匹配器函数确定适用的规则
- **动态调度**：根据表达式类型自动选择最佳规则
- **工厂模式**：提供常用规则和匹配器的快速创建方法

它与其它类的关系如下：

![](/static/images/system_architecture_1.svg)

## 二.   设计原则

- **分离关注点**：规则处理逻辑与匹配逻辑彻底分离，但造成了一定的性能损耗
- **可扩展性**：支持动态添加新规则和匹配器
- **类型安全**：使用类型注解确保接口一致性

<div class="warning">

**注意：**规则函数无需检测此规则对表达式是否可用，直接变换表达式即可。因为只要进入了某一个规则函数，则说明已通过对应匹配器函数的检测，

</div>

## 三.   核心概念

### 1.   上下文字典 (RuleContext)

存储执行计算需要的参数，比如变量、极限方向等

```
RuleContext: Dict[str, Union[Symbol, Expr, str]]
```

### 2.   操作类型 (Operation)

**Sympy 的数学运算类型**，如 Derivative, Determinant, Integral, Limit 等。

### 3.   规则函数 (RuleFunction)

执行实际表达式转换的函数，接收表达式和上下文字典，返回转换结果和解释说明。

```
RuleFunction: Callable[[Expr, RuleContext], Optional[Tuple[Expr, str]]]
```

<div class="warning">

**注意:**   除非该步可以直接得到结果，否则返回的 Expr 应包括 Operation 对象。此外，整数要以 Sympy 的 Integer 对象返回，**返回 Pyhon 的 int 对象将报错**。

</div>

### 4.   匹配器函数 (MatcherFunction)

判断表达式是否适用特定规则的函数，返回规则名称或 None。

```
MatcherFunction: Callable[[Expr, RuleContext], Optional[str]]
```

## 四.   系统架构

#### 1.   工作流程

- **注册阶段**：规则和匹配器被注册到 RuleRegistry
- **匹配阶段**：对输入表达式运行所有匹配器
- **执行阶段**：调用匹配成功的规则函数
- **返回阶段**：返回转换后的表达式和解释

#### 2.   数据流

![](/static/images/rule_registry.svg)

## 五.   属性

<div class="parameter-table">

| 属性名 | 类型 | 描述 | 访问级别 |
|----|----|----|----|
| \_rules | RuleDict | 存储所有已注册的规则函数，键为规则名称，值为规则函数 | 私有 |
| \_matchers | MatcherList | 存储所有已注册的匹配器函数，按注册顺序排列 | 私有 |

</div>

<div class="warning">

**注意:**   匹配器列表的元素顺序保持注册顺序，即注册顺序决定了规则匹配的优先级。

</div>

## 六.   方法

### 1.   初始化方法

```
__init__(self) -> None
```

初始化规则注册器实例，创建空的规则字典和匹配器列表。

### 2.   注册方法

```
_register_rule(self, name: str, func: RuleFunction) -> None
```

通过名称注册单个规则函数。

**参数:**

<div class="parameter-table">

| 参数名 | 类型         | 描述                                           |
|--------|--------------|------------------------------------------------|
| name   | str          | 规则的唯一标识符                               |
| func   | RuleFunction | 规则函数，接收表达式和上下文字典，返回转换结果 |

</div>

<div class="warning">

**注意:**   如果同名规则已存在，将被新注册的规则覆盖。

</div>

```
_register_matcher(self, matcher: MatcherFunction) -> None
```

注册单个匹配器函数。

**参数:**

<div class="parameter-table">

| 参数名  | 类型            | 描述                                   |
|---------|-----------------|----------------------------------------|
| matcher | MatcherFunction | 匹配器函数，检测表达式是否适用特定规则 |

</div>

```
_register_all_rules(self, rules: RuleDict) -> None
```

批量注册规则字典中的所有规则。

**参数:**

<div class="parameter-table">

| 参数名 | 类型     | 描述                                       |
|--------|----------|--------------------------------------------|
| rules  | RuleDict | 规则字典，键值对为规则名称到规则函数的映射 |

</div>

```
_register_all_matchers(self, matchers: MatcherList) -> None
```

批量注册匹配器列表中的所有匹配器。

**参数:**

<div class="parameter-table">

| 参数名   | 类型        | 描述           |
|----------|-------------|----------------|
| matchers | MatcherList | 匹配器函数列表 |

</div>

```
register_all(self, rules: RuleDict, matchers: MatcherList) -> None
```

一次性注册所有规则和匹配器，这是核心注册接口。

**参数:**

<div class="parameter-table">

| 参数名   | 类型        | 描述               |
|----------|-------------|--------------------|
| rules    | RuleDict    | 要注册的规则字典   |
| matchers | MatcherList | 要注册的匹配器列表 |

</div>

### 3.   规则匹配方法

```
get_applicable_rules(self, expr: Expr, context: RuleContext, ) -> RuleList
```

获取适用于给定表达式的所有规则函数。

**参数:**

<div class="parameter-table">

| 参数名  | 类型        | 描述                                 |
|---------|-------------|--------------------------------------|
| expr    | Expr        | 要匹配的 SymPy 表达式                |
| context | RuleContext | 用于计算的上下文字典，包含变量等信息 |

</div>

**返回值:** RuleList - 适用的规则函数列表

**匹配流程:**

1.  遍历所有已注册的匹配器
2.  对每个匹配器，使用表达式和上下文字典进行检测
3.  如果匹配器返回规则名称且在规则字典中存在，则添加对应规则
4.  返回所有匹配成功的规则函数列表

### 4.   工厂方法

```
@staticmethod create_common_rule(operation: Operation, func_name: str) -> RuleFunction
```

创建通用规则函数的工厂方法，适用于常见函数类型的变换规则。

**参数:**

<div class="parameter-table">

| 参数名    | 类型      | 描述                                       |
|-----------|-----------|--------------------------------------------|
| operation | Operation | 数学操作类型（如 Derivative、Integral 等） |
| func_name | str       | 函数名称，用于生成解释文本                 |

</div>

**返回值:** RuleFunction - 生成的规则函数

**生成的规则函数行为:**

- 对输入表达式执行指定的数学操作
- 计算结果并生成 LaTeX 格式的解释说明
- 返回结果和解释的元组

```
@staticmethod create_common_matcher(func: Union[exp, log, InverseTrigonometricFunction, TrigonometricFunction]) -> MatcherFunction
```

创建通用匹配器函数的工厂方法，用于检测特定函数类型。

**参数:**

<div class="parameter-table">

| 参数名 | 类型 | 描述 |
|----|----|----|
| func | Union\[exp, log, InverseTrigonometricFunction, TrigonometricFunction\] | 要匹配的函数类型 |

</div>

**返回值:** MatcherFunction - 生成的匹配器函数

**生成的匹配器函数行为:**

- 检查表达式是否为指定函数类型
- 检查函数的参数是否与上下文字典中的变量匹配
- 如果匹配成功，返回函数名称的小写形式作为规则标识符

## 七.   规则设计指南

1.  **单一职责**：每个规则应专注于一种特定的表达式模式

2.  **明确匹配条件**：匹配器应该精确描述适用条件，避免过度匹配

3.  **有意义的解释**：规则应提供清晰的教学性解释，便于理解计算过程

4.  **步骤最优考虑**：将处理特殊情况的规则放在匹配器列表前面，确保使用当前最优变换

## 八.   注意事项

1.  **名称字符串一致**：确保规则名称在匹配器和规则字典中保持一致

2.  **返回 None**：当无法处理表达式时，规则函数应返回 None，而不是抛出异常

3.  **规则的优先级**：特殊规则应先于通用规则注册

4.  **边界检测**：测试规则在各种边界条件下的行为

5.  **避免重复检测**：规则函数无需检测此规则对表达式是否可用，因为只要进入了某一个规则函数，则说明已通过对应匹配器函数的检测

## 九.   调试技巧

1.  在此类中调试

<div class="core-area">

在 get_applicable_rules 方法中添加 print(applicable) 语句以得到匹配结果

</div>

2.  在调用它的类中调试

<div class="core-area">

在 BaseCalculator 类的 \_apply_rule 方法中启用 print(f”rule: {rule.\_\_name\_\_}“) 语句

</div>
