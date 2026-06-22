---
title: "BaseStepGenerator - 步骤生成器文档"
output: "base_step_generator.html"
---

# **BaseStepGenerator** (core/base_step_generator.py)

步骤生成器的抽象基类，负责记录、管理和格式化符号计算的逐步求解过程。它为数学表达式的分步计算提供了统一的输出框架。

<div class="warning">

**注意:**   出现的所有图中，箭头的方向无实际意义，上面的文字所指可能是前对后，也可能是后对前

</div>

## 一.   类概述

BaseStepGenerator 是一个轻量级的步骤管理组件，主要功能包括：

- **步骤记录**：顺序存储计算过程中的每个表达式状态
- **解释管理**：为每个步骤提供可选的文字说明
- **格式输出**：支持多种输出格式（列表、LaTeX）
- **状态管理**：支持重置以进行多次计算

它与其它类的关系如下：

![BaseStepGenerator](/static/images/base_step_generator.svg)

## 二.   设计原则

- **简单性**：保持接口简洁，易于理解和使用
- **灵活性**：支持无解释步骤和有解释步骤的混合使用
- **可扩展性**：作为抽象基类，允许子类定制输出格式
- **独立性**：与具体计算逻辑解耦，专注于步骤管理

## 三.   属性

<div class="parameter-table">

| 属性名         | 类型         | 描述                                  | 访问级别 |
|----------------|--------------|---------------------------------------|----------|
| steps          | List\[Expr\] | 存储计算过程中每个步骤的 SymPy 表达式 | 公共     |
| \_explanations | List\[str\]  | 存储每个步骤对应的解释说明文本        | 受保护   |

</div>

<div class="warning">

**注意:**   steps 和 \_explanations 列表始终保持相同长度，每个步骤的表达式和解释通过索引对应。

</div>

## 四.   方法

### 1.   初始化与状态管理

```
__init__(self) -> None
```

初始化步骤生成器实例，创建空的步骤列表和解释列表。

**执行效果:**

- 创建空的 `self.steps` 列表
- 创建空的 `self._explanations` 列表

```
reset(self) -> None
```

重置步骤生成器的内部状态，清除所有记录的步骤和解释，准备进行新的计算。

**使用场景:**

- 在开始新的计算任务前调用
- 防止多次计算的结果相互干扰

### 2.   步骤记录方法

```
add_step(self, expr: Expr, explanation: str = "") -> None
```

添加一个新的计算步骤及其可选的解释说明。

**参数:**

<div class="parameter-table">

| 参数名      | 类型 | 描述                         | 默认值   |
|-------------|------|------------------------------|----------|
| expr        | Expr | 当前步骤的 SymPy 表达式      | *必需*   |
| explanation | str  | 对当前步骤的文字说明（可选） | 空字符串 |

</div>

**执行效果:**

- 将 `expr` 添加到 `self.steps` 列表末尾
- 将 `explanation` 添加到 `self._explanations` 列表末尾

### 3.   数据获取方法

```
get_steps(self) -> Tuple[List[Expr], List[str]]
```

获取所有记录的步骤和对应的解释说明。

**返回值:**

<div class="parameter-table">

| 位置 | 类型         | 描述                         |
|------|--------------|------------------------------|
| 0    | List\[Expr\] | 所有步骤的 SymPy 表达式列表  |
| 1    | List\[str\]  | 所有步骤的解释说明字符串列表 |

</div>

**适用场景:** 需要程序化处理计算步骤的场景

### 4.   格式输出方法

```
to_latex(self) -> str
```

将所有步骤格式化为 LaTeX 对齐环境字符串。

**返回值:** 完整的 LaTeX 代码字符串

**输出格式(连等式形式):**

- 使用 `align` 环境进行数学对齐
- 第一行以 `&` 开头，后续行以 `&=` 开头
- 每个步骤之间用 `\newline` 分隔
- 如果有解释说明，使用 `\quad \text{...}` 格式添加

<div class="warning">

**搭配 Jupyter notebook:** 可使用 display(Math(latex_str)) 渲染返回的 LaTeX

</div>

## 五.   使用示例

### 基本用法

BaseStepGenerator 被 BaseCalculator 及其子类用作步骤记录组件

### 自定义步骤生成器

如果需要不同的输出格式，可以创建 BaseStepGenerator 的子类。比如 LimitStepGenerator、DetStepGenerator 和 RefStepGenerator 类。

**最佳实践:**

- 在每次计算前调用 `reset()` 方法确保状态清洁
- 为关键步骤提供有意义的解释说明，增强可读性
- 根据使用场景选择合适的输出格式（程序处理用 `get_steps()`，展示用 `to_latex()`）
- 考虑创建自定义子类来满足特定的格式化需求
