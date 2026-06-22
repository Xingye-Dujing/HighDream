---
title: "CommonMatrixCalculator - 矩阵计算基类文档"
output: "common_matrix_calculator.html"
---

# **CommonMatrixCalculator** (core/common_matrix_calculator.py)

矩阵计算器的抽象基类，为各种矩阵运算提供统一的框架和基础功能。它封装了矩阵输入解析、步骤记录、结果显示等通用操作，简化了具体矩阵计算器的开发。

<div class="warning">

**注意:**   出现的所有图中，箭头的方向无实际意义，上面的文字所指可能是前对后，也可能是后对前

</div>

## 一.   类概述

CommonMatrixCalculator 为矩阵计算提供了一套完整的基础设施：

- **步骤管理**：通过 MatrixStepGenerator 记录计算过程
- **输入解析**：支持多种格式的矩阵和向量输入
- **表达式简化**：自动简化矩阵元素中的符号表达式
- **输出格式化**：生成 LaTeX 格式的计算步骤

它与其它类的关系如下：

![](/static/images/system_architecture_2.svg)

## 二.   设计原则

- **一致性**：为所有矩阵计算器提供统一的接口和用户体验
- **灵活性**：支持多种输入格式，适应不同的使用场景
- **可扩展性**：作为抽象基类，易于添加新的矩阵计算功能
- **教学友好**：详细的步骤记录有助于理解矩阵计算过程

## 三.   输入格式 (parse_matrix_input 方法决定)

### 1.   矩阵输入格式

**嵌套列表格式:** \[\[1, 2\], \[3, 4\]\] - 2×2 矩阵

**逗号隔开的多列表格式:** \[1, 2\], \[3, 4\] - 2×2 矩阵

### 2.   向量输入格式

**列向量格式:** \[\[1\], \[2\], \[3\]\] - 标准列向量格式

**行向量格式:** \[1, 2, 3\] - 程序将自动转换为列向量

## 四.   属性

<div class="parameter-table">

| 属性名 | 类型 | 描述 | 访问级别 |
|----|----|----|----|
| step_generator | MatrixStepGenerator | 矩阵步骤生成器实例，负责管理计算步骤的记录和格式化输出 | 公共 |

</div>

<div class="warning">

**设计说明:** step_generator 使用专门的 MatrixStepGenerator 类来处理矩阵计算特有的步骤显示需求。与 BaseStepGenerator 相比, 主要差在矩阵计算不需要以连等式形式输出过程，且每步表达式与解释文本也不在同一行出现。

</div>

## 五.   方法

### 1.   方法初始化方法

```
__init__(self) -> None
```

初始化矩阵计算器实例，创建 MatrixStepGenerator 用于步骤管理。

**执行效果:** 创建 self.step_generator 实例

### 2.   方法步骤记录方法

```
add_step(self, title: str) -> None
```

向计算过程添加步骤标题，用于组织和描述计算流程。

**参数:**

<div class="parameter-table">

| 参数名 | 类型 | 描述                               | 默认值 |
|--------|------|------------------------------------|--------|
| title  | str  | 步骤的标题文本，将显示在计算过程中 | *必需* |

</div>

```
add_matrix(self, matrix: Matrix, name: str = "A") -> None
```

向步骤管理器中添加矩阵及其标识符。

**参数:**

<div class="parameter-table">

| 参数名 | 类型   | 描述                    | 默认值 |
|--------|--------|-------------------------|--------|
| matrix | Matrix | 要显示的 SymPy 矩阵对象 | *必需* |
| name   | str    | 矩阵的标识符名称        | “A”    |

</div>

```
add_vector(self, vector: Matrix, name: str = "x") -> None
```

向步骤管理器中添加向量及其标识符。

**参数:**

<div class="parameter-table">

| 参数名 | 类型   | 描述                          | 默认值 |
|--------|--------|-------------------------------|--------|
| vector | Matrix | 要显示的 SymPy 向量（列矩阵） | *必需* |
| name   | str    | 向量的标识符名称              | “x”    |

</div>

```
add_equation(self, equation: str) -> None
```

向步骤管理器中添加方程或 LaTeX 文本。

**参数:**

<div class="parameter-table">

| 参数名   | 类型 | 描述                  | 默认值 |
|----------|------|-----------------------|--------|
| equation | str  | 方程的 LaTeX 表示形式 | *必需* |

</div>

### 3.   输出方法

```
get_steps_latex(self) -> str
```

获取完整的计算步骤，以 LaTeX 格式返回。

**返回值:** str - 包含所有计算步骤的 LaTeX 代码

**适用场景:**

- 在网页中显示计算过程
- 在 Jupyter notebook 中渲染

### 4.   方法输入解析方法

```
parse_matrix_input(self, matrix_input: str) -> Matrix
```

将矩阵字符串解析为 SymPy 的 Matrix 对象。

**参数:**

<div class="parameter-table">

| 参数名       | 类型 | 描述             | 默认值 |
|--------------|------|------------------|--------|
| matrix_input | str  | 矩阵的字符串表示 | *必需* |

</div>

**返回值:** Matrix - 解析后的 SymPy 矩阵对象

**异常:** ValueError - 当输入字符串无法解析为有效矩阵时

<div class="warning">

**注意:**   此方法使用 SymPy 的 sympify 函数，支持所有 SymPy 认可的数字和符号表达式。

</div>

```
parse_vector_input(self, vector_input: str) -> Matrix
```

将向量字符串解析为 SymPy 的 Matrix 对象（列向量）。

**参数:**

<div class="parameter-table">

| 参数名       | 类型 | 描述             | 默认值 |
|--------------|------|------------------|--------|
| vector_input | str  | 向量的字符串表示 | *必需* |

</div>

**返回值:** Matrix - 解析后的 SymPy 列向量

**异常:** ValueError - 当输入字符串无法解析为有效向量时

### 4.   工具方法

```
simplify_matrix(self, matrix: Matrix) -> Matrix
```

化简矩阵中的每个元素。

**参数:**

<div class="parameter-table">

| 参数名 | 类型   | 描述                  | 默认值 |
|--------|--------|-----------------------|--------|
| matrix | Matrix | 需要简化的 SymPy 矩阵 | *必需* |

</div>

**返回值:** Matrix - 简化后的矩阵

**特性:**

- 对非零元素应用 SymPy 的 simplify 函数
- 适用于包含符号表达式的矩阵

## 六.   步骤记录指南

1.  **逻辑分组**：使用 add_step() 将相关操作分组

2.  **清晰标识**：为矩阵和向量使用有意义的名称

3.  **完整过程**：记录从输入到输出的完整计算流程

4.  **教学价值**：添加解释性步骤帮助理解数学原理

## 七.   错误处理

1.  **不可逆**：始终处理矩阵不可逆的情况

2.  **维度不匹配**：检查矩阵维度兼容性

3.  **输入格式有误**：验证输入格式的有效性

4.  **简化报错**：处理符号表达式可能出现的简化问题
