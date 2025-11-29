# HighDream Notebook

# 更多详情见可视化网页上的帮助文档

## 项目概述

### 这个项目是什么？

HighDream Notebook 是一个基于 **Python** 和 **SymPy** 的符号计算工具。它采用模块化设计，通过 **广度优先搜索(BFS)算法和规则驱动** 方法，实现对微分、积分、极限、矩阵和等价形式等数学问题的自动化求解，并**提供求解过程**。

### 为什么创建这个项目？

我在上高中时苦于立体几何和圆锥曲线的计算，就一直在想如果计算机能够帮助我自动计算并提供计算过程就好了。高中没有时间去搞这个，大一又忘了这件事。于是拖到今年 9 月，大二开学后终于开始了它的制作。前前后后将近 3 个月，断断续续地终于完成了它的大概框架。

### 为什么命名为 HighDream？

如上所述，这个想法源于高中时期 **(The Dream of High School)**。"high dream" 意味着令人陶醉的梦想。虽然网易有道词典提示这可能意味着无法接受现实，但我只取其陶醉之意。**我希望未来计算机能拥有强大的智能，促进人类理论知识的发展。** 因此，命名为 **HighDream** 既代表了我的高中想法，也代表了我对未来的幻想。

### 特性

- **逐步求解**：显示完整的计算过程，便于教学和理解
- **规则模式匹配**：基于模式匹配的规则应用
- **多领域支持**：微分、积分、极限、矩阵运算、等价变换
- **LaTeX 输出**：美观地渲染数学公式
- **可扩展架构**：易于添加新的计算规则和操作类型
- **两种使用方式**：Jupyter Notebook 中的 Python 代码调用；图形化网页界面

### 技术栈

- **编程语言**：Python 3.11.6
- **核心库**：SymPy（符号计算库）、Flask（Web 可视化）、matplotlib（绘制表达式推导树）、IPython
- **输出格式**：LaTeX

### 特别致谢

完全在 **AI** 辅助下完成。衷心**感谢**各种国内 AI 工具。

## 架构设计

本项目主要由**三个**架构组成：

### (1) 用于微分、积分、极限和行列式的计算

### (2) 用于矩阵相关操作

### (3) 用于前端调用后端计算

## 核心类

- **BaseCalculator** (`core/base_calculator.py`)
- **BaseStepGenerator** (`core/base_step_generator.py`)
- **RuleRegistry** (`core/rule_registry.py`)
- **CommonMatrixCalculator** (`core/common_matrix_calculator.py`)
- **MatrixStepGenerator** (`core/matrix_step_generator.py`)

## 使用方法

- **Python 代码调用**：请先参考 **main.ipynb** 中的示例。（记得安装第三方库）
- **可视化网站**：运行 **app.py 或 HighDream.exe**。(我将录视频演示)

# HighDream Notebook

# More details can be found in the help documentation on the visual website

## Project Overview

### What is this project?

HighDream Notebook is a symbolic computation tool based on **Python** and **SymPy**. It adopts a modular design, implementing automated solving of mathematical problems such as **differentiation, integration, limits, matrices, and equivalent forms** through **BFS algorithm and rule-driven** approaches, while **providing the solution process**.

### Why did I create this project?

In high school, I struggled with calculations in solid geometry and conic sections, always wishing computers could help me compute automatically and provide calculation processes. With no time to work on it during high school and forgetting about it during freshman year, I finally started development after entering my sophomore year in September. After nearly 3 months of intermittent work, I completed the basic framework.

### Why is it named HighDream?

As mentioned above, this idea originated in high school **(The Dream of High School)**. "high dream" translates to an intoxicating dream. While Youdao Dictionary suggests it might mean inability to accept reality, here I only take its meaning of being intoxicated. **I envision a future where computers can possess strong intelligence to promote the development of human theoretical knowledge.** Therefore, naming it **HighDream** represents both my high school ideal and my fantasy.

### Features

- **Step-by-step solving**: Shows complete calculation process for teaching and understanding
- **Rule pattern matching**: Rule application based on pattern matching
- **Multi-domain support**: Differentiation, integration, limits, matrix operations, equivalence transformations
- **LaTeX output**: Beautiful rendering of mathematical formulas
- **Extensible architecture**: Easy to add new calculation rules and operation types
- **Two usage methods**: Python code calls in Jupyter Notebook; graphical web interface

### Tech Stack

- **Programming Language**: Python 3.11.6
- **Core Libraries**: SymPy (symbolic computation library), Flask (web visualization), matplotlib (drawing expression derivation trees), IPython
- **Output Format**: LaTeX

### Special Acknowledgments

Completed entirely with **AI** assistance. Sincere **thanks** to various domestic AIs.

## Architecture Design

This project mainly consists of **three** architectures:

### (1) Used for differentiation, integration, limits, and determinants

### (2) Used for matrix-related operations

### (3) Used for frontend calling backend calculations

## Core Classes

- **BaseCalculator** (core/base_calculator.py)
- **BaseStepGenerator** (core/base_step_generator.py)
- **RuleRegistry** (core/rule_registry.py)
- **CommonMatrixCalculator** (core/common_matrix_calculator.py)
- **MatrixStepGenerator** (core/matrix_step_generator.py)

## Usage

- **Python code invocation**: Please refer to the examples in **main.ipynb** first. (Remember to install third-party libraries)
- **Visual website**: Run **app.py or HighDream.exe**. (I will record a video demonstration)
