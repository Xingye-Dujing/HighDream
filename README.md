# HighDream ([English version description](./README-EN.md))

## 项目概述

### 这个项目是什么？

HighDream 是一个基于 **Python** 和 **SymPy** 的符号计算工具。它采用模块化设计，通过**广度优先搜索(BFS)和规则驱动**，实现对微分、积分、极限、矩阵和等价改写等数学问题的自动化求解，并**提供求解过程**。

### 为什么创建这个项目？

我在上高中时苦于立体几何和圆锥曲线的计算，就一直在想如果计算机能够帮助我自动计算并提供计算过程就好了。高中没有时间去搞这个，大一又忘了这件事。于是拖到今年 9 月，大二开学后终于开始了它的制作。前前后后将近 3 个月，断断续续地终于完成了它的大概框架。

### 为什么命名为 HighDream？

如上所述，这个想法源于高中时期 `(The Dream of High School)`。我查了一下，`high dream` 意味着令人陶醉的梦想。虽然网易有道词典提示这可能意味着无法接受现实，但我只取其陶醉之意。**我希望未来计算机能拥有强大的智能，促进人类理论知识的发展。** 因此，命名为 `HighDream` 既代表了我的高中想法，也代表了我对未来的幻想。

## 图片演示 (在 Debian 的火狐浏览器上运行)

1. 首页

![首页](./docs/img/01.png)

2. 数学符号美观渲染

![数学符号美观渲染](./docs/img/02.png)

3. 积分给出的中间过程示例

![积分给出的中间过程示例](./docs/img/03.png)
![积分给出的中间过程示例](./docs/img/10.png)
![积分给出的中间过程示例](./docs/img/11.png)
![积分给出的中间过程示例](./docs/img/12.png)
![积分给出的中间过程示例](./docs/img/13.png)
![积分给出的中间过程示例](./docs/img/14.png)

4. 微分给出的中间过程示例

![微分给出的中间过程示例](./docs/img/04.png)

5. 极限给出的中间过程示例

![极限给出的中间过程示例](./docs/img/05.png)
![极限给出的中间过程示例](./docs/img/06.png)

6. 矩阵给出的中间过程示例

![矩阵给出的中间过程示例](./docs/img/07.png)
![矩阵给出的中间过程示例](./docs/img/08.png)
![矩阵给出的中间过程示例](./docs/img/09.png)

7. 等价改写给出的中间过程示例

![等价改写给出的中间过程示例](./docs/img/15.png)
![等价改写给出的中间过程示例](./docs/img/16.png)

8. 仿 UE 的蓝图使用方式

![仿 UE 的蓝图使用方式](./docs/img/17.png)

9. 矩阵分析

![矩阵分析](./docs/img/18.png)
![矩阵分析](./docs/img/19.png)
![矩阵分析](./docs/img/20.png)

10. 简易 LaTeX 编辑渲染器,内含自动补全功能

![LaTeX 编辑渲染器](./docs/img/21.png)
![LaTeX 编辑渲染器](./docs/img/22.png)
![LaTeX 编辑渲染器](./docs/img/23.png)

11. 命令行对当前计算调用函数情况的显示（便于后续问题的定位和功能的拓展）

![命令行](./docs/img/24.png)

12. 手动规则选择页面 (`/rule_select`)(在 Windows 的谷歌浏览器上运行)

![手动规则选择页面](./docs/img/01.gif)
![手动规则选择页面](./docs/img/29.png)

13. 蓝图版手动规则选择/全规则遍历 (`/method_tree`)(在 Windows 的谷歌浏览器上运行)

    - 右键拖拽画布平移视图
    - 左键拖动节点移动位置
    - 拖动节点边缘改变节点大小
    - 左键双击节点取消连线
    - 支持逐步确认每个规则是否应用，并实时显示求解进度
    - 支持对整棵方法树进行全规则自动遍历（异步执行，可取消）

![蓝图版手动规则选择](./docs/img/25.png)
![蓝图版手动规则选择](./docs/img/26.png)
![蓝图版手动规则选择](./docs/img/27.png)
![蓝图版手动规则选择](./docs/img/28.png)

## 特性

- 逐步求解：显示完整的计算过程，便于教学和理解
- 规则模式匹配：基于模式匹配的规则应用，没有 AI 参与，输出结果稳定性高
- 多领域支持：微分、积分、极限、矩阵运算、等价改写
- LaTeX 输出：美观地渲染数学公式
- 可扩展架构：易于添加新的计算规则和操作类型
- 两种使用方式：Jupyter Notebook 中 Python 代码调用；图形化网页界面
- 手动规则选择：通过 `/rule_select` 页面，用户可以手动选择每一步要应用的规则，支持微分、积分、极限三个领域
- 全规则遍历方法树：通过 `/method_tree` 页面，可以以蓝图画布的形式枚举并展示每一步所有可应用的规则分支，便于查看完整的求解空间

## 开发技术栈

- 编程语言：Python 3.11.6
- 核心库：SymPy（符号计算库）、Flask（Web 可视化）
- 输出格式：LaTeX（由 MathJax 渲染）

## 架构设计

本项目主要由**四个**架构组成：

1. 用于微分、积分、极限和行列式计算的架构（`BaseCalculator` + `BaseStepGenerator` + `RuleRegistry`）
2. 用于矩阵相关操作的架构（`CommonMatrixCalculator` + `MatrixStepGenerator`）
3. 用于前端调用后端进行计算的架构（Flask 蓝图 + `task_manager` 异步任务）
4. 用于手动规则选择与全规则遍历方法树的架构（`BaseManualStepSolver` 及各领域子类、`MethodTreeEnumerator`）

## 核心类

- [**BaseCalculator**](https://high-dream.vercel.app/help/base_calculator) (`core/base_calculator.py`)
- [**BaseStepGenerator**](https://high-dream.vercel.app/help/base_step_generator) (`core/base_step_generator.py`)
- [**RuleRegistry**](https://high-dream.vercel.app/help/rule_registry) (`core/rule_registry.py`)
- [**CommonMatrixCalculator**](https://high-dream.vercel.app/help/common_matrix_calculator) (`core/common_matrix_calculator.py`)
- [**MatrixStepGenerator**](https://high-dream.vercel.app/help/matrix_step_generator) (`core/matrix_step_generator.py`)
- [**BaseManualStepSolver**](https://high-dream.vercel.app/help/base_manual_step_solver) — 手动规则选择的基础编排器，封装了各领域共用的选择/回退/状态管理逻辑
- [**MethodTreeEnumerator**](https://high-dream.vercel.app/help/method_tree_enumerator) — 全规则遍历方法树的核心枚举器，递归展开所有可应用规则分支

## 使用方法

- Python 代码调用：请先参考 `main.ipynb` 中的示例。（记得安装第三方库）
- 可视化网站：

  （1）运行 `python app_local.py` （需要安装 Python）

  （2）运行 `high-dream.exe` （Release 页下载）

  （3）访问 `https://high-dream.vercel.app/` （国内需翻墙且此方式仅供体验, 计算速度很慢）

## 使用注意事项

0. 在使用时，不要退出命令行（黑框窗口）
1. 乘法运算符 `*` 绝对不能省略！ `ab` 会被解析为一个单独的符号，而不是 `a*b`
2. 反三角函数是 `asin()`, `acos()`, `atan()`，不是 `arcsin()`, `arccos()`, `arctan()`
3. 不要用 `e**` 或 `e^` 而是要用 `exp()`. `e` 会被解析成一个类似于 `x` 的字母变量。`exp()` 才是定义好的自然指数函数。

## 更多详情见帮助文档

https://high-dream.vercel.app/help （需翻墙）

## 特别致谢

1. 参考了 [MathDF](https://mathdf.com/int/cn/) 很多。它是能给出中间过程的神。本项目依据它无法解决的积分类型作了些拓展（比如 `f(x)*e^g(x)` 型）。
2. **AI** 全过程参与。衷心感谢国内各种免费 AI 工具（比如千问和 DeepSeek）。后面补：因为 Qoder 每天有 200 次的免费问答额度，所以我于 2026 年 6 月 19 日开始使用 Qoder CLI。什么时候它没有免费额度了，什么时候我就不再用 Qoder CLI 了。6 月 22 日结束 Qoder CLI 的使用。