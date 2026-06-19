# HighDream ([中文描述](./README.md))

## Project Overview

### What is this project?

HighDream is a symbolic computation tool based on **Python** and **SymPy**. It adopts a modular design and uses **breadth-first search (BFS) and rule-driven** approaches to automatically solve mathematical problems including differentiation, integration, limits, matrices, and equivalent transformations, while **providing step-by-step solutions**.

### Why was this project created?

When I was in high school, I struggled with the calculations in solid geometry and conic sections, and I kept thinking it would be great if a computer could help me compute automatically and show the process. I didn't have time to work on it in high school, and I forgot about it in my first year of university. So it was delayed until September this year, when I started building it at the beginning of my sophomore year. After nearly three months of work on and off, I finally finished the basic framework.

### Why is it named HighDream?

As mentioned above, this idea originated from my high school days `(The Dream of High School)`. I looked it up, and `high dream` means an intoxicating dream. Although NetEase Youdao Dictionary suggests it might imply being disconnected from reality, I only take the meaning of intoxication. **I hope that in the future, computers will have strong intelligence and promote the development of human theoretical knowledge.** Therefore, naming it `HighDream` represents both my high school idea and my vision for the future.

## Image Demos (running on Firefox in Debian)

1. Homepage

![Homepage](./docs/img/01.png)

2. Beautiful rendering of mathematical symbols

![Beautiful rendering of mathematical symbols](./docs/img/02.png)

3. Example of intermediate steps for integration

![Example of intermediate steps for integration](./docs/img/03.png)
![Example of intermediate steps for integration](./docs/img/10.png)
![Example of intermediate steps for integration](./docs/img/11.png)
![Example of intermediate steps for integration](./docs/img/12.png)
![Example of intermediate steps for integration](./docs/img/13.png)
![Example of intermediate steps for integration](./docs/img/14.png)

4. Example of intermediate steps for differentiation

![Example of intermediate steps for differentiation](./docs/img/04.png)

5. Example of intermediate steps for limits

![Example of intermediate steps for limits](./docs/img/05.png)
![Example of intermediate steps for limits](./docs/img/06.png)

6. Example of intermediate steps for matrices

![Example of intermediate steps for matrices](./docs/img/07.png)
![Example of intermediate steps for matrices](./docs/img/08.png)
![Example of intermediate steps for matrices](./docs/img/09.png)

7. Example of intermediate steps for equivalent transformations

![Example of intermediate steps for equivalent transformations](./docs/img/15.png)
![Example of intermediate steps for equivalent transformations](./docs/img/16.png)

8. UE-like blueprint usage

![UE-like blueprint usage](./docs/img/17.png)

9. Matrix analysis

![Matrix analysis](./docs/img/18.png)
![Matrix analysis](./docs/img/19.png)
![Matrix analysis](./docs/img/20.png)

10. Simple LaTeX editor and renderer with auto-completion

![LaTeX editor and renderer](./docs/img/21.png)
![LaTeX editor and renderer](./docs/img/22.png)
![LaTeX editor and renderer](./docs/img/23.png)

11. Command line display of function calls for current computation (facilitates problem localization and future extension)

![Command line](./docs/img/24.png)

12. Manual rule selection page (`/rule_select`)

![Manual rule selection page](./docs/img/25.png)

## Features

- Step-by-step solutions: displays the complete calculation process, facilitating teaching and understanding
- Rule-based pattern matching: rule application based on pattern matching, no AI involvement, high stability of output results
- Multi-domain support: differentiation, integration, limits, matrix operations, equivalent transformations
- LaTeX output: beautiful rendering of mathematical formulas
- Extensible architecture: easy to add new calculation rules and operation types
- Two ways to use: Python code calls in Jupyter Notebook; graphical web interface
- Manual rule selection: via `/rule_select` page, users can manually choose which rule to apply at each derivation step for differentiation, integration, and limits

## Development Tech Stack

- Programming language: Python 3.11.6
- Core libraries: SymPy (symbolic computation library), Flask (web visualization)
- Output format: LaTeX (rendered by MathJax)

## Architecture Design

This project mainly consists of **three** architectures:

1. Architecture for differentiation, integration, limits, and determinant calculations
2. Architecture for matrix-related operations
3. Architecture for frontend to call backend for computation

## Core Classes

- [**BaseCalculator**](https://high-dream.vercel.app/help/base_calculator) (`core/base_calculator.py`)
- [**BaseStepGenerator**](https://high-dream.vercel.app/help/base_step_generator) (`core/base_step_generator.py`)
- [**RuleRegistry**](https://high-dream.vercel.app/help/rule_registry) (`core/rule_registry.py`)
- [**CommonMatrixCalculator**](https://high-dream.vercel.app/help/common_matrix_calculator) (`core/common_matrix_calculator.py`)
- [**MatrixStepGenerator**](https://high-dream.vercel.app/help/matrix_step_generator) (`core/matrix_step_generator.py`)
- [**ManualStepSolver**](core/manual_step_solver.py) — Web UI manual rule selection orchestrator (wraps Select Calculator)

## Usage

- Python code calls: please refer to the examples in `main.ipynb` first. (Remember to install third-party libraries)
- Visual website:

  (1) Run `python app_local.py` (requires Python installation)
  
  (2) Run `high-dream.exe` (download from Releases page, not yet released)
  
  (3) Visit `https://high-dream.vercel.app/` (for experience only, computation speed is slow)

## Notes on Usage

0. Do not close the command line (black window) while using.
1. The multiplication operator `*` must never be omitted! `ab` will be parsed as a single symbol, not `a*b`.
2. Inverse trigonometric functions are `asin()`, `acos()`, `atan()`, not `arcsin()`, `arccos()`, `arctan()`.
3. Do not use `e**` or `e^`; use `exp()` instead. `e` will be parsed as a letter variable like `x`. `exp()` is the properly defined natural exponential function.

## For More Details, See the [Help Documentation](https://high-dream.vercel.app/help)

## Special Thanks
 
1. Heavily referenced [MathDF](https://mathdf.com/int/cn/). It is a god that can provide intermediate steps. This project has made some extensions based on the types of integrals that it cannot solve (such as the `f(x)*e^g(x)` type).
2. **AI** participated throughout the process. Sincere thanks to various free AI tools in China (such as Qianwen and DeepSeek).