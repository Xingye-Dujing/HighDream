# help_docs/ — 帮助文档源文件与构建脚本

本目录把 `/help/*` 帮助页面的**内容**从 HTML 中抽出来，改为 Markdown
撰写，再用 [pandoc](https://pandoc.org/) 一键生成 Flask 服务的 HTML。

```
help_docs/
  src/                   # Markdown 源文件 (git-tracked)
    base_calculator.md
    base_step_generator.md
    rule_registry.md
    common_matrix_calculator.md
    matrix_step_generator.md
  template.html          # pandoc 包装模板 (git-tracked)
  build_help.py          # 构建/反向导入脚本 (git-tracked)
  build.bat              # Windows 快捷入口
  README.md              # 本文件

templates/help/          # 生成产物 (git-ignored)
  base_calculator.html
  ...
```

## 前置依赖

- `pandoc >= 3` 必须已加入 `PATH`（已配置）。
- `beautifulsoup4` 仅在运行 `md`（反向导入）命令时需要：
  `pip install beautifulsoup4`（项目已安装）。

## 常用命令

在项目根目录运行：

```bash
# 常规构建：把 src/*.md 编译为 templates/help/*.html
python help_docs/build_help.py html

# 一次性反向导入：把 templates/help/*.html 转回 src/*.md
# （仅用于首次迁移，或需要以 HTML 为准覆盖 .md 时）
python help_docs/build_help.py md

# 先 md 再 html（一般不需要）
python help_docs/build_help.py all
```

Windows 下也可以直接双击 `help_docs\build.bat`，等价于 `python help_docs/build_help.py html`。

## Markdown 源文件结构

每个 `.md` 文件必须包含 YAML frontmatter：

```yaml
---
title: "BaseCalculator - 符号计算器基类文档"
output: "base_calculator.html"
---
```

正文使用 pandoc-flavored Markdown，可用以下约定：

| 用途 | Markdown 写法 | 生成的 HTML |
|---|---|---|
| 一级/二级标题 | `#` / `##` | `<h1>` / `<h2>` |
| 警告框 | `::: warning` ... `:::` | `<div class="warning">` |
| 代码区域 | `::: core-area` ... `:::` | `<div class="core-area">` |
| 方法签名 | ` ```...``` ` 代码块 | `<pre><code>`（CSS 复用 `.method-signature` 风格） |
| 参数表格 | GFM 表格，外面包一层 `<div class="parameter-table">` | `<div class="parameter-table"><table>...</table></div>` |
| 内联代码 | `` `name` `` | `<code>name</code>` |
| 图片 | `![alt](/static/images/...)` | `<img ...>` |

## 工作流

1. 编辑 `help_docs/src/*.md`。
2. 运行 `python help_docs/build_help.py html`。
3. 启动 Flask（`python app_local.py`），访问 `/help/<name>` 查看效果。

**不要**手动修改 `templates/help/*.html` —— 它们会在下次构建时被覆盖。
