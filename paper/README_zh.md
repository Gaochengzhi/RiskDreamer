# LaTeX 论文项目

<div align="center">
    
[![LaTeX](https://img.shields.io/badge/LaTeX-008080?style=for-the-badge&logo=LaTeX&logoColor=white)](https://www.latex-project.org/)
[![IEEE](https://img.shields.io/badge/IEEE-00629B?style=for-the-badge&logo=IEEE&logoColor=white)](https://www.ieee.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![VSCode](https://img.shields.io/badge/VSCode-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white)](https://code.visualstudio.com/)

[![English](https://img.shields.io/badge/README-English-blue?style=flat-square)](README.md)
[![中文](https://img.shields.io/badge/README-中文-red?style=flat-square)](README_zh.md)

</div>

<div align="center">
    <img src=".assets/IEEE_logo.png" height="200">
</div>

<div align="center">
    此仓库包含 IEEE 期刊投稿的 LaTeX 源代码。
</div>

## 为什么选择这个仓库？与官方 LaTeX 模板相比的特性

*   出版级别的排版（orcid链接，标题与图/表之间的间距，作者简介）
*   结构化的编译（可选的无图模式，分离的图、表和章节）
*   VSCode + Makefile 编译脚本


## 项目结构

```
manuscript/
├── main.tex                 # 主文档文件
├── section/                 # 各个章节
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── method.tex
│   ├── experiments.tex
│   ├── ablation.tex
│   ├── related_works.tex
│   └── conclusion.tex
├── fig/                     # 图片文件夹
├── table/                   # 表格文件
├── biography/               # 作者照片
├── bibtex/                  # 参考文献文件
├── .vscode/                 # VSCode 配置
└── Makefile                 # 构建自动化

```

## 环境要求

- LaTeX 发行版（TeX Live、MiKTeX 或 MacTeX）
- BibTeX 用于参考文献处理
- VSCode 配合 LaTeX Workshop 扩展（推荐）

## 使用方法

### 方法 1：VSCode（推荐）

1. 在 VSCode 中打开项目
2. 安装 LaTeX Workshop 扩展
3. 打开 `main.tex`
4. 按 `Ctrl+Alt+B` 构建文档
5. PDF 将自动在新标签页中打开
6. 编译后会自动清理临时文件

### 方法 2：使用 Makefile 命令行

```bash
# 编译文档
make

# 清理临时文件
make clean

# 清理所有文件（包括 PDF）
make distclean
```

### 方法 3：手动编译

```bash
# 标准 LaTeX 编译流程（包含参考文献）
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 功能特性

- **自动清理**：编译后自动删除临时文件
- **模块化结构**：章节组织在独立文件中，便于维护
- **参考文献管理**：集成 BibTeX 处理引用
- **图片组织**：图片集中存储在 `fig/` 目录中

## VSCode 配置

项目包含针对 LaTeX 编辑优化的 VSCode 设置：

- 保存时自动编译
- 自动清理辅助文件
- 集成 PDF 查看器
- 多种编译配方（latexmk、pdflatex+bibtex）

## 文件类型

### 源文件
- `.tex` - LaTeX 源文件
- `.bib` - 参考文献数据库文件

### 生成文件（自动清理）
- `.aux`、`.log`、`.out` - LaTeX 辅助文件
- `.bbl`、`.blg` - BibTeX 输出文件
- `.fls`、`.fdb_latexmk` - 构建系统文件
- `.synctex.gz` - SyncTeX 文件（编辑器集成用）

## 注意事项

- 主文档是 `main.tex`
- 所有章节通过 `\input{}` 命令包含
- 图片应放置在 `fig/` 目录中
- 参考文献条目位于 `bibtex/bib/` 目录中
- 作者简介和照片位于 `biography/` 目录中