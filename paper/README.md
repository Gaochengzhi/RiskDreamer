# LaTeX Manuscript Project

<div align="center">
    
[![LaTeX](https://img.shields.io/badge/LaTeX-008080?style=for-the-badge&logo=LaTeX&logoColor=white)](https://www.latex-project.org/)
[![IEEE](https://img.shields.io/badge/IEEE-00629B?style=for-the-badge&logo=IEEE&logoColor=white)](https://www.ieee.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![VSCode](https://img.shields.io/badge/VSCode-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white)](https://code.visualstudio.com/)


</div>

<div align="center">
    <img src=".assets/IEEE_logo.png" height="200">
</div>

<div align="center">
    This repository contains the LaTeX source code for the IEEE journal submission.
</div>

<div align="center">

[中文版本](./README_zh.md)

</div>

## Why this repo? Features compared to the official LaTeX template

*   Publication-level typesetting (orcidlink, spacing between titles and figures/tables, biography)
*   Structured compilation (optional figure-free mode, separate figures, tables, and sections)
*   VSCode + Makefile compilation scripts


## Project Structure

```
manuscript/
├── main.tex                 # Main document file
├── section/                 # Individual sections
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── method.tex
│   ├── experiments.tex
│   ├── ablation.tex
│   ├── related_works.tex
│   └── conclusion.tex
├── fig/                     # Figures 
├── table/                   # Table files
├── biography/               # Author photos
├── bibtex/                  # Bibliography files
├── .vscode/                 # VSCode configuration
└── Makefile                 # Build automation

```

## Prerequisites

- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- BibTeX for bibliography processing
- VSCode with LaTeX Workshop extension (recommended)

## Usage

### Method 1: VSCode (Recommended)

1. Open the project in VSCode
2. Install the LaTeX Workshop extension
3. Open `main.tex`
4. Press `Ctrl+Alt+B` to build the document
5. The PDF will automatically open in a new tab
6. Temporary files are automatically cleaned after compilation

### Method 2: Command Line with Makefile

```bash
# Compile the document
make

# Clean temporary files
make clean

# Clean all files including PDF
make distclean
```

### Method 3: Manual Compilation

```bash
# Standard LaTeX compilation with bibliography
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Features

- **Automatic cleanup**: Temporary files are automatically removed after compilation
- **Modular structure**: Sections are organized in separate files for better maintainability
- **Bibliography management**: BibTeX integration for reference handling
- **Figure organization**: Centralized figure storage in `fig/` directory

## VSCode Configuration

The project includes optimized VSCode settings for LaTeX editing:

- Auto-compilation on save
- Automatic cleanup of auxiliary files
- Integrated PDF viewer
- Multiple compilation recipes (latexmk, pdflatex+bibtex)

## File Types

### Source Files
- `.tex` - LaTeX source files
- `.bib` - Bibliography database files

### Generated Files (Auto-cleaned)
- `.aux`, `.log`, `.out` - LaTeX auxiliary files
- `.bbl`, `.blg` - BibTeX output files
- `.fls`, `.fdb_latexmk` - Build system files
- `.synctex.gz` - SyncTeX files for editor integration

## Notes

- The main document is `main.tex`
- All sections are included via `\input{}` commands
- Figures should be placed in the `fig/` directory
- Bibliography entries are in `bibtex/bib/` directory
- Author biographies and photos are in the `biography/` directory

