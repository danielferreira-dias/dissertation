# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Master's Dissertation** for MEIA (Master's in Artificial Intelligence Engineering) at ISEP/Polytechnic of Porto. The dissertation investigates multi-agent systems for dermatological diagnostics using Small Language Models (SLMs), Vision-Language Models (VLMs), and Retrieval-Augmented Generation (RAG), with a focus on privacy-preserving, resource-efficient alternatives to large language models.

## Build Commands

```bash
# Build the PDF (generates main.pdf)
make

# Clean build artifacts
make clean

# Full cleanup (includes biber cache and build directory)
make clean-all
```

**Direct latexmk command:**
```bash
latexmk -outdir=build -auxdir=build -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make main.tex
```

## Required Tools

- **pdflatex**, **makeglossaries**, **biber**, **latexmk** (all must be in PATH)
- A LaTeX distribution (MiKTeX on Windows, TeX Live on Linux/Mac)

## Document Structure

```
main.tex                 # Main document entry point
meia-style.cls           # MEIA dissertation class file (do not modify unless necessary)
mainbibliography.bib     # BibTeX bibliography database
frontmatter/
  frontmatter.tex        # Title page, abstracts, TOC, lists
  glossary.tex           # Acronyms and glossary definitions
ch1/, ch2/, ch3/...      # Chapter directories (one .tex file per chapter)
appendices/              # Appendix content
images/                  # Image assets
build/                   # Generated build artifacts (gitignored)
```

## Key Configuration

- **Document class options** (in main.tex line 30-44): font size (11pt), language (english/portuguese), spacing (singlespacing), parskip
- **Citation style**: authoryear-comp (Harvard-like) via biblatex/biber
- **Glossaries**: Uses `makenoidxglossaries` for Overleaf compatibility

## Thesis Metadata

Edit these in `main.tex` (lines 77-101):
- `\thesistitle{}` - Dissertation title
- `\author{}` - Candidate name
- `\supervisor{}` - Advisor name
- `\cosupervisor{}` - Co-advisor (optional)
- `\keywords{}` - Up to 6 keywords

## Working with Content

- **Adding chapters**: Create a new `chN/` directory with `chapterN.tex`, then uncomment/add `\input{chN/chapterN}` in main.tex
- **Adding bibliography entries**: Edit `mainbibliography.bib`
- **Adding acronyms/glossary terms**: Edit `frontmatter/glossary.tex`
- **Images**: Store in `images/` directory, reference with `\includegraphics`

## Language Switching

To switch between English and Portuguese:
1. Comment/uncomment the language option in main.tex (lines 33-34)
2. Run `make clean` before rebuilding (cached files may cause issues)

## Available Skills

This project uses Claude Code skills located in `.claude/` for academic writing support:

### `/literature-review`
Conduct systematic literature reviews using multiple databases (PubMed, arXiv, bioRxiv, Semantic Scholar). Use for:
- Multi-database systematic search with PRISMA-compliant methodology
- Thematic synthesis of findings
- PDF generation from markdown

### `/citation-management`
Manage citations and references. Use for:
- Converting DOIs to BibTeX entries
- Searching Google Scholar and PubMed
- Validating citation accuracy
- Formatting and cleaning BibTeX files

### `/scientific-writing`
Write scientific manuscripts in flowing prose. Use for:
- IMRAD structure guidance
- Citation style formatting (APA, Nature, Vancouver)
- Converting outlines to full paragraphs

## Citation Management Scripts

```bash
# Convert DOI to BibTeX
python .claude/citation-management/scripts/doi_to_bibtex.py <doi>

# Format and clean BibTeX file
python .claude/citation-management/scripts/format_bibtex.py mainbibliography.bib \
  --deduplicate --sort year --output clean_bibliography.bib

# Validate BibTeX citations
python .claude/citation-management/scripts/validate_citations.py mainbibliography.bib

# Verify DOIs in markdown and generate citation report
python .claude/literature-review/scripts/verify_citations.py <markdown_file>
```

**Python dependencies:**
```bash
pip install requests bibtexparser biopython
```

## Writing Guidelines

1. **Write in full paragraphs** - Never submit bullet points in final manuscript sections
2. **Two-stage process**: First create outlines with key points, then convert to flowing prose
3. **Thematic synthesis** - Organize literature findings by themes, not study-by-study
4. **Verify all citations** - Run validation scripts before finalizing
5. **Follow PRISMA guidelines** for systematic review sections
