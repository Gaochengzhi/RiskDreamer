# LaTeX Makefile for manuscript
MAIN = main
TEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	$(TEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof
	rm -f *.fls *.fdb_latexmk *.synctex.gz
	rm -f *~

distclean: clean
	rm -f $(MAIN).pdf