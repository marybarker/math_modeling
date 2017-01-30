HWSCRIPT = hw04.cu
TEXFILE = hw04.tex
RUNOUT = runfile
#######################################################################
.SUFFIXES: .cu .o .tex .pdf
.cu.o:
	$(GCC) $(COMPFLAGS) $*.cu
.tex.pdf:
	$(TEX) $*.tex

TEX = lualatex
GCC = nvcc
COMPFLAGS = -std=c++11
LDFLAGS = -lm -lGL -lGLU -lglut

$(RUNOUT) : $(HWSCRIPT)
	$(GCC) -o $@ $< $(COMPFLAGS) $(LDFLAGS)

texfile: $(TEXFILE)
	$(TEX) $<
clean: 
	rm *.out *.log *.aux runfile

