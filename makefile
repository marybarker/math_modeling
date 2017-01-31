####HWSCRIPT = hw05.cu
####TEXFILE = hw05.tex
####RUNOUT = runfile
####TEXOUT = hw05.pdf
##########################################################################
###name=$1
###TEXFILE="$name.tex"
###TEXOUT="$name.pdf"
###HWSCRIPT="$name.cu"
###
###.SUFFIXES: .cu .o .tex .pdf
###.cu.o:
###	$(GCC) $(COMPFLAGS) $*.cu
###.tex.pdf:
###	$(TEX) $*.tex
###
###TEX = lualatex
###GCC = nvcc
###COMPFLAGS = -std=c++11
###LDFLAGS = -lm -lGL -lGLU -lglut
###
###$(RUNOUT) : $(HWSCRIPT)
###	$(GCC) -o $@ $< $(COMPFLAGS) $(LDFLAGS)
###
###$(TEXOUT) : $(TEXFILE)
###	$(TEX) $(TEXFILE)
####texfile: $(TEXFILE)
####	$(TEX) $<
###clean: 
###	rm *.out *.log *.aux runfile

name=hw05
TEXTFILE = $(name).tex
CUDAFILE = $(name).cu
OUTCUDA = $(name).o

TEX = lualatex
GCC = nvcc

COMPFLAGS = -std=c++11
LDFLAGS = -lm -lGL -lGLU -lglut

tex: $(TEXTFILE)
	$(TEX) $(TEXTFILE)

cu: $(CUDAFILE)
	$(GCC) -o $(OUTCUDA) $< $(COMPFLAGS) $(LDFLAGS)

clean: 
	rm *.o *.log *.aux

