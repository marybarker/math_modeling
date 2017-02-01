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

