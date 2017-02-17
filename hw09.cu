/*
	Mary Barker HW 9 

	to compile and run: 
		nvcc Barker9.cu -lm -lGL -lGLU -lglut
		./a.out
*/
#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define INF 2e10f
#define SPHERES 20
#define rnd( x ) (x * rand() / RAND_MAX) 
#define MIN(x,y) (x< y) ? x : y
#define xmin -50
#define xmax  50
#define ymin -50
#define ymax  50

float * colors = NULL; 
float * pixels = NULL; 
float * radius = NULL;
float * x = NULL;
float * y = NULL;
float * z = NULL;
float * GPUcolors = NULL; 
float * GPUpixels = NULL; 
float * GPUradius = NULL;
float * GPUx = NULL;
float * GPUy = NULL;
float * GPUz = NULL;

unsigned int window_width = 1024;
unsigned int window_height = 1024;

float stepSizeX = (xmax - xmin)/((float)window_width - 1.0);
float stepSizeY = (ymax - ymin)/((float)window_height - 1.0);

dim3 nthreads = MIN(window_width, 1024);
dim3 nblocks = (window_width*window_height - 1) / nthreads.x + 1;

__device__ float hit(float x, float y, float z, float radius, float ox, float oy, float *n ) {
	float dx = ox - x;
	float dy = oy - y;

	if (dx*dx + dy*dy < radius*radius) {
		float dz = sqrtf( radius * radius - dx*dx - dy*dy );
        	*n = dz / sqrtf( radius * radius );
		return dz + z;
	}
	return -INF;
} 

__global__ void trace_rays(float * c, float * x, float * y, float * z, float * pix, float * r, float dx, float dy, int nx, int ny) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	float xx, yy, maxz=-INF;

	if(i < nx * ny) {

		float rr = 0, gg = 0, bb = 0;

		xx = (xmin + threadIdx.x * dx);
		yy = (ymin +  blockIdx.x * dy);

		for(int j = 0; j < SPHERES; j++){

			float n, t = hit(x[j], y[j], z[j], r[j], xx, yy, &n);

			if(t > maxz){
				rr = n * c[j];
				gg = n * c[j];
				bb = n * c[j];
				maxz = t;
			}
		}
		pix[3*i+0] = rr;
		pix[3*i+1] = gg;
		pix[3*i+2] = bb;
	}
}

void allocate_memory() {
	radius = 	(float*)malloc(  SPHERES * sizeof(float));
	x = 		(float*)malloc(  SPHERES * sizeof(float));
	y = 		(float*)malloc(  SPHERES * sizeof(float));
	z = 		(float*)malloc(  SPHERES * sizeof(float));
	colors = 	(float*)malloc(  SPHERES * sizeof(float));
	pixels = 	(float*)malloc(3*window_width*window_height * sizeof(float));

	cudaMalloc(&GPUradius, 	SPHERES * sizeof(float));
	cudaMalloc(&GPUx, 	SPHERES * sizeof(float));
	cudaMalloc(&GPUy, 	SPHERES * sizeof(float));
	cudaMalloc(&GPUz, 	SPHERES * sizeof(float));
	cudaMalloc(&GPUcolors, 	SPHERES * sizeof(float));
	cudaMalloc(&GPUpixels, 3*window_width*window_height * sizeof(float));

	for(int i = 0; i < SPHERES; i++){
		x[i] = 		rnd(100.0f) - 50;
		y[i] = 		rnd(100.0f) - 50;
		z[i] = 		rnd(100.0f) - 50;
		colors[i] = 	rnd(1.0);
		radius[i] = 	rnd(10.0f) + 2;
	}
	cudaMemcpy(GPUradius,	radius,		SPHERES*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(GPUx,	x,		SPHERES*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(GPUy,	y,		SPHERES*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(GPUz,	z,		SPHERES*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(GPUcolors,	colors,		SPHERES*sizeof(float),cudaMemcpyHostToDevice);
}

void display(void) { 
	allocate_memory();
	trace_rays<<<nblocks,nthreads>>>(GPUcolors, GPUx, GPUy, GPUz, GPUpixels, GPUradius, stepSizeX, stepSizeY, window_width, window_height);
	cudaMemcpy(pixels, GPUpixels, 3*window_width*window_height*sizeof(float), cudaMemcpyDeviceToHost);
	glDrawPixels(window_width, window_height, GL_RGB, GL_FLOAT, pixels); 
	glFlush(); 
}

int main(int argc, char** argv) { 
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(window_width, window_height);
   	glutCreateWindow("Fractals man, fractals.");
   	glutDisplayFunc(display);
   	glutMainLoop();
}

