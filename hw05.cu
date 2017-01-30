/*
	Mary Barker
	Homework 5

	Fractals using GPU 

	to compile: nvcc BarkerHW5.cu -lm -lGL -lGLU -lglut
*/

#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#define MIN(x, y) (x <= y) ? x : y

#define num_time_iterations 1600

unsigned int window_width = 1024;
unsigned int window_height = 1024;

float A = -0.624;
float B = 0.4351;

float xMin = -2.0;
float xMax =  2.0;
float yMin = -2.0;
float yMax =  2.0;

float stepSizeX = (xMax - xMin)/((float)window_width  - 1.0);
float stepSizeY = (yMax - yMin)/((float)window_height - 1.0);

dim3 dimGrid, dimBlock;
__device__ float color (float x, float y, float A, float B) 
{
	float mag,maxMag,t1;
	float maxCount = 200;
	float count = 0;
	maxMag = 10;
	mag = 0.0;
	
	while (mag < maxMag && count < maxCount) 
	{
		t1 = x;	
		x = x*x - y*y + A;
		y = (2.0 * t1 * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count < maxCount) 
	{
		return float(count) / float(maxCount);
	}
	else
	{
		return(0.0);
	}
}

__global__ void calculate_colors(float xmin, float dx, float ymin, float dy, int nx, int ny, float * pix, float A, float B)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	float x,y,color_val;
	if(i < nx * ny)
	{
		x = xmin + threadIdx.x * dx;
		y = ymin +  blockIdx.x * dy;
		color_val=color(x,y,A,B);
		pix[3*i+0] = 0.0;
		pix[3*i+1] = 0.0;
		pix[3*i+2] = 0.0;
		if(color_val > 0)
		{
			pix[3*i+1] = 0.5 - 1.5 * color_val;
			pix[3*i+2] = 0.5 + 1.5 * color_val;
		}
	}
}

void display(void) 
{ 
	float *pixels; 
	float * GPU_pixels;
	int updating = 0;
	float t;

	pixels = (float*)malloc(window_width*window_height*3*sizeof(float));
	cudaMalloc(&GPU_pixels, window_width*window_height*3*sizeof(float));

	while(updating++ < num_time_iterations)
	{

		cudaMemcpy(GPU_pixels, pixels, window_width*window_height*3*sizeof(float), cudaMemcpyHostToDevice);
		calculate_colors<<<dimGrid, dimBlock>>>(xMin,stepSizeX, yMin,stepSizeY, window_width, window_height, GPU_pixels, A, B);
		cudaMemcpy(pixels, GPU_pixels, window_width*window_height*3*sizeof(float), cudaMemcpyDeviceToHost);
		glDrawPixels(window_width, window_height, GL_RGB, GL_FLOAT, pixels); 
		glFlush(); 
		t = float(updating) / float(num_time_iterations);
		A =-0.5 + 0.15 * cos(M_PI * 2.0 * t);
		B = 0.5 + 0.15 * sin(M_PI * 2.0 * t);
	}
}

int main(int argc, char** argv)
{ 
	dimGrid = MIN(window_width, 1024);
	dimBlock = MIN(window_height, 1024);

   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(window_width, window_height);
   	glutCreateWindow("Fractals man, fractals.");
   	glutDisplayFunc(display);
   	glutMainLoop();
}

