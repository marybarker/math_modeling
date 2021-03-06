/*
	Mary Barker
	Homework 3

	Fractal on CPU
	to compile: nvcc BarkerHW3.cu -lm -lGL -lGLU -lglut
*/

#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>


#define A  -0.624
#define B  0.4351

unsigned int window_width = 1024;
unsigned int window_height = 1024;

float xMin = -2.0;
float xMax =  2.0;
float yMin = -2.0;
float yMax =  2.0;

float stepSizeX = (xMax - xMin)/((float)window_width - 1.0);
float stepSizeY = (yMax - yMin)/((float)window_height - 1.0);

float color (float x, float y) 
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
		return(1.0);
	}
	else
	{
		return(0.0);
	}
}

void display(void) 
{ 
	float *pixels; 
	float x, y;
	int k;

	pixels = (float *)malloc(window_width*window_height*3*sizeof(float));
	k=0;

	y = yMin;
	while(y <= yMax) 
	{
		x = xMin;
		while(x <= xMax) 
		{
			pixels[k] = 0.0;//color(x,y);	//Red on or off returned from color
			pixels[k+1] = .4*color(x,y);//0.0; 	//Green off
			pixels[k+2] = .4*color(x,y);//0.0;	//Blue off
			k=k+3;			//Skip to next pixel
			x += stepSizeX;
		}
		y += stepSizeY;
	}

	glDrawPixels(window_width, window_height, GL_RGB, GL_FLOAT, pixels); 
	glFlush(); 
}

int main(int argc, char** argv)
{ 
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(window_width, window_height);
   	glutCreateWindow("Fractals man, fractals.");
   	glutDisplayFunc(display);
   	glutMainLoop();
}

