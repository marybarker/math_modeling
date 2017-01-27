/*
	Mary Barker
	Homework 1 

	Vector addition on CPU
	to compile: nvcc BarkerHW1_CPU.cu

	OUTPUTS: 
	N = 100
	CPU Time in milliseconds= 0.000000000000000
	Last Values are A[99] = 99.000000000000000  B[99] = 99.000000000000000  C[99] = 198.000000000000000

	N = 600
	CPU Time in milliseconds= 0.008000000000000
	Last Values are A[599] = 599.000000000000000  B[599] = 599.000000000000000  C[599] = 1198.000000000000000

	N = 2000
	CPU Time in milliseconds= 0.006000000000000
	Last Values are A[1999] = 1999.000000000000000  B[1999] = 1999.000000000000000  C[1999] = 3998.000000000000000

*/
#include <sys/time.h>
#include <stdio.h>

#define N 2000 

float *A_CPU, *B_CPU, *C_CPU; //CPU pointers

void AllocateMemory()
{					
	//Allocate Host (CPU) Memory
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
}

//Loads values into vectors that we will add.
void Innitialize()
{
	int i;
	
	for(i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)i;
	}
}

//Cleaning up memory after we are finished.
void CleanUp(float *A_CPU,float *B_CPU,float *C_CPU)
{
	free(A_CPU); free(B_CPU); free(C_CPU);
}

//Adds vectors A and B then stores result in vector C
void Addition(float *A, float *B, float *C, int n)
{
	int id;
	for(id = 0; id < n; id++)
	{ 
		C[id] = A[id] + B[id];
	}
}

int main()
{
	int i;
	timeval start, end;
	
	//Partitioning off the memory that you will be using.	
	AllocateMemory();

	//Loading up values to be added.
	Innitialize();

	//Starting the timer	
	gettimeofday(&start, NULL);

	//Add the two vectors
	Addition(A_CPU, B_CPU ,C_CPU, N);

	//Stopping the timer
	gettimeofday(&end, NULL);
	
	//Calculating the total time used in the addition and converting it to milliseconds
	float time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);

	// Displaying vector info you will want to comment out the vector print line when your
	//vector becomes big. This is just to make sure everything is running correctly.
	printf("CPU Time in milliseconds= %.15f\n", (time/1000.0));	
	for(i = 0; i < N; i++)		
	{		
		//printf("A[%d] = %.15f  B[%d] = %.15f  C[%d] = %.15f\n", i, A_CPU[i], i, B_CPU[i], i, C_CPU[i]);
	}

	//Displaying the last value of the addition for a check when all vector display has been commented out.
	printf("Last Values are A[%d] = %.15f  B[%d] = %.15f  C[%d] = %.15f\n", N-1, A_CPU[N-1], N-1, B_CPU[N-1], N-1, C_CPU[N-1]);

	//Your done so cleanup your mess.	
	CleanUp(A_CPU,B_CPU,C_CPU);	
	
	return(0);
}

