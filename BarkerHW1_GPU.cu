/*
 	Mary Barker
	Homework 1

	Vector addition on GPU
	to compile: nvcc BarkerHW1_GPU.cu

	OUTPUTS: 

	N = 100
	Time in milliseconds= 0.026000000000000
	Last Values are A[99] = 198.000000000000000  B[99] = 99.000000000000000  C[99] = 297.000000000000000

	N = 600
	Time in milliseconds= 0.027000000000000
	Last Values are A[599] = 1198.000000000000000  B[599] = 599.000000000000000  C[599] = 1797.000000000000000

	N = 2000
	Time in milliseconds= 0.035000000000000
	Last Values are A[1999] = 3998.000000000000000  B[1999] = 1999.000000000000000  C[1999] = 5997.000000000000000

*/
#include <sys/time.h>
#include <stdio.h>

//Length of vectors to be added.
#define N 100  //if N is greater than dimBlock.x program will break

float *A_CPU, *B_CPU, *C_CPU; //CPU pointers

float *A_GPU, *B_GPU, *C_GPU; //GPU pointers

void AllocateMemory()
{					
	//Allocate Device (GPU) Memory, & allocates the value of the specific pointer/array
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaMalloc(&C_GPU,N*sizeof(float));

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
		A_CPU[i] = (float)2*i;	
		B_CPU[i] = (float)i;
	}
}

//Cleaning up memory after we are finished.
void CleanUp(float *A_CPU,float *B_CPU,float *C_CPU,float *A_GPU,float *B_GPU,float *C_GPU)  //free
{
	free(A_CPU); free(B_CPU); free(C_CPU);
	cudaFree(A_GPU); cudaFree(B_GPU); cudaFree(C_GPU);
}

//This is the kernel. It is the function that will run on the GPU.
//It adds vectors A and B then stores result in vector C
__global__ void Addition(float *A, float *B, float *C, int n)
{

	int id = blockIdx.x;
	
	if(id < n) C[id] = A[id] + B[id];
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

	//Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	
	//Calling the Kernel (GPU) function.	
	Addition<<<dim3(N), 1>>>(A_GPU, B_GPU, C_GPU, N);
	
	//Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);

	//Stopping the timer
	gettimeofday(&end, NULL);

	//Calculating the total time used in the addition and converting it to milliseconds.
	float time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	
	//Displaying the time 
	printf("Time in milliseconds= %.15f\n", (time/1000.0));	

	// Displaying vector info you will want to comment out the vector print line when your
	//vector becomes big. This is just to make sure everything is running correctly.	
	for(i = 0; i < N; i++)		
	{		
		//printf("A[%d] = %.15f  B[%d] = %.15f  C[%d] = %.15f\n", i, A_CPU[i], i, B_CPU[i], i, C_CPU[i]);
	}

	//Displaying the last value of the addition for a check when all vector display has been commented out.
	printf("Last Values are A[%d] = %.15f  B[%d] = %.15f  C[%d] = %.15f\n", N-1, A_CPU[N-1], N-1, B_CPU[N-1], N-1, C_CPU[N-1]);
	
	//You're done so cleanup your mess.
	CleanUp(A_CPU,B_CPU,C_CPU,A_GPU,B_GPU,C_GPU);	
	
	return(0);
}
