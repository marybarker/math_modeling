/*
	Barker Homework 8

	Finding the problem with GPU dot product

	To compile: nvcc dotProductRobustNot.cu -O3 -o dotProductRobustNot -lcudart
	To run: ./dotProductRobustNot lengthofvector sizeofblock
*/
#include <sys/time.h>
#include <stdio.h>

// max number of block 65535 
// max number of threads per block 1024
// max number of threads 67107840

//#define THREADSPERBLOCK 1024

int THREADSPERBLOCK;

int N; //Global that holds the length of the vectors. It will be loaded from the command line

double *A_CPU, *B_CPU, *C_CPU; //CPU pointers

double *A_GPU, *B_GPU, *C_GPU; //GPU pointers

dim3 dimBlock; //This variable will hold the Dimensions of your block
dim3 dimGrid; //This variable will hold the Dimensions of your grid

//Select the block and grid architecture for the threads on the GPU
void SetUpCudaDevices()
{	
	//Threads in a block
	dimBlock.x = THREADSPERBLOCK;
	dimBlock.y = 1;
	dimBlock.z = 1;
	
	//Blocks in a grid
	dimGrid.x = (N - 1)/dimBlock.x + 1; //Makes enough blocks to add the whole vector. If N is greater than dimBlock.x*65535 you are out of luck.
	dimGrid.y = 1;
	dimGrid.z = 1;
}

void AllocateMemory()
{					
	//Allocate Device (GPU) Memory, & allocates the value of the specific pointer/array
	cudaMalloc(&A_GPU,N*sizeof(double));
	cudaMalloc(&B_GPU,N*sizeof(double));
	cudaMalloc(&C_GPU,N*sizeof(double));

	//Allocate Host (CPU) Memory
	A_CPU = (double*)malloc(N*sizeof(double)); //(float*) to prevent from being a void
	B_CPU = (double*)malloc(N*sizeof(double));
	C_CPU = (double*)malloc(N*sizeof(double));
}

//Loads values into vectors that we will dot.
void Innitialize()
{
	int i;
	
	for(i = 0; i < N; i++)
	{		
		A_CPU[i] = (double)1;	
		B_CPU[i] = (double)1;
	}
}

//Cleaning up memory after we are finished.
void CleanUp(double *A_CPU,double *B_CPU,double *C_CPU,double *A_GPU,double *B_GPU,double *C_GPU)
{
	free(A_CPU); free(B_CPU); free(C_CPU);
	cudaFree(A_GPU); cudaFree(B_GPU); cudaFree(C_GPU);
}

//This is the kernel. It is the function that will run on the GPU.
__global__ void Addition(double *A, double *B, double *C, int n)
{	
	int id = blockDim.x*blockIdx.x + threadIdx.x;

	//Multiplying the vectors
	if(id < n)
	{
		C[id] = A[id] * B[id];
	}
	__syncthreads();

	int fold = blockDim.x;
	while(fold > 1)
	{  
		if(fold%2 != 0)  //Checking to see if the fold is even
		{
			if(threadIdx.x == 0 && id + (fold - 1) < n) //If fold is not even add the last element and now it is
			{
				fold = fold - 1;
				C[id] = C[id] + C[id + fold];
			}
		}
		
		fold = fold/2; //Fold the remaining parts in
		if(threadIdx.x < fold && id + fold < n)
		{
			C[id] = C[id] + C[id + fold];
		}
		__syncthreads();
	}
}

int main(int argc, char** argv)
{
	int i;
	
	N = atoi(argv[1]); //Reading the length of the vectors from the command line
	
	THREADSPERBLOCK = atoi(argv[2]); //Reading the size of the blocks
	
	timeval start, end;
	cudaError_t err;

	//Set the thread structure that you will be using on the GPU		
	SetUpCudaDevices();

	//Partitioning off the memory that you will be using.
	AllocateMemory();

	//Loading up values to be added.
	Innitialize();
	
	//Starting the timer
	gettimeofday(&start, NULL);

	//Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(double), cudaMemcpyHostToDevice);
	
	//Calling the Kernel (GPU) function.	
	Addition<<<dimGrid, dimBlock>>>(A_GPU, B_GPU, C_GPU, N);

	//Checking to see if the Kernel had any problems.
	err = cudaGetLastError();
	if (err != 0) 
	{
		printf("\n CUDA error = %s\n", cudaGetErrorString(err));
		return(1);
	}
	
	//Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, N*sizeof(double), cudaMemcpyDeviceToHost);

	//Stopping the timer
	gettimeofday(&end, NULL);

	//Calculating the total time used in the addition and converting it to milliseconds.
	float time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	
	//Displaying the time
	printf("GPU Time in milliseconds= %.15f\n", (time/1000.0));

	//Add up the remaining info to get the final dot product
	double dot = 0.0;	
	for(i = 0; i < N; i = i + dimBlock.x)	
	{		
		dot = dot + C_CPU[i];
	}

	//Displaying the dot product.
	printf("\n *** N = %d DotProduct = %.15f ***\n",N, dot);

	//You're done so cleanup your mess.	
	CleanUp(A_CPU,B_CPU,C_CPU,A_GPU,B_GPU,C_GPU);	
	
	return(0);
}
