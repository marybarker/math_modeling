/*
 	Mary Barker
	Homework 6

	Vector dot product on GPU to compile: nvcc BarkerHW6.cu
*/
#include <sys/time.h>
#include <stdio.h>

#define N 10000  //if N is greater than dimBlock.x program will break
#define MIN(x,y) (x<y)?x:y

float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers

dim3 grid, block;

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

	block = MIN(1024, N);
	grid = (N > 1024) ? ((N - 1) / block.x + 1) : 1;
}

//Loads values into vectors that we will add.
void Innitialize()
{
	int i;
	for(i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)1;	
		B_CPU[i] = (float)1;
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
__global__ void DotProduct(float *A, float *B, float *C, int n)
{

	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int odd, offset = blockDim.x * blockIdx.x, new_n = blockDim.x;
	bool not_done_yet = true;
	
	if(id < n) C[id] = A[id] * B[id];

	// 'Fold' the vector in half repeatedly
	while(not_done_yet)
	{
		__syncthreads();
		odd = new_n % 2;
		new_n = new_n / 2;
		if(new_n > 0)
		{
			if(id < (offset + new_n))
			{
				if(id + new_n < n)
				{
					C[id] += C[id+new_n];
					if( (odd > 0) && (id < offset + 1) )
						C[id] += C[id+2*new_n];
				}
			}
		}
		else
		{
			not_done_yet = false;
		}
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

	//Copy Memory from CPU to GPU		
	cudaMemcpy(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);

	//Calling the Kernel (GPU) function.	
	DotProduct<<<grid, block>>>(A_GPU, B_GPU, C_GPU, N);
	
	//Copy Memory from GPU to CPU	
	cudaMemcpy(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);

	if(grid.x > 1)
	{
		for(i = 1; i < grid.x; i++)
		{
			C_CPU[0] += C_CPU[i*block.x];
		}
	}

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

	//Displaying the value of the dot product
	printf("Value is %f\n", C_CPU[0]);
	
	//You're done so cleanup your mess.
	CleanUp(A_CPU,B_CPU,C_CPU,A_GPU,B_GPU,C_GPU);	
	
	return(0);
}

