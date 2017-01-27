/* 
Mary Barker
HW 1 
This program takes vectors of (user-input) length and computes their dot product and sum. 
*/
#include <iostream>
using namespace std;

int n = 0;
float * A_CPU = NULL;
float * B_CPU = NULL;
float * C_CPU = NULL;
float * A_GPU = NULL;
float * B_GPU = NULL;
float * C_GPU = NULL;

void AllocateMemory()
{
	if( (A_CPU = (float*)malloc(n * sizeof(float))) == NULL)
	{
		cout << "Error: Could not allocate memory on CPU. Exiting" << endl;
                exit(0);
	}
	if( (B_CPU = (float*)malloc(n * sizeof(float))) == NULL)
	{
		cout << "Error: Could not allocate memory on CPU. Exiting" << endl;
                exit(0);
	}
	if( (C_CPU = (float*)malloc(n * sizeof(float))) == NULL)
	{
		cout << "Error: Could not allocate memory on CPU. Exiting" << endl;
                exit(0);
	}
	if( CudaMalloc(&A_GPU, n * sizeof(float)) == CudaErrorMemoryAllocation)
	{
		cout << "Error: Could not allocate memory on GPU. Exiting" << endl;
		exit(0);

	if( CudaMalloc(&B_GPU, n * sizeof(float)) == CudaErrorMemoryAllocation)
	{
		cout << "Error: Could not allocate memory on GPU. Exiting" << endl;
		exit(0);

	if( CudaMalloc(&C_GPU, n * sizeof(float)) == CudaErrorMemoryAllocation)
	{
		cout << "Error: Could not allocate memory on GPU. Exiting" << endl;
		exit(0);

	}
}
void InitializeVectors()
{
	for(int i = 0; i < n; i++)
	{
		A_CPU[i] = i;
		B_CPU[i] = 2*i;
	}
	cudaMemcpy(A_GPU, A_CPU, n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_GPU, B_CPU, n*sizeof(float), cudaMemcpyHostToDevice);
}
void FreeMemory()
{
	free(A_CPU, B_CPU, C_CPU);
        cudaFree(A_GPU);
        cudaFree(B_GPU);
        cudaFree(C_GPU);
}

__device__ void GPUAdd(float * A, float * B, float * C, int n)
{
	int id = ThreadIdx.x;
	if(id < n)
	{
		C[id] = A[id] + B[id];
	}
}
__device__ void GPUMultiply(float * A, float * B, float * C, int n)
{
	int id = ThreadIdx.x;
	if(id < n)
	{
		C[id] = A[id] * B[id];
	}
}


int main(void)
{
	int vec_len;

	cout << "Enter the length for vectors -> ";
        cin >> n;

	/* Allocate CPU and GPU arrays */
	AllocateMemory();

	/* Fill in vectors with values and copy to GPU */
	InitializeVectors();

	/* Add the two vectors together */
	<<n, 1>>GPUAdd(A_GPU, B_GPU, C_GPU);
	cudaMemcpy(C_CPU, C_GPU, n*sizeof(float), cudaMemcpyDeviceToHost);

	/* Compute the dot product of the two vectors */
	<<n, 1>>GPUMultiply(A_GPU, B_GPU, C_GPU);
	cudaMemcpy(C_CPU, C_GPU, n*sizeof(float), cudaMemcpyDeviceToHost);

	/* Free up arrays */
	FreeMemory();

	return(0);
}
