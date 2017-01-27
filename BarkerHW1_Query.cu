/*
	Mary Barker
	Homework 1

	to compile: nvcc BarkerHW1_Query.cu

OUTPUT (When run on GreyJoy): 

   --- General Information for device 0 ---
	Name: GeForce GTX 580
	Compute capability: 2.0
	Clock rate: 1600000
	Device copy overlap: Enabled
	Kernel execution timeout : Enabled
   --- Memory Information for device 0 ---
	Total global mem: 1542324224
	Total constant mem: 65536
	Max mem pitch: 2147483647
	Texture Alignment: 512
   --- MP Information for device 0 ---
	Multiprocessor count: 16
	Shared mem per mp: 49152
	Registers per mp: 32768
	Threads in warp: 32
	Max threads per block: 1024
	Max thread dimensions: (1024, 1024, 64)
	Max grid dimensions: (65535, 65535, 65535)

   --- General Information for device 1 ---
	Name: GeForce GTX 580
	Compute capability: 2.0
	Clock rate: 1600000
	Device copy overlap: Enabled
	Kernel execution timeout : Disabled
   --- Memory Information for device 1 ---
	Total global mem: 1545469952
	Total constant mem: 65536
	Max mem pitch: 2147483647
	Texture Alignment: 512
   --- MP Information for device 1 ---
	Multiprocessor count: 16
	Shared mem per mp: 49152
	Registers per mp: 32768
	Threads in warp: 32
	Max threads per block: 1024
	Max thread dimensions: (1024, 1024, 64)
	Max grid dimensions: (65535, 65535, 65535)
*/
#include <stdio.h>

int main(void){
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount( &count );
	for(int i = 0; i < count; i++)
	{
		cudaGetDeviceProperties( &prop, i );
		printf("   --- General Information for device %d ---\n", i);
		printf( "Name: %s\n", prop.name);
		printf( "Compute capability: %d.%d\n", prop.major, prop.minor);
		printf( "Clock rate: %d\n", prop.clockRate);
		printf( "Device copy overlap: ");
		if(prop.deviceOverlap)
			printf("Enabled\n");
		else
			printf("Disabled\n");
		printf("Kernel execution timeout : ");
		if(prop.kernelExecTimeoutEnabled)
			printf("Enabled\n");
		else
			printf("Disabled\n");

		printf("   --- Memory Information for device %d ---\n", i);
		printf("Total global mem: %ld\n", prop.totalGlobalMem);
		printf("Total constant mem: %ld\n", prop.totalConstMem);
		printf("Max mem pitch: %ld\n", prop.memPitch);
		printf("Texture Alignment: %ld\n", prop.textureAlignment);
		printf("   --- MP Information for device %d ---\n", i);
		printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
		printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
		printf("Registers per mp: %d\n", prop.regsPerBlock);
		printf("Threads in warp: %d\n", prop.warpSize);
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("\n");
	}

}

