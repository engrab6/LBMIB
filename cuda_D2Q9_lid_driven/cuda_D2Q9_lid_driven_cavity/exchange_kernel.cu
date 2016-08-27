#include "D2Q9.h"

__global__ void exchange(int nx, int ny, int num_threads,
double *f3_New, double *f6_New, double *f7_New,
double *f1_New, double *f5_New, double *f8_New){
	//Setup indexing
	int nbx=nx / num_threads;
	int num_threads1 = blockDim.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int bx;
	int xStart , yStart;
	int xStartW , xTargetW;
	int xStartE , xTargetE;
	int kStartW , kTargetW;
	int kStartE , kTargetE;
	//Exchange across boundaries
	for (bx=0; bx<nbx-1; bx++)	{
		xStart = bx*num_threads;
		xStartW = xStart + 2*num_threads - 1;
		xTargetW = xStartW - num_threads;
		yStart = (by) * num_threads1 + tx;
		kStartW = nx * yStart + xStartW;
		kTargetW = nx * yStart + xTargetW;
		f3_New[kTargetW] = f3_New[kStartW];
		f6_New[kTargetW] = f6_New[kStartW];
		f7_New[kTargetW] = f7_New[kStartW];
	}
	for (bx=nbx-2; bx>=0; bx--)	{
		xStart = bx*num_threads;
		xStartE = xStart;
		xTargetE = xStartE+num_threads;
		yStart = (by)*num_threads1 +  tx;
		kStartE = nx*yStart + xStartE;
		kTargetE = nx*yStart + xTargetE;
		f1_New[kTargetE] = f1_New[kStartE];
		f5_New[kTargetE] = f5_New[kStartE];
		f8_New[kTargetE] = f8_New[kStartE];
	}
}
