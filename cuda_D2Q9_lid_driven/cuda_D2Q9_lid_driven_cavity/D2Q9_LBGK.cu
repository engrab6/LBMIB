#include "D2Q9.h"

int main(int argc, char ** argv){

	int nx = atoi(argv[1]); //256
	int ny = atoi(argv[2]); //256
	double uw = atof(argv[3]); //0.1
	double Re = atof(argv[4]); //400.0
	int t_max = atoi(argv[5]); //time_step
	int num_threads = NT;
	int L = ny;
	int t=0;

	double *f0, *f1, *f2, *f3, *f4;
	double *f5, *f6, *f7, *f8;

	double tau=3*L*uw/Re+0.5; // relaxation time for BGK

	float milliseconds = 0;
    cudaEvent_t start, stop;
    double calc_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	// Set size
	int size_mat = nx * ny;
	int mem_size_mat = sizeof(double)* size_mat;
	// CPU memory allocation
	f0 =(double *)malloc(mem_size_mat);
	f1 =(double *)malloc(mem_size_mat);
	f2 =(double *)malloc(mem_size_mat);
	f3 =(double *)malloc(mem_size_mat);
	f4 =(double *)malloc(mem_size_mat);
	f5 =(double *)malloc(mem_size_mat);
	f6 =(double *)malloc(mem_size_mat);
	f7 =(double *)malloc(mem_size_mat);
	f8 =(double *)malloc(mem_size_mat);

	unsigned int mem_size_mat_char = sizeof(char)* size_mat;
	char* geo =(char *)malloc(mem_size_mat_char);

	//GPU memory allocation
	double* f0_dev_Old = NULL;
	cudaMalloc((void**)&f0_dev_Old, mem_size_mat);
	double* f1_dev_Old = NULL;
	cudaMalloc((void**)&f1_dev_Old, mem_size_mat);
	double* f2_dev_Old = NULL;
	cudaMalloc((void**)&f2_dev_Old, mem_size_mat);
	double* f3_dev_Old = NULL;
	cudaMalloc((void**)&f3_dev_Old, mem_size_mat);
	double* f4_dev_Old = NULL;
	cudaMalloc((void**)&f4_dev_Old, mem_size_mat);
	double* f5_dev_Old = NULL;
	cudaMalloc((void**)&f5_dev_Old, mem_size_mat);
	double* f6_dev_Old = NULL;
	cudaMalloc((void**)&f6_dev_Old, mem_size_mat);
	double* f7_dev_Old = NULL;
	cudaMalloc((void**)&f7_dev_Old, mem_size_mat);
	double* f8_dev_Old = NULL;
	cudaMalloc((void**)&f8_dev_Old, mem_size_mat);

	double* f0_dev_New = NULL;
	cudaMalloc((void**)&f0_dev_New, mem_size_mat);
	double* f1_dev_New = NULL;
	cudaMalloc((void**)&f1_dev_New, mem_size_mat);
	double* f2_dev_New = NULL;
	cudaMalloc((void**)&f2_dev_New, mem_size_mat);
	double* f3_dev_New = NULL;
	cudaMalloc((void**)&f3_dev_New, mem_size_mat);
	double* f4_dev_New = NULL;
	cudaMalloc((void**)&f4_dev_New, mem_size_mat);
	double* f5_dev_New = NULL;
	cudaMalloc((void**)&f5_dev_New, mem_size_mat);
	double* f6_dev_New = NULL;
	cudaMalloc((void**)&f6_dev_New, mem_size_mat);
	double* f7_dev_New = NULL;
	cudaMalloc((void**)&f7_dev_New, mem_size_mat);
	double* f8_dev_New = NULL;
	cudaMalloc((void**)&f8_dev_New, mem_size_mat);

	char* geo_dev = NULL;
	cudaMalloc((void**)&geo_dev, mem_size_mat_char);

	//Initialize
	Init_Eq(nx,ny,f0,f1,f2,f3,f4,f5,f6,f7,f8);
	init_geo(nx,ny,geo);

	//Copy data from CPU to GPU
	cudaMemcpy(f0_dev_Old, f0, mem_size_mat, cudaMemcpyHostToDevice);
	cudaMemcpy(f1_dev_Old, f1, mem_size_mat, cudaMemcpyHostToDevice);
	cudaMemcpy(f2_dev_Old, f2, mem_size_mat, cudaMemcpyHostToDevice);
	cudaMemcpy(f3_dev_Old, f3, mem_size_mat, cudaMemcpyHostToDevice);
	cudaMemcpy(f4_dev_Old, f4, mem_size_mat, cudaMemcpyHostToDevice);
	cudaMemcpy(f5_dev_Old, f5, mem_size_mat, cudaMemcpyHostToDevice);
	cudaMemcpy(f6_dev_Old, f6, mem_size_mat, cudaMemcpyHostToDevice);
	cudaMemcpy(f7_dev_Old, f7, mem_size_mat, cudaMemcpyHostToDevice);
	cudaMemcpy(f8_dev_Old, f8, mem_size_mat, cudaMemcpyHostToDevice);

	cudaMemcpy(f0_dev_New, f0, mem_size_mat, cudaMemcpyHostToDevice);
	cudaMemcpy(f1_dev_New, f1, mem_size_mat, cudaMemcpyHostToDevice);
	cudaMemcpy(f2_dev_New, f2, mem_size_mat, cudaMemcpyHostToDevice);
	cudaMemcpy(f3_dev_New, f3, mem_size_mat, cudaMemcpyHostToDevice);
	cudaMemcpy(f4_dev_New, f4, mem_size_mat, cudaMemcpyHostToDevice);
	cudaMemcpy(f5_dev_New, f5, mem_size_mat, cudaMemcpyHostToDevice);
	cudaMemcpy(f6_dev_New, f6, mem_size_mat, cudaMemcpyHostToDevice);
	cudaMemcpy(f7_dev_New, f7, mem_size_mat, cudaMemcpyHostToDevice);
	cudaMemcpy(f8_dev_New, f8, mem_size_mat, cudaMemcpyHostToDevice);

	cudaMemcpy(geo_dev, geo, mem_size_mat_char, cudaMemcpyHostToDevice);

	//Define block and grid sizes
	dim3 threads(num_threads, 1, 1);
	dim3 grid1(nx/num_threads, ny);
	dim3 grid2(1, ny/num_threads);
	while(t <t_max)	{

		//Execute kernel collision_propagation
		cudaEventRecord(start);
		collision_propagation<<< grid1, threads >>>(nx, ny,
		num_threads, tau, uw, geo_dev, f0_dev_Old, f1_dev_Old,
		f2_dev_Old, f3_dev_Old, f4_dev_Old, f5_dev_Old,
		f6_dev_Old, f7_dev_Old, f8_dev_Old, f0_dev_New,
		f1_dev_New, f2_dev_New, f3_dev_New, f4_dev_New,
		f5_dev_New, f6_dev_New, f7_dev_New, f8_dev_New);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		calc_time += milliseconds;


		//Execute kernel exchange
		cudaEventRecord(start);
		exchange<<< grid2, threads >>>(nx, ny, num_threads,
		f3_dev_New, f6_dev_New, f7_dev_New,
		f1_dev_New, f5_dev_New, f8_dev_New);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		calc_time += milliseconds;

		t++;
	}

	//Copy results back to CPU
	cudaMemcpy(f0, f0_dev_Old,mem_size_mat, cudaMemcpyDeviceToHost);
	cudaMemcpy(f1, f1_dev_Old,mem_size_mat, cudaMemcpyDeviceToHost);
	cudaMemcpy(f2, f2_dev_Old,mem_size_mat, cudaMemcpyDeviceToHost);
	cudaMemcpy(f3, f3_dev_Old,mem_size_mat, cudaMemcpyDeviceToHost);
	cudaMemcpy(f4, f4_dev_Old,mem_size_mat, cudaMemcpyDeviceToHost);
	cudaMemcpy(f5, f5_dev_Old,mem_size_mat, cudaMemcpyDeviceToHost);
	cudaMemcpy(f6, f6_dev_Old,mem_size_mat, cudaMemcpyDeviceToHost);
	cudaMemcpy(f7, f7_dev_Old,mem_size_mat, cudaMemcpyDeviceToHost);
	cudaMemcpy(f8, f8_dev_Old,mem_size_mat, cudaMemcpyDeviceToHost);

	printf("calc_time=%f milliseconds\n", calc_time);

	cudaFree(f0_dev_Old);cudaFree(f1_dev_Old);cudaFree(f2_dev_Old);
	cudaFree(f3_dev_Old);cudaFree(f4_dev_Old);cudaFree(f5_dev_Old);
	cudaFree(f6_dev_Old);cudaFree(f7_dev_Old);cudaFree(f8_dev_Old);

	cudaFree(f0_dev_New);cudaFree(f1_dev_New);cudaFree(f2_dev_New);
	cudaFree(f3_dev_New);cudaFree(f4_dev_New);cudaFree(f5_dev_New);
	cudaFree(f6_dev_New);cudaFree(f7_dev_New);cudaFree(f8_dev_New);

	free(f0);free(f1);free(f2);
	free(f3);free(f4);free(f5);
	free(f6);free(f7);free(f8);
}

//=========================================================
//-------------------------------------------------------------------
// Subroutine: initialization with the equilibrium method
//------------------------------------------------------------------
//
void Init_Eq(int Nx, int Ny, double* f0,
	double* f1, double* f2, double* f3, double* f4,
	double* f5, double* f6, double* f7, double* f8){
	int i;
	double rho=rho0*0.1111111111111111111111;
	for (i=0;i<Nx*Ny;i++){
		f0[i]=rho*4;
		f1[i]=rho;
		f2[i]=rho;
		f3[i]=rho;
		f4[i]=rho;
		f5[i]=rho*0.25;
		f6[i]=rho*0.25;
		f7[i]=rho*0.25;
		f8[i]=rho*0.25;
	}
}

void init_geo(int nx, int ny,char* geo){
	int i;
	for(i=0;i<nx*ny;i++)
		geo[i] = FLUID;
	for(i=0;i<nx;i++)
		geo[i] = BOTTOM_WALL;
	for(i=0;i<ny;i++)
		geo[i*nx] = LEFT_WALL;
	for(i=0;i<ny;i++)
		geo[nx-1+i*nx] = RIGHT_WALL;
	for(i=0;i<nx;i++)
		geo[nx*(ny-1)+i] = TOP_WALL;
}
