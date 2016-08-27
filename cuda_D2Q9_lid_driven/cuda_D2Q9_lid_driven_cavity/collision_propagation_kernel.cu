#include "D2Q9.h"

__global__ void collision_propagation(int nx, int ny, int num_threads,
double tau, double uw, char* geoD, double *f0_Old,
double *f1_Old, double *f2_Old, double *f3_Old, double *f4_Old,
double *f5_Old, double *f6_Old, double *f7_Old, double *f8_Old,
double *f0_New,
double *f1_New, double *f2_New, double *f3_New, double *f4_New,
double *f5_New, double *f6_New, double *f7_New, double *f8_New){
	//Setup indexing
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int xStart = tx + bx*num_threads;
	int yStart = by;
	int k = nx*yStart + xStart;
	//Allocate shared memory
	__shared__ double F1_OUT[NT];
	__shared__ double F2_OUT[NT];
	__shared__ double F3_OUT[NT];
	__shared__ double F4_OUT[NT];
	__shared__ double F5_OUT[NT];
	__shared__ double F6_OUT[NT];
	__shared__ double F7_OUT[NT];
	__shared__ double F8_OUT[NT];

	// __shared__ double F1_OUT[NT+1];
	// __shared__ double F2_OUT[NT+1];
	// __shared__ double F3_OUT[NT+1];
	// __shared__ double F4_OUT[NT+1];
	// __shared__ double F5_OUT[NT+1];
	// __shared__ double F6_OUT[NT+1];
	// __shared__ double F7_OUT[NT+1];
	// __shared__ double F8_OUT[NT+1];

	double F0_IN=f0_Old[k];
	double F1_IN=f1_Old[k];
	double F2_IN=f2_Old[k];
	double F3_IN=f3_Old[k];
	double F4_IN=f4_Old[k];
	double F5_IN=f5_Old[k];
	double F6_IN=f6_Old[k];
	double F7_IN=f7_Old[k];
	double F8_IN=f8_Old[k];

	double rho,vx,vy,square,origin_rho;
	double f_eq0,f_eq1,f_eq2,f_eq3,f_eq4,f_eq5,f_eq6,f_eq7,f_eq8;
	double tau_inv=1/tau;

	//Check if it is a fluid or boundary node
	if (geoD[k] == FLUID){
	//Collisioin
		rho=F0_IN + F1_IN + F2_IN + F3_IN + F4_IN + F5_IN + F6_IN + F7_IN + F8_IN;
		vx=(F1_IN - F3_IN + F5_IN + F8_IN - F6_IN - F7_IN) / rho;
		vy=(F2_IN - F4_IN + F5_IN + F6_IN - F7_IN - F8_IN) / rho;
		square =1.5*(vx * vx + vy * vy);
		f_eq0 =4. /9.* rho *(1. - square);

		origin_rho = rho;
		rho*=0.1111111111111111111111;
		f_eq1=rho * (1. + 3.0 * vx + 4.5 * vx * vx - square);
		f_eq3=f_eq1 - 6.0 * vx * rho;
		f_eq2=rho * (1. + 3.0 * vy + 4.5 *vy * vy - square);
		f_eq4=f_eq2 - 6.0 * vy * rho;

		rho*=0.25;
		f_eq5=rho * (1. + 3.0 * (vx + vy) + 4.5 * (vx + vy) * (vx + vy) - square);
		f_eq7=f_eq1 - 6.0 * (vx + vy) * rho;
		f_eq6=rho * (1. + 3.0 * (vy - vx) + 4.5 * (vy - vx) * (vy - vx) - square);
		f_eq8=f_eq2 - 6.0 * (vy - vx) * rho;

		F0_IN +=(f_eq0 - F0_IN) * tau_inv;
		F1_IN +=(f_eq1 - F1_IN) * tau_inv;
		F2_IN +=(f_eq2 - F2_IN) * tau_inv;
		F3_IN +=(f_eq3 - F3_IN) * tau_inv;
		F4_IN +=(f_eq4 - F4_IN) * tau_inv;
		F5_IN +=(f_eq5 - F5_IN) * tau_inv;
		F6_IN +=(f_eq6 - F6_IN) * tau_inv;
		F7_IN +=(f_eq7 - F7_IN) * tau_inv;
		F8_IN +=(f_eq8 - F8_IN) * tau_inv;
	}
	else if (geoD[k] == TOP_WALL){
	//Velocity boundary condition on Top wall
		F4_OUT[tx]=F2_IN;
		F7_OUT[tx]=F5_IN-0.5*(F1_IN-F3_IN)+origin_rho*uw/6;
    	F8_OUT[tx]=F6_IN+0.5*(F1_IN-F3_IN)+origin_rho*uw/6;
	}
	else if (geoD[k] == BOTTOM_WALL){
	//Bottom_Wall boundary condition
		F2_OUT[tx]=F4_IN;
		F5_OUT[tx]=F7_IN;
		F5_OUT[tx]=F8_IN;
	}
	else if (geoD[k] == LEFT_WALL){
	//Left_Wall boundary condition
		F1_OUT[tx]=F3_IN;
		F5_OUT[tx]=F7_IN;
		F8_OUT[tx]=F6_IN;
	}
 	else if (geoD[k] == RIGHT_WALL){
	//Right_Wall boundary condition
		F3_OUT[tx]=F1_IN;
		F7_OUT[tx]=F5_IN;
		F6_OUT[tx]=F8_IN;
	}
	else{
		printf("Error!\n");
	}

	//Write to shared memory and Propagation
	if (tx ==0)	{
		F1_OUT[tx+1]=F1_IN;
		F3_OUT[num_threads-1]=F3_IN;
		F5_OUT[tx+1]=F5_IN;
		F6_OUT[num_threads-1]=F6_IN;
		F7_OUT[num_threads-1]=F7_IN;
		F8_OUT[tx+1]=F8_IN;
	}
	else if (tx==num_threads-1) {
		F1_OUT[0]= F1_IN;
		F3_OUT[tx-1]=F3_IN;
		F5_OUT[0]= F5_IN;
		F6_OUT[tx-1]=F6_IN;
		F7_OUT[tx-1]=F7_IN;
		F8_OUT[0]= F8_IN;
	}
	else{
		F1_OUT[tx+1]=F1_IN;
		F3_OUT[tx-1]=F3_IN;
		F5_OUT[tx+1]=F5_IN;
		F6_OUT[tx-1]=F6_IN;
		F7_OUT[tx-1]=F7_IN;
		F8_OUT[tx+1]=F8_IN;
	}
	//Synchronize
	__syncthreads();
	//Write to global memory
	f0_New[k]= F0_IN;
	f1_New[k]=F1_OUT[tx];
	f3_New[k]=F3_OUT[tx];
	if (by < ny-1) {
		k = nx*(yStart + 1) + xStart;
		f2_New[k]= F2_IN;
		f5_New[k]=F5_OUT[tx];
		f6_New[k]=F6_OUT[tx];
	}
	if (by > 0)	{
		k = nx*(yStart-1) + xStart;
		f4_New[k]= F4_IN;
		f7_New[k]=F7_OUT[tx];
		f8_New[k]=F8_OUT[tx];
	}
}
