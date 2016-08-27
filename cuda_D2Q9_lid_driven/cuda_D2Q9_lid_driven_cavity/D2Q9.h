#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define NT 16

#define rho0 1.0    // initial density
#define ux0  0.0    // initial velocity component in x direction
#define uy0  0.0    // initial velocity component in y direction
// #define uw  0.1
// #define Re 400.0

#define FLUID 0
#define TOP_WALL 1
#define BOTTOM_WALL 2
#define LEFT_WALL 3
#define RIGHT_WALL 4


void Init_Eq(int Nx, int Ny, double* f0,
double* f1, double* f2, double* f3, double* f4,
double* f5, double* f6, double* f7, double* f8);
void init_geo(int nx, int ny,char* geo);

__global__ void collision_propagation(int nx, int ny, int num_threads,
double tau, double uw, char* geoD, double *f0_Old,
double *f1_Old, double *f2_Old, double *f3_Old,
double *f4_Old, double *f5_Old, double *f6_Old,
double *f7_Old, double *f8_Old,
double *f0_New,
double *f1_New, double *f2_New, double *f3_New, double *f4_New,
double *f5_New, double *f6_New, double *f7_New, double *f8_New);

__global__ void exchange(int nx, int ny, int num_threads,
double *f3_New, double *f6_New, double *f7_New,
double *f1_New, double *f5_New, double *f8_New);
