//=========================================================
//---------------------------------------------------------
//         ----- Header file of the D2Q9 model -----
//---------------------------------------------------------
//File name: D2Q9.h
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#define Nx 4095     // number of cells in the x-direction
#define Ny 4095     // number of cells in the y-direction
#define	Nx1 (Nx+1)
#define Ny1 (Ny+1)
#define L (Ny+1)    // width of the cavity
#define Q 9        // number of discrete velocities
#define rho0 1.0    // initial density
#define ux0  0.0    // initial velocity component in x direction
#define uy0  0.0    // initial velocity component in y direction
#define uw  1.0
#define Re 800.0
#define TIME_STEP 1200
int cx[Q]={0, 1, 0, -1, 0, 1, -1, -1, 1};
int cy[Q]={0, 0, 1, 0, -1, 1, 1, -1, -1};

double f[Ny1][Nx1][Q]; //array of the distribution functions (DFs)
double f_post[Ny1][Nx1][Q]; // array of the post-collision DFs
double rho[Ny1][Nx1], ux[Ny1][Nx1], uy[Ny1][Nx1]; // arrays of fluid density and velocity
double tau;  // relaxation time for BGK model
// double s[Q]; // relaxation rates for MRT model
// double D[Q]={9, 36, 36, 6, 12, 6, 12, 4, 4};  // D = M*MT
double w[Q]={4.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/36,1.0/36,1.0/36,1.0/36}; //  the weights in the EDF
// int rc[Q]={0,3,4,1,2,7,8,5,6}; // index of reversed velocity

void Init_Eq(void);      //Initialization
double feq(double RHO, double U, double V, int k);  // Equilibrium distribution function
void Coll_BGK(void);     // BGK collision
// void Coll_MRT(void);     // MRT collision
// double meq(double RHO, double U, double V, int k);  // Equilibrium  momenta
void Streaming(void);    // Streaming
void Den_Vel(void);      // Fluid variables
void Bounce_back(void);  // Bounce-back boundary condition
double Err(void);        // Difference in velocity field
double u0[Ny1][Nx1],v0[Ny1][Nx1];
void Data_Output(void);  // Output simulation data
double get_cur_time();
//=========================================================

//=========================================================
int main(int argc, char **argv) {
  int k,M2,N2;
  double err;
  double t0,t1,calc_time=0;
  M2=Ny/2; N2=Nx/2;

  k=0;
  err=1.0;
  tau=3*L*uw/Re+0.5; // relaxation time for BGK
  // s[7]=s[8]=1.0/tau;
  // s[0]=s[3]=s[5]=0.0;
  // s[4]=s[6]=8*(2-s[7])/(8-s[7]);
  // s[1]=1.6;
  // s[2]=1.8; // relaxation rates for MRT

  Init_Eq();

  while(k<TIME_STEP) {
    k++;
    t0 = get_cur_time();
    Coll_BGK();    //BGK collision
//    Coll_MRT();  //MRT collision
    Streaming();   // Streaming
    Bounce_back(); // No-Slip boundary condition
    t1 = get_cur_time();
    calc_time += t1 - t0;
    // Den_Vel();     // Fluid variables

    // if(k%1000==0) {
    //   err=Err();   // Velocity differences between two successive 1000 steps
    //   printf("err=%f ux_center=%f  uy_center=%f k=%d\n",err,ux[M2][N2],uy[M2][N2], k);  // Display some results
    // }
  }
  Data_Output();   // Output simulation data
  printf("calc_time=%f\n", calc_time);
}


//=========================================================
//-------------------------------------------------------------------
// Subroutine: initialization with the equilibrium method
//------------------------------------------------------------------
//
void Init_Eq(){
	int j, i, k;
	for (j=0;j<=Ny;j++)
		for(i=0;i<=Nx;i++) {
			rho[j][i]=rho0;
			ux[j][i]=ux0;
			uy[j][i]=uy0;
			for(k=0;k<Q;k++)
				f[j][i][k]=feq(rho[j][i],ux[j][i],uy[j][i],k);
	}
}
//========================================================

//=========================================================
//-----------------------------------------------------------------
// Subroutine: calculation the equilibrium distribution
//----------------------------------------------------------------
//
double feq(double RHO, double U, double V, int k){
  double cu, U2;
  cu=cx[k]*U+cy[k]*V; // c k*u
  U2=U*U+V*V;         // u*u;
  return w[k]*RHO*(1.0+3.0*cu+4.5*cu*cu-1.5*U2);
}
//=========================================================

//=========================================================
//---------------------------------------------------------
// Subroutine: BGK collision
//---------------------------------------------------------
void Coll_BGK() {
	int j, i, k;
  	double FEQ;
    double tau_inv=1/tau;
  	for (j=0;j<=Ny;j++)
  		for(i=0;i<=Nx;i++)
  			for(k=0;k<Q;k++) {
    	FEQ=feq(rho[j][i],ux[j][i],uy[j][i],k);  //  EDF
    	f_post[j][i][k] = f[j][i][k]-(f[j][i][k]-FEQ)*tau_inv;
		// Post-collision DFs
	}
}
//=========================================================


//=========================================================
//---------------------------------------------------------
// Subroutine: Streaming
//---------------------------------------------------------
void Streaming() {
  int j, i, jd, id, k;
  for (j=0;j<=Ny;j++)
  	for(i=0;i<=Nx;i++)
  		for(k=0;k<Q;k++) {
		  	jd=j-cy[k]; id=i-cx[k]; // upwind node
			if(jd>=0 && jd<=Ny && id>=0 && id<=Nx) // fluid node
		        f[j][i][k]=f_post[jd][id][k]; // streaming
	   }
}
//=========================================================

//=========================================================
//---------------------------------------------------------
// Subroutine: Bounce-back scheme
//---------------------------------------------------------
void Bounce_back(){
  int i,j;
  //  j=Ny: top plate use Zou-He Velocity and Pressure BCs
  for(i=0;i<=Nx;i++)  {
    f[Ny][i][4]=f_post[Ny][i][2];
    // f[Ny][i][7]=f_post[Ny][i][5]+6*rho[Ny][i]*w[7]*cx[7]*uw;
    f[Ny][i][7]=f_post[Ny][i][5]-0.5*(f_post[Ny][i][1]-f_post[Ny][i][3])
    			+rho[Ny][i]*uw/6;
    // f[Ny][i][8]=f_post[Ny][i][6]+6*rho[Ny][i]*w[8]*cx[8]*uw;
    f[Ny][i][8]=f_post[Ny][i][6]+0.5*(f_post[Ny][i][1]-f_post[Ny][i][3])
    			+rho[Ny][i]*uw/6;
  }
  //  j=0: bottom plate
  for(i=0;i<=Nx;i++)  {
     f[0][i][2]=f_post[0][i][4];
     f[0][i][5]=f_post[0][i][7];
     f[0][i][6]=f_post[0][i][8];
  }

  //  i=0: left wall
  for(j=0;j<=Ny;j++)  {
     f[j][0][1]=f_post[j][0][3];
     f[j][0][5]=f_post[j][0][7];
     f[j][0][8]=f_post[j][0][6];
  }

  //  i=Nx: right wall
  for(j=0;j<=Ny;j++)  {
     f[j][Nx][3]=f_post[j][Nx][1];
     f[j][Nx][7]=f_post[j][Nx][5];
     f[j][Nx][6]=f_post[j][Nx][8];
  }

}
//=========================================================


//=========================================================
//------------------------------------------------------------
// Subroutine: Fluid variables (density and velocity)
//------------------------------------------------------------
void Den_Vel(){
  int j, i;
  for(j=0;j<=Ny;j++)
  	for(i=0;i<=Nx;i++)  {
		rho[j][i]=f[j][i][0]+f[j][i][1]+f[j][i][2]+f[j][i][3]
		+f[j][i][4]+f[j][i][5]+f[j][i][6]+f[j][i][7]+
		f[j][i][8];

		ux[j][i]=(f[j][i][1]+f[j][i][5]+f[j][i][8]-f[j][i][3]-
		f[j][i][6]-f[j][i][7])/rho[j][i];

		uy[j][i]=(f[j][i][5]+f[j][i][6]+f[j][i][2]-f[j][i][7]-
		f[j][i][8]-f[j][i][4])/rho[j][i];
  }
}

//=========================================================

double Err()  // Calculating the relative difference in velocity between two steps
{
  int j, i;
  double e1,e2;
    e1=e2=0.0;
  for(j=1;j<Ny;j++)
  	for(i=0;i<Nx;i++)  {
    e1+=sqrt((ux[j][i]-u0[j][i])*(ux[j][i]-u0[j][i])
		+(uy[j][i]-v0[j][i])*(uy[j][i]-v0[j][i]));
    e2+=sqrt(ux[j][i]*ux[j][i]+uy[j][i]*uy[j][i]);
    u0[j][i]=ux[j][i];
    v0[j][i]=uy[j][i];
  }
  return e1/e2;
}


void  Data_Output() {// Output data
	int i,j;
	FILE *fp;

	fp=fopen("x.dat","w+");
	for(i=0;i<=Nx;i++)
		fprintf(fp,"%e \n", (i+0.5)/L);
	fclose(fp);

	fp=fopen("y.dat","w+");
	for(j=0;j<=Ny;j++)
		fprintf(fp,"%e \n", (j+0.5)/L);
	fclose(fp);

	fp=fopen("ux.csv","w");
	for(j=0;j<=Ny;j++) {
	  for (i=0; i<=Nx; i++)
	  	fprintf(fp,"%e,",ux[j][i]);
	  fprintf(fp,"\n");
	}
	fclose(fp);

	fp=fopen("uy.csv","w");
	for(j=0;j<=Ny;j++){
	  for (i=0; i<=Nx; i++)
	  	fprintf(fp,"%e,",uy[j][i]);
	  fprintf(fp,"\n");
	}
	fclose(fp);

	fp=fopen("rho.csv","w");
	for(j=0;j<=Ny;j++){
	  for (i=0; i<=Nx; i++)
	  	fprintf(fp,"%e,",rho[j][i]);
	  fprintf(fp,"\n");
	}
	fclose(fp);
}

double get_cur_time() {
  struct timeval   tv;
  struct timezone  tz;
  double cur_time;

  gettimeofday(&tv, &tz);
  cur_time = tv.tv_sec + tv.tv_usec / 1000000.0;

  return cur_time;
}
