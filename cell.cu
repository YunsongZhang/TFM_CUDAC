/* amotph.cu */

/* TSP */

#include <math.h>               /* standard include files */
#include <stdlib.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <sys/time.h>

#include <Xm/ArrowB.h>          /* Motif include files for widgets */
#include <Xm/DrawingA.h>
#include <Xm/Form.h>
#include <Xm/Frame.h>
#include <Xm/Label.h>
#include <Xm/Text.h>
#include <Xm/PushB.h>
#include <Xm/RowColumn.h>
#include <Xm/Scale.h>
#include <Xm/ToggleB.h>
#include <X11/keysym.h>         /* X include file */
#include "nrutil.h"         /* Numerical Recipies */
#include "nrutil.c"         /* Numerical Recipies */
#define TSTRING(s)      XmRString, s, strlen (s) + 1

#define NRANSI
#define TOL 2.0e-4
#define TOLF 1.0e-6
#define TINY 1.0e-20;
#define SWAP(a,b) itemp=(a);(a)=(b);(b)=itemp;
#define ITMAX 10000
#define CGOLD 0.3819660
#define ZEPS 1.0e-10
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);
#define GOLD 1.6180339887
#define GLIMIT 100.0
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);
#define MOV3(a,b,c, d,e,f) (a)=(d);(b)=(e);(c)=(f);
#define EPS 1.0e-10
#define NSTACK 50
#define threadsPerBlock 512
#define BSZX 32
#define BSZY 16
#define fac_threads threadsPerBlock*BSZX/(BSZX-2)*BSZY/(BSZY-2)
void                            /* function prototypes */
   GxStartGraphics (int, char **, int *, int *),
   GxEndGraphics (),
   GxOpenWindow (int, int), 
   GxAssignColors (const char **),
   GxSetFgColor (int), 
   GxMakePixmap (int, int),
   CbButton (Widget, XtPointer, XtPointer),
   CbScale (Widget, XtPointer, XtPointer),
   CbToggle (Widget, XtPointer, XtPointer),
   CbArrow (Widget, XtPointer, XtPointer),
   CbExpose (Widget, XtPointer, XmDrawingAreaCallbackStruct *), 
   CbResize (Widget, XtPointer, XmDrawingAreaCallbackStruct *), 
   DoEvents (Widget, XtPointer, XEvent *),
   Redraw (int, int),
   MoveBall (int, int),
   SetVel (),
   GxDelay (int),
   build(),	
   initial_md(),
   initial_cell(),
   calc_all_energy(),
   calc_all_energy_GPU(),
   calc_all_energy_no_shared_GPU(),
   CALC_U_AND_V(),
   CALC_FORCE(),
   time_step(),
   conjugate_gradient(),
   stretch(),
   initial_gpu_lists(),
   initial_gpu_lists_atoms(int,double,double),
   initial_gpu_lists_bonds(int,double,double,double,double,double,double);
int
   frprmn(double),
   frprmn_GPU(double);
int
   WorkProc ();                  /* work proc */

FILE *pFile,*pFile_vel,*pFile_vel2,*pFile_neigh,*pFile_diag;
XtAppContext app;               /* application context */
Display *dpy;                   /* display struct */
Window win;                     /* window struct */
GC gc;                          /* graphic context */
Pixel *pix;                     /* pixels for color */
Pixmap pixmap;                  /* pixmap struct */
Widget wScale[8],wText,wText2,wText3,wText4,wRadBox;               /* scale widgets */
int md_cg_mode,N,T,num,iteration,update,temperature,winWidth, winHeight , ballRad,redr,fac_sarig,num_realizations,
   td,running,ballColor,num_atoms,num_bonds,**neigh,eta10,**neigh_bond,**neigh_bond0,**bond_atom,num_vert1,num_vert2,num_hor,*n_safa,
   num_points,*type,k_teta_01,zoom_y,zoom_x,ndump,**neigh0,**bond_atom0,zoom_size,*type_odd_even,lists,shared,
   *X_PLOT,*Y_PLOT,*temp_int1,*temp_int2,cpu_gpu,Na,Nb,blocksPerGrid,blocksPerGrid2,blocksPerGrid_b,blocksPerGrid_mc,blocksPerGrid_mc_b,flag_for_dump;
double A_lat,*X,*Y,YMIN,YMAX,XMIN,XMAX,k_teta,*X0,*Y0,r_ball,x_par,fret,old_energy[5],energy[5],Y_CV,X_CV,DX_PLOT,DY_PLOT,time_md,*A_lat0,*temp_double;
double DT,*UX,*UY,*VX,*VY,*AX,*AY,ETA,*energy_store;
double *AX_GPU,*AY_GPU,*VX_GPU,*VY_GPU,*X_GPU,*Y_GPU,*A_lat0_GPU,*energy_GPU,*p,*p_GPU,*g,*g_GPU,*xi,*xi_GPU,*h,*h_GPU,gg_dgg[2],*gg_dgg_GPU,*pcom,*xicom,*xicom_GPU,*pcom_GPU,*xt,*xt_GPU;
int *n_safa_GPU,*X_PLOT_GPU,*Y_PLOT_GPU;
size_t size,size2,size_int,sizeb,size_intb,size4,size_num_realizations;
int max_nx,max_ny,block_x,block_y,*neigh_small_grid,*atom_list,*atom_id,*atom_id2,*atom_list_GPU,*atom_id_GPU,*atom_id2_GPU,*neigh_small_grid_GPU,
    *type_odd_even_lists,*type_odd_even_lists_GPU,blocksPerGridMat,NMat,*neigh_lists,*neigh_lists_GPU,*neigh_small_grid_big_GPU,*neigh_small_grid_big,
    blocksPerGridMat_b,NMat_b,*bond_list,*bond_id,*bond_id_GPU,max_nx_b,max_ny_b,*atom_list_bonds,*num_atoms_in_bonds,*bond_atom_small1,*bond_atom_small2,
    *atom_list_bonds_small,*atom_list_bonds_GPU,*num_atoms_in_bonds_GPU,*bond_atom_small1_GPU,*bond_atom_small2_GPU,*bond_list_GPU,*bond_atom1_GPU,*bond_atom2_GPU,
    *bond_atom0_small1,*bond_atom0_small2,*neigh_small_grid0,*neigh1_GPU,*neigh2_GPU,*neigh3_GPU,*neigh4_GPU,*neigh5_GPU,*neigh6_GPU,*type_odd_even_GPU;
curandState* devStates1,*devStates2,*devStates3;

double** Make2DDoubleArray(int, int);
int** Make2DIntArray(int, int);
cudaExtent extent;
const char *error;
struct timespec start, finish,start2, finish2,start3,finish3,start10,finish10,start11,finish11,start12,finish12,start13,finish13,start14,finish14;
double elapsed,elapsed2=0.0,elapsed3=0.0,elapsed10=0.0,elapsed11=0.0,elapsed12=0.0,elapsed13=0.0,elapsed14=0.0;
const double PI=M_PI;
const char
  *colors[] = {"pink", "dark blue", "green", "yellow", "red" , "black" ,"White" , NULL};
char tmp[255];

__device__ double atomicAdd_double(double* address, double val)
{ 
 unsigned long long int* address_as_ull = (unsigned long long int*)address; 
 unsigned long long int old = *address_as_ull, assumed; 
 do 
 { 
 assumed = old; 
 old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
 } while (assumed != old); 
 return __longlong_as_double(old); 
}

__global__ void CALC_U_AND_V_GPU(double* AX_G, double* AY_G, double* VX_G, double* VY_G, double* X_G, double* Y_G, int* n_safa_G, int N, double DT)
{
int mone = blockDim.x * blockIdx.x + threadIdx.x;
 if ((mone>0)&&(mone < N))
 {
  if(n_safa_G[mone]!=1)
  {
   VX_G[mone]=VX_G[mone]+(AX_G[mone])*DT;
   VY_G[mone]=VY_G[mone]+(AY_G[mone])*DT;
   X_G[mone]=X_G[mone]+VX_G[mone]*DT;
   Y_G[mone]=Y_G[mone]+VY_G[mone]*DT;
  }
 }	      
}

__global__ void ZEROING_A_GPU(double* AX_G, double* AY_G, int N)
{
int mone = blockDim.x * blockIdx.x + threadIdx.x;
 if ((mone>0)&&(mone <= N))
 {
   AX_G[mone]=0.0;
   AY_G[mone]=0.0;
 }
}

__global__ void INITIALIZING_P_XI_GPU(double* AX_G,double* AY_G,double* X_G,double* Y_G,int* n_safa_G,double* p_G,double* xi_G,int N)
{
int mone = blockDim.x * blockIdx.x + threadIdx.x;
 if ((mone>0)&&(mone <= N))
 {
   if(n_safa_G[mone]==0)
   {
     p_G[2*mone-1]=X_G[mone];
     p_G[2*mone]=Y_G[mone];
     xi_G[2*mone-1]=-AX_G[mone];
     xi_G[2*mone]=-AY_G[mone];
   }
 }
}

__global__ void INITIALIZING_G_H_XI_GPU(double* xi_G,double*  g_G,double*  h_G, int N)
{
int mone = blockDim.x * blockIdx.x + threadIdx.x;
 if ((mone>0)&&(mone <= N))
 {
    g_G[mone] = -xi_G[mone];
    xi_G[mone]=h_G[mone]=g_G[mone];
 }
}

__global__ void CALCULATING_XI_GPU(double* AX_G,double* AY_G,double* xi_G,int N)
{
int mone = blockDim.x * blockIdx.x + threadIdx.x;
 if ((mone>0)&&(mone <= N))
 {   
    xi_G[2*mone-1]=-AX_G[mone];
    xi_G[2*mone]=-AY_G[mone];
 }
}

__global__ void CALCULATING_GG_DGG_GPU(double* gg_dgg_G,double* g_G, double* xi_G,int* n_safa_G ,int N)
{
    __shared__ double temp[threadsPerBlock]; /*thread shared memory*/
    __shared__ double temp2[threadsPerBlock]; /*thread shared memory*/
    int mone=threadIdx.x + blockIdx.x * blockDim.x;
    int j=threadIdx.x ;
    temp[j]=0.0;
    temp2[j]=0.0;
    if ((mone>0)&&(mone < N))
    {
        if(n_safa_G[mone]==0)
	{
	  temp[j] = g_G[2*mone-1]*g_G[2*mone-1]+g_G[2*mone]*g_G[2*mone];
	  temp2[j] = (xi_G[2*mone-1]+g_G[2*mone-1])*xi_G[2*mone-1]+(xi_G[2*mone]+g_G[2*mone])*xi_G[2*mone];
	}
    }
    __syncthreads();
    
    if( 0 == threadIdx.x ) 
    {
      double sum = 0.0;
      double sum2 = 0.0;
      for( int i = 0; i < threadsPerBlock; i++ )
      {
        sum += temp[i];
	sum2 += temp2[i];
      }
      atomicAdd_double(&gg_dgg_G[0] ,sum);
      atomicAdd_double(&gg_dgg_G[1] ,sum2);
    }    
}

__global__ void CALCULATING_G_XI_GPU(double* xi_G,double* g_G,double* h_G,int* n_safa_G,int N,double gam)
{
int mone = blockDim.x * blockIdx.x + threadIdx.x;
 if ((mone>0)&&(mone <= N))
 {
   if(n_safa_G[mone]==0)
   {
     g_G[2*mone-1] = -xi_G[2*mone-1];
     xi_G[2*mone-1]=h_G[2*mone-1]=g_G[2*mone-1]+gam*h_G[2*mone-1];
     g_G[2*mone] = -xi_G[2*mone];
     xi_G[2*mone]=h_G[2*mone]=g_G[2*mone]+gam*h_G[2*mone];
   }
 }
}

__global__ void CALCULATING_PCOM_XICOM_GPU(double* xi_G,double* xicom_G,double* p_G,double* pcom_G,int N)
{
int mone = blockDim.x * blockIdx.x + threadIdx.x;
 if ((mone>0)&&(mone <= N))
 {
   pcom_G[mone]=p_G[mone];
   xicom_G[mone]=xi_G[mone];
 }
}

__global__ void CALCULATING_XI_P_GPU(double* xi_G,double* p_G,int N,double xmin)
{
int mone = blockDim.x * blockIdx.x + threadIdx.x;
 if ((mone>0)&&(mone <= N))
 {
    xi_G[mone] *= xmin;
    p_G[mone] += xi_G[mone];   
 }
}

__global__ void CALCULATING_X_Y_GPU(double* X_G,double* Y_G, double* p_G, int* n_safa_G, int N)
{
int mone = blockDim.x * blockIdx.x + threadIdx.x;
 if ((mone>0)&&(mone <= N))
 {
    if(n_safa_G[mone]==0)
    {
      X_G[mone]=p_G[2*mone-1];
      Y_G[mone]=p_G[2*mone];
    }
 }
}

__global__ void CALCULATING_XT_DF_GPU(double* pcom_G,double* xicom_G, double* xt_G, int* n_safa_G, int N, double x)
{
int mone = blockDim.x * blockIdx.x + threadIdx.x;
 if ((mone>0)&&(mone <= N))
 {
    if(n_safa_G[mone]==0)
    {
      xt_G[2*mone-1]=pcom_G[2*mone-1]+x*xicom_G[2*mone-1];
      xt_G[2*mone]=pcom_G[2*mone]+x*xicom_G[2*mone];
    }	 
 }
}

__global__ void CALC_FORCE_GPU(double* AX_G, double* AY_G, double* VX_G, double* VY_G, double* X_G, double* Y_G, int* bond_atom_small1_G,
                               int* bond_atom_small2_G,double ETA, double* A_lat0_G, int* bond_id_G,
			       int* num_atoms_in_bonds_G, int* atom_list_bonds_G, int* bond_list_G, int md_cg_mode)
      {
      int a,b,m;
      double R12,X1,Y1,X2,Y2,F_RAD,SCALAR,VX1,VY1,VX2,VY2,K12,A_lat0_G1;
      
      int i = threadIdx.x;    
      __shared__ double x_sh[threadsPerBlock];
      __shared__ double y_sh[threadsPerBlock];
      __shared__ double vx_sh[threadsPerBlock];
      __shared__ double vy_sh[threadsPerBlock];
      __shared__ double ax_sh[threadsPerBlock];
      __shared__ double ay_sh[threadsPerBlock];
      
      if (i < num_atoms_in_bonds_G[blockIdx.x])
      {
        m=atom_list_bonds_G[blockIdx.x * threadsPerBlock + i];
	x_sh[i]=X_G[m];
        y_sh[i]=Y_G[m];
        vx_sh[i]=VX_G[m];
        vy_sh[i]=VY_G[m];
        ax_sh[i]=0.0;
        ay_sh[i]=0.0;
      }
      
      __syncthreads();
     
      if (i < bond_id_G[blockIdx.x])
      {
	a=bond_atom_small1_G[blockIdx.x*threadsPerBlock + i];
	b=bond_atom_small2_G[blockIdx.x*threadsPerBlock + i];
	
	if((a!=-1)&&(b!=-1))
        {
	  A_lat0_G1=A_lat0_G[bond_list_G[blockIdx.x*threadsPerBlock + i]];
	  /* RADIAL-FORCE */
	  X1=x_sh[a];
          Y1=y_sh[a];
          X2=x_sh[b];
          Y2=y_sh[b];
          R12=sqrt((X1-X2)*(X1-X2)+(Y1-Y2)*(Y1-Y2));
	  K12=1.0;
          F_RAD=(R12-A_lat0_G1)*K12;

	  atomicAdd_double(&ax_sh[a],(X2-X1)/R12*F_RAD);
          atomicAdd_double(&ay_sh[a],(Y2-Y1)/R12*F_RAD);
          atomicAdd_double(&ax_sh[b],(X1-X2)/R12*F_RAD);
          atomicAdd_double(&ay_sh[b],(Y1-Y2)/R12*F_RAD);

          /* KELVIN VISCOSITY */
          if(md_cg_mode==1)
	  {
	    VX1=vx_sh[a];
            VY1=vy_sh[a];
            VX2=vx_sh[b];
            VY2=vy_sh[b];
            SCALAR=ETA*K12*((VX2-VX1)*(X2-X1)+(VY2-VY1)*(Y2-Y1))/R12;
          	  
	    atomicAdd_double(&ax_sh[a],(X2-X1)/R12*SCALAR);
            atomicAdd_double(&ay_sh[a],(Y2-Y1)/R12*SCALAR);
            atomicAdd_double(&ax_sh[b],(X1-X2)/R12*SCALAR);
            atomicAdd_double(&ay_sh[b],(Y1-Y2)/R12*SCALAR);
	  }
	 }
       }

      __syncthreads();       
 
      if (i < num_atoms_in_bonds_G[blockIdx.x])
      {
	atomicAdd_double(&AX_G[m],ax_sh[i]);
        atomicAdd_double(&AY_G[m],ay_sh[i]);
      } 
           
      }

__global__ void CALC_3_BODY_FORCE_GPU(double* AX_G, double* AY_G, double* X_G, double* Y_G, double k_teta,
			       int* atom_list_G,int* atom_id_G,int *atom_id2_G,int *neigh_small_grid_G,int *type_odd_even_lists_G)
      {
      int a,b,c,vec1[4],vec2[4],m,mm;
      double X1,Y1,X2,Y2,X00,Y00,R1,R2,cos_teta1,teta1,K1,K2;

        /* 3-BODY FORCE-LAW */
      
      int i = threadIdx.x;

      __shared__ double x_sh[fac_threads];
      __shared__ double y_sh[fac_threads];
      __shared__ double ax_sh[fac_threads];
      __shared__ double ay_sh[fac_threads];
      __shared__ int neigh_sh[fac_threads][7];

      
      if(i < atom_id_G[blockIdx.x])
      {
        m=atom_list_G[blockIdx.x * fac_threads + i];
        x_sh[i]=X_G[m];
        y_sh[i]=Y_G[m];
        ax_sh[i]=0.0;
        ay_sh[i]=0.0;
        for(c=1; c<=6; c++)
        {
          neigh_sh[i][c]=neigh_small_grid_G[blockIdx.x * fac_threads * 7 + i *7 + c];
        }
      }
      
      if(i < (atom_id2_G[blockIdx.x]-atom_id_G[blockIdx.x]))
      {
        mm=atom_list_G[blockIdx.x * fac_threads + i + atom_id_G[blockIdx.x]];
        x_sh[atom_id_G[blockIdx.x]+i]=X_G[mm];
        y_sh[atom_id_G[blockIdx.x]+i]=Y_G[mm];
        ax_sh[atom_id_G[blockIdx.x]+i]=0.0;
        ay_sh[atom_id_G[blockIdx.x]+i]=0.0;
	for(c=1; c<=6; c++)
        {
          neigh_sh[i+atom_id_G[blockIdx.x]][c]=neigh_small_grid_G[blockIdx.x * fac_threads * 7 + (i+atom_id_G[blockIdx.x]) *7 + c];
        }     
      }

      __syncthreads();
     
      if (i < atom_id_G[blockIdx.x])
      {

	    if(type_odd_even_lists_G[blockIdx.x * fac_threads + i]==1)
	    {
	      vec1[1]=2; vec1[2]=1; vec1[3]=3;
              vec2[1]=6; vec2[2]=5; vec2[3]=4;
            } 
	    if(type_odd_even_lists_G[blockIdx.x * fac_threads + i]==2)
	    {
	      vec1[1]=1; vec1[2]=2; vec1[3]=3;
	      vec2[1]=5; vec2[2]=6; vec2[3]=4;
            } 
            X00=x_sh[i]; Y00=y_sh[i];
	    for(c=1; c<=3; c++)
            {
              a=neigh_sh[i][vec1[c]];	     
 	      if(a!=-1)
 	      {
 	  	X1=x_sh[a]; Y1=y_sh[a];
 	      }
 	      else
	      {
          	X1=-999999.0;  
	      }
              
	      b=neigh_sh[i][vec2[c]];
	      if(b!=-1)
	      {
	  	X2=x_sh[b]; Y2=y_sh[b];
	      }
	      else
	      {
          	X2=-999999.0;
	      }
              R1=sqrt((X1-X00)*(X1-X00)+(Y1-Y00)*(Y1-Y00));
              R2=sqrt((X2-X00)*(X2-X00)+(Y2-Y00)*(Y2-Y00));
	      K1=1.0;
              K2=1.0;
	      if((X1!=-999999.0)&&(X2!=-999999.0))
              {
                 cos_teta1=((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R1/R2;
		 teta1=fabs(PI-acos((cos_teta1+1.0e-15)));
		 
		 atomicAdd_double(&ax_sh[i],-k_teta*K1*K2*teta1/(sin(teta1)+1e-30)*((-(X1-X00)-(X2-X00))/R1/R2+
	  				    (X1-X00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R1/R1/R1/R2+
	        			    (X2-X00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R2/R2/R2/R1));
          	 atomicAdd_double(&ay_sh[i],-k_teta*K1*K2*teta1/(sin(teta1)+1e-30)*((-(Y1-Y00)-(Y2-Y00))/R1/R2+
	  				    (Y1-Y00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R1/R1/R1/R2+
	        			    (Y2-Y00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R2/R2/R2/R1));

		 atomicAdd_double(&ax_sh[a],-k_teta*K1*K2*teta1/(sin(teta1)+1e-30)*((X2-X00)/R1/R2-
	  				    (X1-X00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R1/R1/R1/R2));
          	 atomicAdd_double(&ay_sh[a],-k_teta*K1*K2*teta1/(sin(teta1)+1e-30)*((Y2-Y00)/R1/R2-
	  				    (Y1-Y00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R1/R1/R1/R2));
                 
          	 atomicAdd_double(&ax_sh[b],-k_teta*K1*K2*teta1/(sin(teta1)+1e-30)*((X1-X00)/R1/R2-
	  				    (X2-X00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R2/R2/R2/R1));
          	 atomicAdd_double(&ay_sh[b],-k_teta*K1*K2*teta1/(sin(teta1)+1e-30)*((Y1-Y00)/R1/R2-
	  				    (Y2-Y00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R2/R2/R2/R1));
              }  
	    }
     }
     
      __syncthreads();
      
      if (i < atom_id_G[blockIdx.x])
      {  
	atomicAdd_double(&AX_G[m],ax_sh[i]);
        atomicAdd_double(&AY_G[m],ay_sh[i]);
      }
      if (i < atom_id2_G[blockIdx.x]-atom_id_G[blockIdx.x])
      {
	atomicAdd_double(&AX_G[mm],ax_sh[atom_id_G[blockIdx.x]+i]);
        atomicAdd_double(&AY_G[mm],ay_sh[atom_id_G[blockIdx.x]+i]);
      }
}

__global__ void CALC_FORCE_no_shared_GPU(double* AX_G, double* AY_G, double* VX_G, double* VY_G, double* X_G, double* Y_G, int N, int* bond_atom1_G,
                               int* bond_atom2_G,double ETA, double* A_lat0_G,int md_cg_mode)
      {
      int a,b;
      double R12,X1,Y1,X2,Y2,F_RAD,SCALAR,VX1,VY1,VX2,VY2,K12,A_lat0_G1;
      
      int mone = blockDim.x * blockIdx.x + threadIdx.x;
      
      if ((mone>0)&&(mone < N))
       {
	a=bond_atom1_G[mone];
	b=bond_atom2_G[mone];
	if((a>0)&&(b>0))
        {
	  A_lat0_G1=A_lat0_G[mone];
	  /* RADIAL-FORCE */
	  X1=X_G[a];
          Y1=Y_G[a];
          X2=X_G[b];
          Y2=Y_G[b];
          R12=sqrt((X1-X2)*(X1-X2)+(Y1-Y2)*(Y1-Y2));
	  K12=1.0;
          F_RAD=(R12-A_lat0_G1)*K12;
	  atomicAdd_double(&AX_G[a],(X2-X1)/R12*F_RAD);
          atomicAdd_double(&AY_G[a],(Y2-Y1)/R12*F_RAD);
          atomicAdd_double(&AX_G[b],(X1-X2)/R12*F_RAD);
          atomicAdd_double(&AY_G[b],(Y1-Y2)/R12*F_RAD);

          /* KELVIN VISCOSITY */
          if(md_cg_mode==1)
	  {
	    VX1=VX_G[a];
            VY1=VY_G[a];
            VX2=VX_G[b];
            VY2=VY_G[b];
            SCALAR=ETA*K12*((VX2-VX1)*(X2-X1)+(VY2-VY1)*(Y2-Y1))/R12;
            atomicAdd_double(&AX_G[a],(X2-X1)/R12*SCALAR);
            atomicAdd_double(&AY_G[a],(Y2-Y1)/R12*SCALAR);
            atomicAdd_double(&AX_G[b],(X1-X2)/R12*SCALAR);
            atomicAdd_double(&AY_G[b],(Y1-Y2)/R12*SCALAR);
	  }
	 }
       }
      }

__global__ void CALC_3_BODY_FORCE_no_shared_GPU(double* AX_G, double* AY_G, double* X_G, double* Y_G, int N, double k_teta,int* type_odd_even_G,
			       int* neigh1_G,int* neigh2_G,int* neigh3_G,int* neigh4_G,int* neigh5_G,int* neigh6_G)
      {
      int a,b,c,vec1[4],vec2[4];
      double X1,Y1,X2,Y2,X00,Y00,R1,R2,cos_teta1,teta1;

        /* 3-BODY FORCE-LAW */
	      
      int mone = blockDim.x * blockIdx.x + threadIdx.x; 

      if ((mone>0)&&(mone < N))
      {           
	    if(type_odd_even_G[mone]==1)
	    {
	      vec1[1]=2; vec1[2]=1; vec1[3]=3;
              vec2[1]=6; vec2[2]=5; vec2[3]=4;
            } 
	    if(type_odd_even_G[mone]==2)
	    {
              vec1[1]=1; vec1[2]=2; vec1[3]=3;
              vec2[1]=5; vec2[2]=6; vec2[3]=4;
            } 
            X00=X_G[mone]; Y00=Y_G[mone];
	    for(c=1; c<=3; c++)
            {
              if(vec1[c]==1)a=neigh1_G[mone];
              if(vec1[c]==2)a=neigh2_G[mone];
              if(vec1[c]==3)a=neigh3_G[mone];
              if(vec1[c]==4)a=neigh4_G[mone];
              if(vec1[c]==5)a=neigh5_G[mone];
              if(vec1[c]==6)a=neigh6_G[mone];
	     
 	      if(a!=-1)
 	      {
 	  	X1=X_G[a]; Y1=Y_G[a];
 	      }
 	      else
	      {
          	X1=-999999.0;  
	      }
              if(vec2[c]==1)b=neigh1_G[mone];
              if(vec2[c]==2)b=neigh2_G[mone];
              if(vec2[c]==3)b=neigh3_G[mone];
              if(vec2[c]==4)b=neigh4_G[mone];
              if(vec2[c]==5)b=neigh5_G[mone];
              if(vec2[c]==6)b=neigh6_G[mone];
	      if(b!=-1)
	      {
	  	X2=X_G[b]; Y2=Y_G[b];
	      }
	      else
	      {
          	X2=-999999.0;  
	      }
              R1=sqrt((X1-X00)*(X1-X00)+(Y1-Y00)*(Y1-Y00));
              R2=sqrt((X2-X00)*(X2-X00)+(Y2-Y00)*(Y2-Y00));
              if((X1!=-999999.0)&&(X2!=-999999.0))
              {
                 cos_teta1=((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R1/R2;
		 teta1=fabs(PI-acos((cos_teta1+1.0e-15)));
		 atomicAdd_double(&AX_G[mone],-k_teta*teta1/(sin(teta1)+1e-30)*((-(X1-X00)-(X2-X00))/R1/R2+
	  				    (X1-X00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R1/R1/R1/R2+
	        			    (X2-X00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R2/R2/R2/R1));
          	 atomicAdd_double(&AY_G[mone],-k_teta*teta1/(sin(teta1)+1e-30)*((-(Y1-Y00)-(Y2-Y00))/R1/R2+
	  				    (Y1-Y00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R1/R1/R1/R2+
	        			    (Y2-Y00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R2/R2/R2/R1));
	
          	 atomicAdd_double(&AX_G[a],-k_teta*teta1/(sin(teta1)+1e-30)*((X2-X00)/R1/R2-
	  				    (X1-X00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R1/R1/R1/R2));
          	 atomicAdd_double(&AY_G[a],-k_teta*teta1/(sin(teta1)+1e-30)*((Y2-Y00)/R1/R2-
	  				    (Y1-Y00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R1/R1/R1/R2));

          	 atomicAdd_double(&AX_G[b],-k_teta*teta1/(sin(teta1)+1e-30)*((X1-X00)/R1/R2-
	  				    (X2-X00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R2/R2/R2/R1));
          	 atomicAdd_double(&AY_G[b],-k_teta*teta1/(sin(teta1)+1e-30)*((Y1-Y00)/R1/R2-
	  				    (Y2-Y00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R2/R2/R2/R1));
              }  
	    }    
       }   
}

__global__ void CALC_UXY_PLOT(int* X_PLOT_G,int* Y_PLOT_G,double* X_G,double* Y_G,double DX_PLOT,double DY_PLOT,double X_CV,double Y_CV,double YMIN,
                              double XMIN,double A_lat,int winHeight,int winWidth, int N)
{
      int mone = blockDim.x * blockIdx.x + threadIdx.x;
           	      
      if ((mone>0)&&(mone < N))
       {
         Y_PLOT_G[mone]=(int)(((Y_G[mone]-YMIN+A_lat/2)/(DY_PLOT)-Y_CV)*(double)(winHeight)*0.95);
         X_PLOT_G[mone]=(int)(((X_G[mone]-XMIN+A_lat/2)/(DX_PLOT)-X_CV)*(double)(winWidth)*0.95);    
       }       
}

__global__ void CALC_KINETIC_ENERGY(double* VX_G, double* VY_G, double* energy_G, int N)
{
    __shared__ double temp[threadsPerBlock]; /*thread shared memory*/
    int mone=threadIdx.x + blockIdx.x * blockDim.x;
    int j=threadIdx.x ;
    temp[j]=0.0;
    if ((mone>0)&&(mone < N))
    {
        temp[j] = 0.5*(VX_G[mone]*VX_G[mone]+VY_G[mone]*VY_G[mone]);
    }
    __syncthreads();
    
    if( 0 == threadIdx.x ) 
    {
      double sum = 0.0;
      for( int i = 0; i < threadsPerBlock; i++ )
      sum += temp[i];
      atomicAdd_double(&energy_G[4] ,sum);
    }
}

__global__ void  calc_raidial_energy_GPU(double* X_G, double* Y_G, int* bond_atom_small1_G, double* energy_G,
                                         int* bond_atom_small2_G, double* A_lat0_G, int* bond_id_G,int* num_atoms_in_bonds_G, int* atom_list_bonds_G,
					 int* bond_list_G)
{
    int a,b;
    double r;    
    int mone = threadIdx.x;    
    __shared__ double x_sh[threadsPerBlock];
    __shared__ double y_sh[threadsPerBlock];
    __shared__ double temp[threadsPerBlock];
      
    if (mone < num_atoms_in_bonds_G[blockIdx.x])
    {
      x_sh[mone]=X_G[atom_list_bonds_G[blockIdx.x * threadsPerBlock + mone]];
      y_sh[mone]=Y_G[atom_list_bonds_G[blockIdx.x * threadsPerBlock + mone]];
    }
      
    __syncthreads();
     
    temp[mone]=0.0;
    if (mone < bond_id_G[blockIdx.x])
    {
      a=bond_atom_small1_G[blockIdx.x*threadsPerBlock + mone];
      b=bond_atom_small2_G[blockIdx.x*threadsPerBlock + mone];
      if((a!=-1)&&(b!=-1))
        {
          r=sqrt((x_sh[a]-x_sh[b])*(x_sh[a]-x_sh[b])+(y_sh[a]-y_sh[b])*(y_sh[a]-y_sh[b]));
          temp[mone] +=0.5*(r-A_lat0_G[bond_list_G[blockIdx.x*threadsPerBlock + mone]])*(r-A_lat0_G[bond_list_G[blockIdx.x*threadsPerBlock + mone]]);
       }   
    }
    __syncthreads();
    
    if( 0 == threadIdx.x ) 
    {
      double sum = 0.0;
      for( int i = 0; i < threadsPerBlock; i++ )
      sum += temp[i];
      atomicAdd_double(&energy_G[1] ,sum);
    }
}

__global__ void  calc_3body_energy_GPU(double* X_G, double* Y_G, double k_teta,double* energy_G,
			       int* atom_list_G,int* atom_id_G,int* atom_id2_G,int *neigh_small_grid_G,int *type_odd_even_lists_G)
{
    int a,b,c,vec1[7],vec2[7],m,mm;
    int mone = threadIdx.x;
    double X1,Y1,X2,Y2,R1,R2,cos_teta1,X00,Y00,teta1;

    __shared__ double x_sh[fac_threads];
    __shared__ double y_sh[fac_threads];
    __shared__ int neigh_sh[fac_threads][7];
    __shared__ double temp[fac_threads];
        
    if(mone < atom_id_G[blockIdx.x])
    {
      m=atom_list_G[blockIdx.x * fac_threads + mone];
      x_sh[mone]=X_G[m];
      y_sh[mone]=Y_G[m];
      for(c=1; c<=6; c++)
      {
    	neigh_sh[mone][c]=neigh_small_grid_G[blockIdx.x * fac_threads * 7 + mone *7 + c];
      }
    }
    
    if(mone < (atom_id2_G[blockIdx.x]-atom_id_G[blockIdx.x]))
    {
      mm=atom_list_G[blockIdx.x * fac_threads + mone + atom_id_G[blockIdx.x]];
      x_sh[atom_id_G[blockIdx.x]+mone]=X_G[mm];
      y_sh[atom_id_G[blockIdx.x]+mone]=Y_G[mm];
      for(c=1; c<=6; c++)
      {
    	neigh_sh[mone+atom_id_G[blockIdx.x]][c]=neigh_small_grid_G[blockIdx.x * fac_threads * 7 + (mone+atom_id_G[blockIdx.x]) *7 + c];
      }     
    }

    __syncthreads();     
       
    temp[mone] = 0.0;

    if (mone < atom_id_G[blockIdx.x])
    {

	  if(type_odd_even_lists_G[blockIdx.x * fac_threads + mone]==1)
	  {
	    vec1[1]=2; vec1[2]=1; vec1[3]=3;
            vec2[1]=6; vec2[2]=5; vec2[3]=4;
          } 
	  if(type_odd_even_lists_G[blockIdx.x * fac_threads + mone]==2)
	  {
            vec1[1]=1; vec1[2]=2; vec1[3]=3;
            vec2[1]=5; vec2[2]=6; vec2[3]=4;
          } 
          X00=x_sh[mone]; Y00=y_sh[mone];
	  for(c=1; c<=3; c++)
          {
	    a=neigh_sh[mone][vec1[c]];
 	    if(a!=-1)
 	    {
 	      X1=x_sh[a]-X00; Y1=y_sh[a]-Y00;
 	    }
 	    else
	    {
              X1=-999999.0;  
	    }
            
	    b=neigh_sh[mone][vec2[c]];
	    if(b!=-1)
	    {
	      X2=x_sh[b]-X00; Y2=y_sh[b]-Y00;
	    }
	    else
	    {
              X2=-999999.0;
	    }
 	
	    R1=sqrt(X1*X1+Y1*Y1);
            R2=sqrt(X2*X2+Y2*Y2);
            if((X1!=-999999.0)&&(X2!=-999999.0))
            {
              cos_teta1=(X1*X2+Y1*Y2)/R1/R2;
	      teta1=fabs(PI-acos((cos_teta1+1.0e-15)));
            }
            else
            {
              teta1=0.0;
            }
            temp[mone] += 0.5*k_teta*teta1*teta1;
	    }
    }
    
    __syncthreads();
    
    if( 0 == threadIdx.x ) 
    {
      double sum = 0.0;
      for( int i = 0; i < threadsPerBlock; i++ )
      sum += temp[i];
      atomicAdd_double(&energy_G[2] ,sum);
    }

}

__global__ void  calc_raidial_energy_no_shared_GPU(double* X_G, double* Y_G, int N, int* bond_atom1_G, double* energy_G,
                               int* bond_atom2_G, double* A_lat0_G)
{
    __shared__ double cache[threadsPerBlock]; /*thread shared memory*/
    int mone=threadIdx.x + blockIdx.x * blockDim.x;
    int i=0,cacheIndex=0;
    double temp = 0;
    int a,b;
    double r;
    cacheIndex = threadIdx.x;
    while ((mone>0)&&(mone < N)) {
        a=bond_atom1_G[mone];
        b=bond_atom2_G[mone];	
        if((a>0)&&(b>0))
        {
          r=sqrt((X_G[a]-X_G[b])*(X_G[a]-X_G[b])+(Y_G[a]-Y_G[b])*(Y_G[a]-Y_G[b]));
          temp +=0.5*(r-A_lat0_G[mone])*(r-A_lat0_G[mone]);
       }
        mone += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = temp;
    __syncthreads();
    for (i=blockDim.x/2; i>0; i>>=1) {
        if (threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
    }
    __syncthreads();
    if (cacheIndex==0) {
        atomicAdd_double(&energy_G[1] ,cache[0]);
    } 
}  

__global__ void  calc_3body_energy_no_shared_GPU(double* X_G, double* Y_G, int N, double k_teta,int* type_odd_even_G,double* energy_G,
			       int* neigh1_G,int* neigh2_G,int* neigh3_G,int* neigh4_G,int* neigh5_G,int* neigh6_G)
{
    __shared__ double cache[threadsPerBlock]; /*thread shared memory*/
    int mone=threadIdx.x + blockIdx.x * blockDim.x;
    int i=0,cacheIndex=0;
    double temp = 0;
    int a,b,c,vec1[4],vec2[4];
    double X1,Y1,X2,Y2,R1,R2,cos_teta1,X00,Y00,teta1;
    cacheIndex = threadIdx.x;
    while ((mone>0)&&(mone < N)) {
	    X00=X_G[mone]; Y00=Y_G[mone];
	    if(type_odd_even_G[mone]==1)
	    {
	      vec1[1]=2; vec1[2]=1; vec1[3]=3;
              vec2[1]=6; vec2[2]=5; vec2[3]=4;
            } 
	    if(type_odd_even_G[mone]==2)
	    {
              vec1[1]=1; vec1[2]=2; vec1[3]=3;
              vec2[1]=5; vec2[2]=6; vec2[3]=4;
            } 
	    for(c=1; c<=3; c++)
            {
              if(vec1[c]==1)a=neigh1_G[mone];
              if(vec1[c]==2)a=neigh2_G[mone];
              if(vec1[c]==3)a=neigh3_G[mone];
              if(vec1[c]==4)a=neigh4_G[mone];
              if(vec1[c]==5)a=neigh5_G[mone];
              if(vec1[c]==6)a=neigh6_G[mone];
	     
 	      if(a!=-1)
 	      {
 	        X1=X_G[a]-X00; Y1=Y_G[a]-Y00;
 	      }
 	      else
	      {
          	X1=-999999.0;  
	      }
              if(vec2[c]==1)b=neigh1_G[mone];
              if(vec2[c]==2)b=neigh2_G[mone];
              if(vec2[c]==3)b=neigh3_G[mone];
              if(vec2[c]==4)b=neigh4_G[mone];
              if(vec2[c]==5)b=neigh5_G[mone];
              if(vec2[c]==6)b=neigh6_G[mone];
 	      if(b!=-1)
 	      {
 	        X2=X_G[b]-X00; Y2=Y_G[b]-Y00;
 	      }
	      else
	      {
          	X2=-999999.0;  
	      }
	      R1=sqrt(X1*X1+Y1*Y1);
              R2=sqrt(X2*X2+Y2*Y2);
              if((X1!=-999999.0)&&(X2!=-999999.0))
              {
                cos_teta1=(X1*X2+Y1*Y2)/R1/R2;
		teta1=fabs(PI-acos((cos_teta1+1.0e-15)));
              }
              else
              {
                teta1=0.0;
              }
              temp += 0.5*k_teta*teta1*teta1; 
	      }
	      mone += blockDim.x * gridDim.x;
	      }
    cache[cacheIndex] = temp;
    __syncthreads();
    for (i=blockDim.x/2; i>0; i>>=1) {
        if (threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
    }
    __syncthreads();
    if (cacheIndex==0) {
        atomicAdd_double(&energy_G[2] ,cache[0]);
    }
}  

__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int mone = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init ( seed, mone, 0, &state[mone] );
}

int main (int argc, char **argv)
{
  int dpyWidth, dpyHeight;
  
  /* start */
  GxStartGraphics (argc, argv, &dpyWidth, &dpyHeight);
  GxAssignColors (colors);
  /* initial window size */ 
  winWidth = (int)(dpyWidth * 0.9);  winHeight = (int)(dpyHeight * 0.9);
  /* initial ball position, ball size */
  md_cg_mode=0;
  num_realizations=3;
  srand48((unsigned)time(NULL));
  td=10;   /* initial td */
  DT=1.0/((double)(td))*1.0;
  k_teta_01=2;
  k_teta=((double)(k_teta_01))/100.0;
  redr=0;  /* no. of iterations done */
  ballRad=4;
  ballColor=1;
  running=0;
  update=10; /* redraw each "update" computions */
  A_lat=4.0;
  fac_sarig=1;
  N=99*fac_sarig;
  T=200*fac_sarig;
  zoom_x=26; 
  zoom_y=33;
  zoom_size=100;
  eta10=25;
  ETA=((double)(eta10))/100.0;
  ndump=0;
  cpu_gpu=1;
  shared=0;
  r_ball=5.1*A_lat;
  x_par=r_ball;
   
  GxOpenWindow (winWidth, winHeight);
  
  /*if (cpu_gpu == 1)
  {
    int dev;
    cudaDeviceProp prop;
    cudaChooseDevice(&dev,&prop);
    cudaSetDevice(dev);
  } */

  build();  /* create reshet */
  
  /* specify work procedure */
  XtAppAddWorkProc (app, (XtWorkProc) WorkProc, 0);
  /* main processing/event loop */
  XtAppMainLoop (app);

  return 0;
}

/* start graphics and return display size */
void GxStartGraphics (int argc, char **argv, int *w, int *h)
{
  /* initialize X toolkit */
  XtToolkitInitialize ();
  /* create application context */
  app = XtCreateApplicationContext ();
  /* open display */
  dpy = XtOpenDisplay (app, NULL, NULL, "crn", NULL, 0, &argc, argv);
  *w = DisplayWidth (dpy, DefaultScreen (dpy));
  *h = DisplayHeight (dpy, DefaultScreen (dpy));
}

/* end graphics and exit */
void GxEndGraphics ()
{
  XtDestroyApplicationContext (app);
  exit (0);
}

/* open window for graphics, setup controls */
void GxOpenWindow (int width, int height)
{
  Widget wTop, wForm, wRowCol, wButton,
     wFrame, wDraw;
  int n;
  const char *bLabel[] = {"Quit","Start","Stretch","Dump to File"},
       *tLabel[] = {"Optimization", "Molecular-Dynamics"};

  /* shell widget */
  wTop = XtVaAppCreateShell (NULL, NULL, applicationShellWidgetClass,
     dpy, ((void*)0));
  /* form widget to organize child widgets */
  wForm = XtVaCreateManagedWidget ("fm", xmFormWidgetClass, wTop,
     ((void*)0));
  /* vertical row-column widget for controls */
  wRowCol = XtVaCreateManagedWidget ("rc", xmRowColumnWidgetClass, wForm,
     XmNpacking,          XmPACK_TIGHT,
     XmNspacing,          2,
     XmNorientation,      XmVERTICAL,  
     XmNleftAttachment,   XmATTACH_FORM,
     XmNtopAttachment,    XmATTACH_FORM,
     XmNbottomAttachment, XmATTACH_FORM,
     ((void*)0));
  /* pushbutton widgets */
  for (n = 0; n < 1; n ++) {
    wButton = XtVaCreateManagedWidget ("pb", xmPushButtonWidgetClass,
       wRowCol,
       XtVaTypedArg,      XmNlabelString, TSTRING (bLabel[n]),
       ((void*)0));
    XtAddCallback (wButton, XmNactivateCallback, CbButton, (XtPointer)((long)(n)));
  }

  /* label widget */
  XtVaCreateManagedWidget ("la", xmLabelWidgetClass, wRowCol,
     XtVaTypedArg,        XmNlabelString, TSTRING ("Run-Mode"), 
     ((void*)0));
  /* frame widget surrounding next row-column widget */
  wFrame = XtVaCreateManagedWidget ("fr", xmFrameWidgetClass, wRowCol,
     XmNshadowType,       XmSHADOW_ETCHED_IN,
     ((void*)0));
  /* radio-box and togglebutton widgets */
  wRadBox = XtVaCreateManagedWidget ("rc", xmRowColumnWidgetClass, wFrame,
     XmNpacking,          XmPACK_COLUMN,
     XmNorientation,      XmVERTICAL,
     XmNisHomogeneous,    True,
     XmNentryClass,       xmToggleButtonWidgetClass,
     XmNradioBehavior,    True,
     XmNradioAlwaysOne,   True,
     ((void*)0));
  for (n = 0; n <= 1; n ++) {
    wButton = XtVaCreateManagedWidget ("tg", xmToggleButtonWidgetClass,
       wRadBox,
       XtVaTypedArg,      
       XmNlabelString,    TSTRING (tLabel[n]),
       ((void*)0));
    XtAddCallback (wButton, XmNvalueChangedCallback, CbToggle,
       (XtPointer)((long)(n)));
    /* set initial selection */
    if (n == md_cg_mode) XmToggleButtonSetState (wButton, True, False);
  }

XtVaCreateManagedWidget ("la", xmLabelWidgetClass, wRowCol,
     XtVaTypedArg,        XmNlabelString, TSTRING ("Molecular-Dynamics Data:"), 
     ((void*)0));

  /* frame widget surrounding next row-column widget */
  wFrame = XtVaCreateManagedWidget ("fr3", xmFrameWidgetClass, wRowCol,
     XmNshadowType,       XmSHADOW_ETCHED_IN,
     ((void*)0));
     
  wText2 = XtVaCreateManagedWidget ("ta3", xmTextWidgetClass, wFrame,
    XmNcolumns,                23,
    XmNrows,                   5,
    XmNeditable,               False,
    XmNeditMode,               XmMULTI_LINE_EDIT,
    XmNcursorPositionVisible,  False,
    XmNhighlightThickness,     0,
    XmNshadowType,             XmSHADOW_OUT,
    XmNleftAttachment,         XmATTACH_FORM,
    XmNrightAttachment,        XmATTACH_WIDGET,
    XmNrightWidget,            wFrame,
    XmNrightOffset,            5,
    XmNtopAttachment,          XmATTACH_WIDGET,
    XmNtopWidget,              wScale[0],
    XmNtopOffset,              40,
    ((void*)0));

   for (n = 1; n < 3; n ++) {
    wButton = XtVaCreateManagedWidget ("pb", xmPushButtonWidgetClass,
       wRowCol,
       XtVaTypedArg,      XmNlabelString, TSTRING (bLabel[n]),
       ((void*)0));
    XtAddCallback (wButton, XmNactivateCallback, CbButton, (XtPointer)((long)(n)));
  }
  
  /* scale widgets */
  wScale[0] = XtVaCreateManagedWidget ("sc", xmScaleWidgetClass, wRowCol, 
     XmNorientation,      XmHORIZONTAL,
     XmNwidth,            150,
     XmNshowValue,        True,  
     XmNminimum,          0,
     XmNmaximum,          100,
     XmNvalue,            k_teta_01,
     XtVaTypedArg,        XmNtitleString, TSTRING ("100*k_teta"),
     ((void*)0));
  XtAddCallback (wScale[0], XmNvalueChangedCallback, CbScale,
     (XtPointer) 0);
     
     wScale[1] = XtVaCreateManagedWidget ("sc2", xmScaleWidgetClass, wRowCol, 
     XmNorientation,      XmHORIZONTAL,
     XmNwidth,            150,
     XmNshowValue,        True,  
     XmNminimum,          1,
     XmNmaximum,          1000,
     XmNvalue,            update,
     XtVaTypedArg,        XmNtitleString, TSTRING ("Display update rate"),
     ((void*)0));
  XtAddCallback (wScale[1], XmNvalueChangedCallback, CbScale,
     (XtPointer) 1);

     wScale[2] = XtVaCreateManagedWidget ("sc3", xmScaleWidgetClass, wRowCol, 
     XmNorientation,      XmHORIZONTAL,
     XmNwidth,            150,
     XmNshowValue,        True,  
     XmNminimum,          -10,
     XmNmaximum,          100,
     XmNvalue,            zoom_x,
     XtVaTypedArg,        XmNtitleString, TSTRING ("Shift Left-Right"),
     ((void*)0));
  XtAddCallback (wScale[2], XmNvalueChangedCallback, CbScale,
     (XtPointer) 2);

     wScale[3] = XtVaCreateManagedWidget ("sc4", xmScaleWidgetClass, wRowCol, 
     XmNorientation,      XmHORIZONTAL,
     XmNwidth,            150,
     XmNshowValue,        True,  
     XmNminimum,          -225,
     XmNmaximum,          100,
     XmNvalue,            zoom_y,
     XtVaTypedArg,        XmNtitleString, TSTRING ("Shift Up-Down"),
     ((void*)0));
  XtAddCallback (wScale[3], XmNvalueChangedCallback, CbScale,
     (XtPointer) 3);

/* frame widget surrounding next row-column widget */
  wFrame = XtVaCreateManagedWidget ("fr", xmFrameWidgetClass, wForm,
     XmNleftAttachment,   XmATTACH_WIDGET,      /* attach to widget */
     XmNleftWidget,       wRowCol,
     XmNrightAttachment,  XmATTACH_FORM,        /* attach to form */
     XmNtopAttachment,    XmATTACH_FORM,
     XmNbottomAttachment, XmATTACH_FORM,
     XmNshadowType,       XmSHADOW_IN,
     ((void*)0));

  /* drawing area widget for graphics */
  wDraw = XtVaCreateManagedWidget ("da", xmDrawingAreaWidgetClass,
     wFrame, 
     XmNwidth,            width, 
     XmNheight,           height, 
     ((void*)0));
  /* callbacks for expose and resize events */
  XtAddCallback (wDraw, XmNexposeCallback, (XtCallbackProc) CbExpose, 0);
  XtAddCallback (wDraw, XmNresizeCallback, (XtCallbackProc) CbResize, 0);
  /* event handler for button and key press events */
  XtAddEventHandler (wDraw, ButtonPressMask | KeyPressMask, False,
     (XtEventHandler) DoEvents, ((void*)0));
 
     wScale[4] = XtVaCreateManagedWidget ("sc5", xmScaleWidgetClass, wRowCol, 
     XmNorientation,      XmHORIZONTAL,
     XmNwidth,            150,
     XmNshowValue,        True,  
     XmNminimum,          0,
     XmNmaximum,          300,
     XmNvalue,            eta10,
     XtVaTypedArg,        XmNtitleString, TSTRING ("100*eta"),
     ((void*)0));
  XtAddCallback (wScale[4], XmNvalueChangedCallback, CbScale,
     (XtPointer) 4);
     
   for (n = 3; n < 4; n ++) {
    wButton = XtVaCreateManagedWidget ("pb9", xmPushButtonWidgetClass,
       wRowCol,
       XtVaTypedArg,      XmNlabelString, TSTRING (bLabel[n]),
       ((void*)0));
    XtAddCallback (wButton, XmNactivateCallback, CbButton, (XtPointer)((long)(n)));
  }
    
    wScale[5] = XtVaCreateManagedWidget ("sc41", xmScaleWidgetClass, wRowCol, 
     XmNorientation,      XmHORIZONTAL,
     XmNwidth,            150,
     XmNshowValue,        True,  
     XmNminimum,          10,
     XmNmaximum,          400,
     XmNvalue,            zoom_size,
     XtVaTypedArg,        XmNtitleString, TSTRING ("Zoom"),
     ((void*)0));
  XtAddCallback (wScale[5], XmNvalueChangedCallback, CbScale,
     (XtPointer) 5);
  
   /* realize and display all widgets */
  XtRealizeWidget (wTop);
  /* get window pointer */
  win = XtWindow (wDraw);
  /* create graphic context (gc) and set some values */
  gc = XCreateGC (dpy, win, 0, NULL);
  /* create pixmap */
  pixmap = (Pixmap) NULL;
  GxMakePixmap (width, height);
}

/* create a few colors for drawing */
void GxAssignColors (const char **colors)
{
  Colormap cmap;
  XColor xcScreen, xcExact;
  int n_col;

  /* get default colormap */
  cmap = DefaultColormap (dpy, DefaultScreen (dpy));
  pix = (Pixel *) malloc (10 * sizeof (Pixel));
  /* get indices of colors based on names (assume <= 10 colors) */
  for (n_col = 0; n_col < 10; n_col ++) {
    if (colors[n_col] == NULL) break;
    XAllocNamedColor (dpy, cmap, colors[n_col], &xcScreen, &xcExact);
    /* copy colormap index of requested color */
    pix[n_col] = xcScreen.pixel;
  }
}

/* set foreground color */
void GxSetFgColor (int n_col)
{
  XSetForeground (dpy, gc, pix[n_col]);
}

/* create pixmap */
void GxMakePixmap (int width, int height)
{
  /* free old pixmap - if any */
  if (pixmap) XFreePixmap (dpy, pixmap);
  /* pixmap has same size as window */
  pixmap = XCreatePixmap (dpy, win, width, height,
     DefaultDepth (dpy, DefaultScreen (dpy)));
}

/* drawing-area widget expose callback */
void CbExpose (Widget w, XtPointer clientData,
   XmDrawingAreaCallbackStruct *cbs)
{
   /*only  if final expose event of series*/
  if (((XExposeEvent *) cbs->event)->count == 0) 
     Redraw (winWidth, winHeight);
}

/* drawing-area widget resize callback */
void CbResize (Widget w, XtPointer clientData,
   XmDrawingAreaCallbackStruct *cbs)
{
  Dimension sx, sy;

  XtVaGetValues (w, XmNwidth, &sx, XmNheight, &sy, ((void*)0));
  winWidth = sx;  winHeight = sy;
  GxMakePixmap (winWidth, winHeight);
  Redraw (winWidth, winHeight);
}

/* other drawing-area widget events */
void DoEvents (Widget w, XtPointer unused, XEvent *pEvent)
{
  KeySym keySym;
  char buffer[1];

  switch (pEvent->type) {
    /* mouse button #1 press event */
   
    /* keyboard event */
    case KeyPress:
      /* interpret key code and check if letter 'q' */
      if (XLookupString ((XKeyEvent *) pEvent, buffer, 1, &keySym,
         NULL) == 1 && keySym == XK_q) GxEndGraphics ();
      break;
  }
}

/* pushbutton widgets callback */
void CbButton (Widget w, XtPointer clientData, XtPointer callData)
{
  int id,mone;
  const char *label[] = {"Stop", "Start"};

  id = (long) clientData;
  switch (id) {
    /* quit button */
    case 0:
      GxEndGraphics ();
      break;
    /* start/stop button */
    case 1:
      running = ! running;
      clock_gettime(CLOCK_MONOTONIC, &start2);
      
      /* switch labels */
      
      XtVaSetValues (w, XtVaTypedArg, XmNlabelString,
         TSTRING (label[running ? 0 : 1]), ((void*)0));
	   /*XtSetSensitive(wRadBox,!running);*/
      break;

     case 2:  /* Stretching the lattice */
    
    if(running==0)
    {
      stretch();
    }
	 break;

    case 3:  /* dump_file */
         if(running==0)
	 {
	  ndump++;
          sprintf(tmp,"plot.%d",ndump);
	  pFile = fopen (tmp,"w");
	  if(cpu_gpu==1)
	  {
	    cudaMemcpy(X,X_GPU, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(Y,Y_GPU, size, cudaMemcpyDeviceToHost);	    
	  }
	  for(mone=1; mone<=num_atoms; mone++)
	  {
	     fprintf (pFile,"%lf %lf %d %d %d %d %d %d %d\n",X[mone],Y[mone],neigh[mone][1],neigh[mone][2],neigh[mone][3],neigh[mone][4],neigh[mone][5],neigh[mone][6],n_safa[mone]);
	  }
	  fclose (pFile);
	 }
	 break;
  }
}

/* radio-box togglebutton callbacks */
void CbToggle (Widget w, XtPointer clientData, XtPointer callData)
{
  int id,mone;

  id = (long) clientData;
  /* check which button selected */
  
    if (! running)
      {
  
  if (id >= 0 && id <= 1) {
    if (((XmToggleButtonCallbackStruct *) callData)->set) md_cg_mode = id;
  }
   redr=0;
  if(md_cg_mode==1)
  {
    calc_all_energy();
    energy[4]=0.0;
    for(mone=1; mone<=num_atoms; mone++)
    {
      energy[4]+=0.5*(VX[mone]*VX[mone]+VY[mone]*VY[mone]);
    }      
      
    sprintf(tmp,"Total Energy: %.2lf\nRadial Energy: %.2lf\n3-body Energy: %.2lf\nkinetic_energy: %.2lf\nIts: %d\n",
          energy[3]+energy[4],energy[1],energy[2],energy[4],redr); /* on screen*/
     XmTextSetString (wText2,tmp);
  } 
  Redraw (winWidth, winHeight);
  
  }
}

/* scale widgets value-changed callback */
void CbScale (Widget w, XtPointer clientData, XtPointer callData)
{
  int id,mone;

  id = (long) clientData;
  /* check which scale */
  switch (id) {
    case 0:  
      k_teta_01 = ((XmScaleCallbackStruct *) callData)->value;
      k_teta=((double)(k_teta_01))/100.0;
      if(cpu_gpu==0)
      {
        calc_all_energy();      
      }
      if(cpu_gpu==1)
      {      
	calc_all_energy_GPU();    
      }
      sprintf(tmp,"Total Energy: %.2lf\nRadial Energy: %.2lf\n3-body Energy: %.2lf\nkinetic_energy: %.2lf\nIts: %d\n",
          energy[3]+energy[4],energy[1],energy[2],energy[4],redr); /* on screen*/
         XmTextSetString (wText2,tmp);	
      break;
    case 1:  /* change no. of unredrawing computions */
      update = ((XmScaleCallbackStruct *) callData)->value;
      break;
    case 2: 
      zoom_x = ((XmScaleCallbackStruct *) callData)->value;      
      DY_PLOT=sqrt(3.0)*A_lat/2.0*((double)(zoom_size)/1.515152-1.0);
      DX_PLOT=(double)(zoom_size)*A_lat;
      Y_CV=(double)(zoom_y-1)/100.0 *(double)(2*N+2)/((double)(zoom_size)/1.515152);    
      X_CV=(double)(zoom_x-1)/100.0 *(double)(T)/((double)(zoom_size)+1.0);           
      if(cpu_gpu==0)
      {
        for(mone=1; mone<=num_atoms; mone++)
        {
         Y_PLOT[mone]=(int)(((Y[mone]-YMIN+A_lat/2)/(DY_PLOT)-Y_CV)*(double)(winHeight)*0.95);
         X_PLOT[mone]=(int)(((X[mone]-XMIN+A_lat/2)/(DX_PLOT)-X_CV)*(double)(winWidth)*0.95);
        }
      }
      else
      {
         cudaMemcpy(X,X_GPU, size, cudaMemcpyDeviceToHost);
         cudaMemcpy(Y,Y_GPU, size, cudaMemcpyDeviceToHost);
         CALC_UXY_PLOT<<<blocksPerGrid, threadsPerBlock>>>(X_PLOT_GPU,Y_PLOT_GPU,X_GPU,Y_GPU,DX_PLOT,DY_PLOT,X_CV,Y_CV,YMIN,XMIN,A_lat,winHeight,winWidth,Na);
	 cudaMemcpy(X_PLOT, X_PLOT_GPU, size_int, cudaMemcpyDeviceToHost);
         cudaMemcpy(Y_PLOT, Y_PLOT_GPU, size_int, cudaMemcpyDeviceToHost);      
      }
      Redraw (winWidth, winHeight);
      break;
    case 3: 
      zoom_y = ((XmScaleCallbackStruct *) callData)->value;      
      DY_PLOT=sqrt(3.0)*A_lat/2.0*((double)(zoom_size)/1.515152-1.0);
      DX_PLOT=(double)(zoom_size)*A_lat;
      Y_CV=(double)(zoom_y-1)/100.0 *(double)(2*N+2)/((double)(zoom_size)/1.515152);    
      X_CV=(double)(zoom_x-1)/100.0 *(double)(T)/((double)(zoom_size)+1.0);       
      if(cpu_gpu==0)
      {
        for(mone=1; mone<=num_atoms; mone++)
        {
         Y_PLOT[mone]=(int)(((Y[mone]-YMIN+A_lat/2)/(DY_PLOT)-Y_CV)*(double)(winHeight)*0.95);
         X_PLOT[mone]=(int)(((X[mone]-XMIN+A_lat/2)/(DX_PLOT)-X_CV)*(double)(winWidth)*0.95);
        }
      }
      else
      {
         cudaMemcpy(X,X_GPU, size, cudaMemcpyDeviceToHost);
         cudaMemcpy(Y,Y_GPU, size, cudaMemcpyDeviceToHost);
         CALC_UXY_PLOT<<<blocksPerGrid, threadsPerBlock>>>(X_PLOT_GPU,Y_PLOT_GPU,X_GPU,Y_GPU,DX_PLOT,DY_PLOT,X_CV,Y_CV,YMIN,XMIN,A_lat,winHeight,winWidth,Na);
	 cudaMemcpy(X_PLOT, X_PLOT_GPU, size_int, cudaMemcpyDeviceToHost);
         cudaMemcpy(Y_PLOT, Y_PLOT_GPU, size_int, cudaMemcpyDeviceToHost);      
      }
      Redraw (winWidth, winHeight);
      break;
     case 4:   
      eta10=((XmScaleCallbackStruct *) callData)->value; 
      ETA=((double)(eta10))/100.0;
      break;
     case 5:   
      zoom_size=((XmScaleCallbackStruct *) callData)->value;
      DY_PLOT=sqrt(3.0)*A_lat/2.0*((double)(zoom_size)/1.515152-1.0);
      DX_PLOT=(double)(zoom_size)*A_lat;
      Y_CV=(double)(zoom_y-1)/100.0 *(double)(2*N+2)/((double)(zoom_size)/1.515152);    
      X_CV=(double)(zoom_x-1)/100.0 *(double)(T)/((double)(zoom_size)+1.0);           
      if(cpu_gpu==0)
      {
        for(mone=1; mone<=num_atoms; mone++)
        {
         Y_PLOT[mone]=(int)(((Y[mone]-YMIN+A_lat/2)/(DY_PLOT)-Y_CV)*(double)(winHeight)*0.95);
         X_PLOT[mone]=(int)(((X[mone]-XMIN+A_lat/2)/(DX_PLOT)-X_CV)*(double)(winWidth)*0.95);
        }
      }
      else
      {
         cudaMemcpy(X,X_GPU, size, cudaMemcpyDeviceToHost);
         cudaMemcpy(Y,Y_GPU, size, cudaMemcpyDeviceToHost);
         CALC_UXY_PLOT<<<blocksPerGrid, threadsPerBlock>>>(X_PLOT_GPU,Y_PLOT_GPU,X_GPU,Y_GPU,DX_PLOT,DY_PLOT,X_CV,Y_CV,YMIN,XMIN,A_lat,winHeight,winWidth,Na);
	 cudaMemcpy(X_PLOT, X_PLOT_GPU, size_int, cudaMemcpyDeviceToHost);
         cudaMemcpy(Y_PLOT, Y_PLOT_GPU, size_int, cudaMemcpyDeviceToHost);         
      }
      Redraw (winWidth, winHeight);
     break;
  }
}


/* redraw entire window contents */
void Redraw (int width, int height)
{ 
  int j,mone,mone_bond,b1,b2,for_printing,Xmid_PLOT,Ymid_PLOT,ballRad_celY,ballRad_celX;
  double Xmid,Ymid;
  /* set color */
 
  for_printing=1;
  XSetLineAttributes(dpy, gc, 1, LineSolid ,CapButt ,JoinMiter);
  GxSetFgColor (ballColor+4);
  if(for_printing==1)GxSetFgColor (ballColor+5);
  /* fill window background - draw into pixmap */
  XFillRectangle (dpy, pixmap, gc, 0, 0, width, height);
  /* set chosen color */

  GxSetFgColor (ballColor + 2);
  if(for_printing==1)GxSetFgColor (ballColor);

  Xmid=A_lat*double(T)/2.0+1e-5;
  Ymid=0.0+1e-5;
  DY_PLOT=sqrt(3.0)*A_lat/2.0*((double)(zoom_size)/1.515152-1.0);
  DX_PLOT=(double)(zoom_size)*A_lat;
  Y_CV=(double)(zoom_y-1)/100.0 *(double)(2*N+2)/((double)(zoom_size)/1.515152);    
  X_CV=(double)(zoom_x-1)/100.0 *(double)(T)/((double)(zoom_size)+1.0);  
  Ymid_PLOT=(int)(((Ymid-YMIN+A_lat/2)/(DY_PLOT)-Y_CV)*(double)(winHeight)*0.95);
  Xmid_PLOT=(int)(((Xmid-XMIN+A_lat/2)/(DX_PLOT)-X_CV)*(double)(winWidth)*0.95);
  ballRad_celY=(int)(((x_par)/(DY_PLOT))*(double)(winHeight)*0.95);
  ballRad_celX=(int)(((x_par)/(DX_PLOT))*(double)(winWidth)*0.95);
  XFillArc (dpy, pixmap, gc, Xmid_PLOT-ballRad_celX,Ymid_PLOT-ballRad_celY,
      ballRad_celX*2, ballRad_celY*2, 0, 360 * 64);

  if(for_printing==1)GxSetFgColor (ballColor+3);      
  for(mone=1; mone<=num_atoms; mone++)
  {	
    int index_tsiur=0;
    for(j=1;j<=6;j++)
    {
      if(neigh[mone][j]!=-1)
      {
        index_tsiur=1;
      }
    }
    if((Y_PLOT[mone]>(int)((double)(-winHeight)*0.05))&&(Y_PLOT[mone]<(int)((double)(winHeight)*1.05))&&
    (X_PLOT[mone]>(int)((double)(-winWidth)*0.05))&&(X_PLOT[mone]<(int)((double)(winWidth)*1.05))&&(index_tsiur==1))
    XFillArc (dpy, pixmap, gc, X_PLOT[mone]-ballRad,Y_PLOT[mone]-ballRad,
      2 * ballRad, 2 * ballRad, 0, 360 * 64);
   }    
   
  if(for_printing==1)XSetLineAttributes(dpy, gc, 2, LineSolid ,CapButt ,JoinMiter);
  for(mone_bond=1; mone_bond<=num_bonds; mone_bond++)
  {
     b1=bond_atom[mone_bond][1];
     b2=bond_atom[mone_bond][2];
     if((b1>0)&&(b2>0))
     { 
      if((Y_PLOT[b1]>(int)((double)(-winHeight)*0.05))&&(Y_PLOT[b1]<(int)((double)(winHeight)*1.05))&&
      (X_PLOT[b1]>(int)((double)(-winWidth)*0.05))&&(X_PLOT[b1]<(int)((double)(winWidth)*1.05)))
      XDrawLine (dpy, pixmap , gc, X_PLOT[b1], Y_PLOT[b1], X_PLOT[b2], Y_PLOT[b2]);
     }
  }
     
 XSetLineAttributes(dpy, gc, 2, LineSolid ,CapButt ,JoinMiter);

   /* copy pixmap to window   */ 
  XCopyArea (dpy, pixmap, win, gc, 0, 0, width, height, 0, 0);
  /* flush X queue */
  XSync (dpy, False);

}


/* work procedure */
int WorkProc ()
{
  int i;
  if (running)
  { 
     if(md_cg_mode==0)
     {
       conjugate_gradient();
       energy_store[redr]=energy[3];
       if(redr<(num_realizations-1))
       {
         initial_cell();
         stretch();
       }
       if(redr>=(num_realizations-1))
       {
         running=0;
	  sprintf(tmp,"energy.txt");
	  pFile = fopen (tmp,"w");
          for(i=0;i<num_realizations;i++)
	  {
	    fprintf (pFile,"%lf\n",energy_store[i]);
	  }
	  fclose (pFile);
       }
     }
     
     if(md_cg_mode==1)
     {
       time_step();
       if ((redr%update==0)&&(redr!=0))
       {
         Redraw (winWidth, winHeight);
       }   
     }
     redr++;  
  }
  return (0); /* must be zero to reschedule procedure */
}


/* timed delay */
void GxDelay (int mSec)
{
  struct timeval tv;
  
  tv.tv_sec = mSec / 1000;  tv.tv_usec = 1000 * (mSec % 1000);
  select (0, NULL, NULL, NULL, &tv);
}

void build()	/* build reshet */
{
int i,j,a,b,mone,mone_bond;

    num_atoms=(2*N+2)*T;    
    num_bonds=(2*N+1)*T+(2*N+1)*(T-1)+(T-1)*(2*N+2);
 
    Na = num_atoms+1;
    Nb = num_bonds+1;
    size = Na * sizeof(double);
    size2 = 2*Na * sizeof(double);
    sizeb = Nb * sizeof(double);
    size4 = 5 * sizeof(double);
    size_int = Na * sizeof(int);
    size_intb = Nb * sizeof(int);
    size_num_realizations = num_realizations * sizeof(double);

    X = (double*)malloc(size);
    Y = (double*)malloc(size);
    X0 = (double*)malloc(size);
    Y0 = (double*)malloc(size);
    n_safa = (int*)malloc(size_int);
    Y_PLOT = (int*)malloc(size_int);
    X_PLOT = (int*)malloc(size_int);
    A_lat0 = (double*)malloc(sizeb);
    bond_atom = Make2DIntArray(Nb, 3);
    neigh = Make2DIntArray(Na, 7);
    temp_int1 = (int*)malloc(size_int);
    temp_int2 = (int*)malloc(size_intb);
    temp_double = (double*)malloc(sizeb);
    type = (int*)malloc(size_intb);
    type_odd_even = (int*)malloc(size_int);
    energy_store = (double*)malloc(size_num_realizations); 
    
    mone=0;  
	 
    YMIN=-sqrt(3.0)*A_lat/2.0*(double)(N)-sqrt(3.0)*A_lat/4.0;
    for(j=1; j<=2*(N+1); j=j+2)
     {
      for(i=1; i<=T; i++)          /* ODD COLUMNS*/ 
      {
         mone++;
        	
	 Y[mone]=YMIN+sqrt(3.0)*A_lat/2.0*(double)(j-1);
         X[mone]=(double)(i-1)*A_lat;
	 type_odd_even[mone]=1;
	 
	  neigh[mone][1]=mone-T-1;
	  neigh[mone][2]=mone-1;
	  neigh[mone][3]=mone+T-1;	
	  neigh[mone][4]=mone-T;
	  neigh[mone][5]=mone+T;
	  neigh[mone][6]=mone+1;
	 
	  if(j==1)
	  {
	   neigh[mone][1]=-1;
	   neigh[mone][4]=-1;
	  }
	 
	  if(i==T)
	  {
	   neigh[mone][6]=-1;
	  }
	  if(i==1)
	  {
	  neigh[mone][1]=-1;
	  neigh[mone][2]=-1;
	  neigh[mone][3]=-1;
	  }
       }
       mone=mone+T;
      }
      
      
      mone=T;
      for(j=2; j<=2*(N+1); j=j+2)
       {
        for(i=1; i<=T; i++)          /*EVEN COLUMNS*/
        {	 
	  mone++;
	  Y[mone]=YMIN+sqrt(3)*A_lat/2.0*(double)(j-1);
	  X[mone]=A_lat/2.0+(double)(i-1)*A_lat;
	  type_odd_even[mone]=2;

	  neigh[mone][1]=mone-1;
	  neigh[mone][2]=mone-T;
	  neigh[mone][3]=mone+T;	
	  neigh[mone][4]=mone-T+1;
	  neigh[mone][5]=mone+1;
	  neigh[mone][6]=mone+T+1;	
	 
	  if(j==(2*N+2))
	  {
	   neigh[mone][3]=-1;
	   neigh[mone][6]=-1;
	  }
	  if(i==T)
	  {
	   neigh[mone][4]=-1;
	   neigh[mone][5]=-1;
	   neigh[mone][6]=-1;
	  } 
	  if(i==1)
	  {
	   neigh[mone][1]=-1;
	  } 
       }
       mone=mone+T;
      }
      mone=mone-T;

      mone=0;
      for(j=1; j<=2*(N+1); j=j+1)
      {
       for(i=1; i<=T; i++)
        {
         mone++;
	 if((j==1)||(j==(2*N+2))||(i==1)||(i==T))
	  {
	   n_safa[mone]=1;
	  }
	 else
	  {
	   n_safa[mone]=0;
	  }
        }
      }
      
      mone_bond=0;
      for(j=1; j<=num_atoms; j++)
       {
        for(i=1; i<=6; i++)
	 {
	  if(neigh[j][i]>0)
	   {
	    mone_bond++;
	   }
	 }
       }
      
      YMAX=Y[(2*N+2)*T];
      XMIN=X[1];
      XMAX=DMAX(X[(2*N+2)*T],X[(2*N+1)*T]);
             
      DY_PLOT=sqrt(3.0)*A_lat/2.0*((double)(zoom_size)/1.515152-1.0);
      DX_PLOT=(double)(zoom_size)*A_lat;
      Y_CV=(double)(zoom_y-1)/100.0 *(double)(2*N+2)/((double)(zoom_size)/1.515152);    
      X_CV=(double)(zoom_x-1)/100.0 *(double)(T)/((double)(zoom_size)+1.0);  
           
      for(mone=1; mone<=num_atoms; mone++)
      {
       Y_PLOT[mone]=(int)(((Y[mone]-YMIN+A_lat/2)/(DY_PLOT)-Y_CV)*(double)(winHeight)*0.95);
       X_PLOT[mone]=(int)(((X[mone]-XMIN+A_lat/2)/(DX_PLOT)-X_CV)*(double)(winWidth)*0.95);
       Y0[mone]=Y[mone];
       X0[mone]=X[mone];
      }
      
      num_vert1=(2*N+1)*T;
      num_vert2=(2*N+1)*(T-1);
      num_hor=(T-1)*(2*N+2);
      
      mone_bond=0;
      for(j=1; j<=2*N+1; j++)
       {
        for(i=1; i<=T; i++)         
        {
	   mone_bond=mone_bond+1;
           bond_atom[mone_bond][1]=T*(j-1)+i;
           bond_atom[mone_bond][2]=T*(j)+i;
	}
       }
           
      for(j=1; j<=2*N+1; j=j+2)
       {
        for(i=1; i<=(T-1); i++)         
        {
	  mone_bond=mone_bond+1;
          bond_atom[mone_bond][1]=T*(j-1)+i+1;
          bond_atom[mone_bond][2]=T*(j)+i;
	}
       }
      for(j=2; j<=2*N+1; j=j+2)
       {
        for(i=1; i<=(T-1); i++)         
        {
	  mone_bond=mone_bond+1;
          bond_atom[mone_bond][1]=T*(j-1)+i;
          bond_atom[mone_bond][2]=T*(j)+i+1;
	}
       }

      for(j=1; j<=2*(N+1); j++)
       {
        for(i=1; i<=(T-1); i++)         
        {
	  mone_bond=mone_bond+1;
	  bond_atom[mone_bond][1]=T*(j-1)+i;
          bond_atom[mone_bond][2]=T*(j-1)+i+1;
	}
       }
	     
     for(mone_bond=num_vert1+num_vert2+1;mone_bond<=num_bonds; mone_bond++)
     {
       a=bond_atom[mone_bond][1];
       b=bond_atom[mone_bond][2];
    
       if(X0[a]<X0[b])
       {
         type[mone_bond]=1;
       }
       else
       {
         type[mone_bond]=2;    
       }
     }
            
      for(mone_bond=1;mone_bond<=num_bonds; mone_bond++)
      {
        A_lat0[mone_bond]=A_lat;
      }
      

      for(i=1; i<=4; i++)
      {
        energy[i]=0.0;
        old_energy[i]=0.0;
      }

      calc_all_energy();
      
      sprintf(tmp,"Total Energy: %.2lf\nRadial Energy: %.2lf\n3-body Energy: %.2lf\nkinetic_energy: %.2lf\nIts: %d\n",
          energy[3]+energy[4],energy[1],energy[2],energy[4],redr); /* on screen*/
         XmTextSetString (wText2,tmp);
      
     if (cpu_gpu == 1)
     {
       /*Allocate vectors in device memory*/
       cudaMalloc(&X_GPU, size);
       cudaMalloc(&Y_GPU, size);
       cudaMalloc(&X_PLOT_GPU, size_int);
       cudaMalloc(&Y_PLOT_GPU, size_int);

       cudaMemcpy(X_GPU, X, size, cudaMemcpyHostToDevice);
       cudaMemcpy(Y_GPU, Y, size, cudaMemcpyHostToDevice);
       
       cudaMalloc(&n_safa_GPU, size_int);
       cudaMemcpy(n_safa_GPU, n_safa, size_int, cudaMemcpyHostToDevice);
    }
    initial_md();
}

void calc_all_energy()
{
 int i,j,a,b,c,vec1[4],vec2[4],mone;
 double r,R1,R2,X1,X2,Y1,Y2,cos_teta1,teta1;
 
 for(i=1; i<=4; i++)
 {
  energy[i]=0.0;
 }
 for(i=1; i<=num_bonds; i++)
 {
  if((bond_atom[i][1]>0)&&(bond_atom[i][2]>0))
   {
     r=sqrt((X[bond_atom[i][1]]-X[bond_atom[i][2]])*(X[bond_atom[i][1]]-X[bond_atom[i][2]])+(Y[bond_atom[i][1]]-Y[bond_atom[i][2]])*(Y[bond_atom[i][1]]-Y[bond_atom[i][2]]));
     energy[1]+=0.5*(r-A_lat0[i])*(r-A_lat0[i]);
   }
 }

if(k_teta>1.0e-5)
 {
    vec1[1]=2; vec1[2]=1; vec1[3]=3; 
    vec2[1]=6; vec2[2]=5; vec2[3]=4; 
    mone=0;       
    for(j=1; j<=2*(N+1); j=j+2)
     {
      for(i=1; i<=T; i++)          /* ODD COLUMNS*/ 
      {
        mone++;
        for(c=1; c<=3; c++)
        {
          a=neigh[mone][vec1[c]];
 	  if(a!=-1)
 	  {
 	    X1=X[a]-X[mone]; Y1=Y[a]-Y[mone];
 	  }
 	  else
	  {
            X1=-999999.0;	 
	  }
	  b=neigh[mone][vec2[c]];
	  if(b!=-1)
	  {
	    X2=X[b]-X[mone]; Y2=Y[b]-Y[mone];
	  }
	  else
	  {
            X2=-999999.0;	 
	  }
	  R1=sqrt(X1*X1+Y1*Y1);
          R2=sqrt(X2*X2+Y2*Y2);
          if((X1!=-999999.0)&&(X2!=-999999.0))
          {
            cos_teta1=(X1*X2+Y1*Y2)/R1/R2;
	    teta1=fabs(PI-acos((cos_teta1+1.0e-15)));
          }
          else
          {
            teta1=0.0;
	    cos_teta1=-1.0;
          }
          energy[2]=energy[2]+0.5*k_teta*teta1*teta1;
	  /*printf("%d %d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %d\n",mone,a,b,0.5*k_teta*teta1*teta1,teta1,cos_teta1,X1,X2,Y1,Y2,R1,R2,n_safa[mone]);*/
	  /*energy[2]=energy[2]+0.5*k_teta*(cos_teta1+1.0)*(cos_teta1+1.0);*/
        }
     }
       mone=mone+T;
     }
      
    
     vec1[1]=1; vec1[2]=2; vec1[3]=3; 
     vec2[1]=5; vec2[2]=6; vec2[3]=4; 
     mone=T;
     for(j=2; j<=2*(N+1); j=j+2)
       {
        for(i=1; i<=T; i++)          /*EVEN COLUMNS*/
        {	 
 	   
         mone++;
         for(c=1; c<=3; c++)
         {
           a=neigh[mone][vec1[c]];
  	   if(a!=-1)
  	   {
  	     X1=X[a]-X[mone]; Y1=Y[a]-Y[mone];
 	   }
 	   else
	   {
             X1=-999999.0;	 
	   }
	   b=neigh[mone][vec2[c]];
	   if(b!=-1)
	   {
	     X2=X[b]-X[mone]; Y2=Y[b]-Y[mone];
	   }
	   else
	   {
             X2=-999999.0;	 
	   }
	   R1=sqrt(X1*X1+Y1*Y1);
           R2=sqrt(X2*X2+Y2*Y2);
           if((X1!=-999999.0)&&(X2!=-999999.0))
           {
             cos_teta1=(X1*X2+Y1*Y2)/R1/R2;
	     teta1=fabs(PI-acos((cos_teta1+1.0e-15)));
           }
           else
           {
             teta1=0.0;
	     cos_teta1=-1.0;
           }
          energy[2]=energy[2]+0.5*k_teta*teta1*teta1;
	  /*printf("%d %d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %d\n",mone,a,b,0.5*k_teta*teta1*teta1,teta1,cos_teta1,X1,X2,Y1,Y2,R1,R2,n_safa[mone]);*/
	  /*energy[2]=energy[2]+0.5*k_teta*(cos_teta1+1.0)*(cos_teta1+1.0);*/
         }
        }
        mone=mone+T;
       }
     mone=mone-T;
 }


  energy[3]=energy[1]+energy[2];
  /*printf("in energy, %2lf %2lf %2lf\n",energy[1],energy[2],energy[3]);*/

}

void calc_all_energy_GPU()
{
  int i;
  
  for(i=1; i<=4; i++)
  {
   energy[i]=0.0;
  }
  cudaMemcpy(energy_GPU, energy,size4, cudaMemcpyHostToDevice);
  
  if(shared==1)
  { 
    calc_raidial_energy_GPU<<<blocksPerGridMat_b, threadsPerBlock>>>(X_GPU,Y_GPU,bond_atom_small1_GPU,energy_GPU,
                               bond_atom_small2_GPU,A_lat0_GPU,bond_id_GPU,
			       num_atoms_in_bonds_GPU,atom_list_bonds_GPU,bond_list_GPU);		  
    if(k_teta>1.0e-5)
    {
      calc_3body_energy_GPU<<<blocksPerGridMat, threadsPerBlock>>>(X_GPU,Y_GPU,k_teta,energy_GPU,
			       atom_list_GPU,atom_id_GPU,atom_id2_GPU,neigh_small_grid_GPU,type_odd_even_lists_GPU);
    }
  }
  if(shared==0)
  {
    calc_raidial_energy_no_shared_GPU<<<blocksPerGrid_b, threadsPerBlock>>>(X_GPU,Y_GPU,Nb,bond_atom1_GPU,energy_GPU,
                               bond_atom2_GPU,A_lat0_GPU);
    if(k_teta>1.0e-5)
    {
      calc_3body_energy_no_shared_GPU<<<blocksPerGrid, threadsPerBlock>>>(X_GPU,Y_GPU,Na,k_teta,type_odd_even_GPU,energy_GPU,
			       neigh1_GPU,neigh2_GPU,neigh3_GPU,neigh4_GPU,neigh5_GPU,neigh6_GPU);
    }
  }
  cudaMemcpy(energy,energy_GPU,size4, cudaMemcpyDeviceToHost);
  energy[3]=energy[1]+energy[2];  
}

      void CALC_FORCE()
      {
      int mone,a,b,c,vec1[4],vec2[4];
      double R12,X1,Y1,X2,Y2,F_RAD,SCALAR,VX1,VY1,VX2,VY2,X00,Y00,R1,R2,K12,cos_teta1,teta1;
      
      for(mone=1; mone<=num_atoms; mone++)
       {
        AX[mone]=0.0;
        AY[mone]=0.0;
       }	
            
      for(mone=1; mone<=num_bonds; mone++)
       {
	if((bond_atom[mone][1]>0)&&(bond_atom[mone][2]>0))
        {
	  /* RADIAL-FORCE */
	  X1=X[bond_atom[mone][1]];
          Y1=Y[bond_atom[mone][1]];
          X2=X[bond_atom[mone][2]];
          Y2=Y[bond_atom[mone][2]];
          R12=sqrt((X1-X2)*(X1-X2)+(Y1-Y2)*(Y1-Y2));
	  K12=1.0;
	
          F_RAD=(R12-A_lat0[mone])*K12;
	  AX[bond_atom[mone][1]]+=(X2-X1)/R12*F_RAD;
          AY[bond_atom[mone][1]]+=(Y2-Y1)/R12*F_RAD;
          AX[bond_atom[mone][2]]+=(X1-X2)/R12*F_RAD;
          AY[bond_atom[mone][2]]+=(Y1-Y2)/R12*F_RAD;

          /* KELVIN VISCOSITY */
          if(md_cg_mode==1)
	  {
	    VX1=VX[bond_atom[mone][1]];
            VY1=VY[bond_atom[mone][1]];
            VX2=VX[bond_atom[mone][2]];
            VY2=VY[bond_atom[mone][2]];
            SCALAR=ETA*K12*((VX2-VX1)*(X2-X1)+(VY2-VY1)*(Y2-Y1))/R12;
            AX[bond_atom[mone][1]]+=(X2-X1)/R12*SCALAR;
            AY[bond_atom[mone][1]]+=(Y2-Y1)/R12*SCALAR;
            AX[bond_atom[mone][2]]+=(X1-X2)/R12*SCALAR;
            AY[bond_atom[mone][2]]+=(Y1-Y2)/R12*SCALAR;
	  }
	 }
       }

      if(k_teta>1.0e-5)
      { 
        /* 3-BODY FORCE-LAW */      
        
	for(mone=1;mone<=num_atoms;mone++)
	{
	if(type_odd_even[mone]==1)
	    {
	      vec1[1]=2; vec1[2]=1; vec1[3]=3;
              vec2[1]=6; vec2[2]=5; vec2[3]=4;
            } 
	    if(type_odd_even[mone]==2)
	    {
              vec1[1]=1; vec1[2]=2; vec1[3]=3;
              vec2[1]=5; vec2[2]=6; vec2[3]=4;
            } 
            X00=X[mone]; Y00=Y[mone];
	    for(c=1; c<=3; c++)
            {
              a=neigh[mone][vec1[c]];
 	      if(a!=-1)
 	      {
 	  	X1=X[a]; Y1=Y[a];
 	      }
 	      else
	      {
          	X1=-999999.0;  
	      }
	      b=neigh[mone][vec2[c]];
	      if(b!=-1)
	      {
	  	X2=X[b]; Y2=Y[b];
	      }
	      else
	      {
          	X2=-999999.0;  
	      }
	      R1=sqrt((X1-X00)*(X1-X00)+(Y1-Y00)*(Y1-Y00));
              R2=sqrt((X2-X00)*(X2-X00)+(Y2-Y00)*(Y2-Y00));
              if((X1!=-999999.0)&&(X2!=-999999.0))
              {
		 cos_teta1=((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R1/R2;
	         teta1=fabs(PI-acos((cos_teta1+1.0e-15)));
		 AX[mone]-=k_teta*teta1/(sin(teta1)+1e-30)*((-(X1-X00)-(X2-X00))/R1/R2+
	  				    (X1-X00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R1/R1/R1/R2+
	        			    (X2-X00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R2/R2/R2/R1);
          	 AY[mone]-=k_teta*teta1/(sin(teta1)+1e-30)*((-(Y1-Y00)-(Y2-Y00))/R1/R2+
	  				    (Y1-Y00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R1/R1/R1/R2+
	        			    (Y2-Y00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R2/R2/R2/R1);
	
          	 AX[a]-=k_teta*teta1/(sin(teta1)+1e-30)*((X2-X00)/R1/R2-
	  				    (X1-X00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R1/R1/R1/R2);
          	 AY[a]-=k_teta*teta1/(sin(teta1)+1e-30)*((Y2-Y00)/R1/R2-
	  				    (Y1-Y00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R1/R1/R1/R2);

          	 AX[b]-=k_teta*teta1/(sin(teta1)+1e-30)*((X1-X00)/R1/R2-
	  				    (X2-X00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R2/R2/R2/R1);
          	 AY[b]-=k_teta*teta1/(sin(teta1)+1e-30)*((Y1-Y00)/R1/R2-
	  				    (Y2-Y00)*((X1-X00)*(X2-X00)+(Y1-Y00)*(Y2-Y00))/R2/R2/R2/R1);
              }
            }
           }
        }
  
      }          
      
      void CALC_U_AND_V()
      {
      int mone;
      
       for(mone=1; mone<=num_atoms; mone++)
       {
        if(n_safa[mone]!=1)
	{
         VX[mone]=VX[mone]+(AX[mone])*DT;
         VY[mone]=VY[mone]+(AY[mone])*DT;
	 X[mone]=X[mone]+VX[mone]*DT;
         Y[mone]=Y[mone]+VY[mone]*DT;
        }
       }            
      }

      void initial_md()
      {
       int J,mone;
      
       time_md=0.0;
      
       AX = (double*)malloc(size);
       AY = (double*)malloc(size);
       VX = (double*)malloc(size);
       VY = (double*)malloc(size);
   
       bond_atom0 = Make2DIntArray(Nb, 3);
       neigh0 = Make2DIntArray(Na, 7);

      if(cpu_gpu==1)
       {          
	  initial_gpu_lists();		  
	  blocksPerGrid = (Na + threadsPerBlock - 1) / threadsPerBlock;
          blocksPerGrid_b = (Nb + threadsPerBlock - 1) / threadsPerBlock;
       }

       for(mone=1; mone<=num_atoms; mone++)
       {
         VX[mone]=0.0;
         VY[mone]=0.0;
         Y0[mone]=Y[mone];
         X0[mone]=X[mone];
	 for (J=1; J<=6; J++)
	 {
	   neigh0[mone][J]=neigh[mone][J];
	 }
       }
      for(mone=1; mone<=num_bonds; mone++)
       {
        bond_atom0[mone][1]=bond_atom[mone][1];
	bond_atom0[mone][2]=bond_atom[mone][2];
       }

      if(cpu_gpu==1)
       {
          /* Allocate vectors in device memory*/
	  cudaMalloc(&AX_GPU, size);	  
	  cudaMalloc(&AY_GPU, size);
          cudaMalloc(&VX_GPU, size);
          cudaMalloc(&VY_GPU, size);
          cudaMalloc(&A_lat0_GPU, sizeb);
          cudaMalloc(&energy_GPU,size4);
 	  
	  if(shared==0)
	  {
	    cudaMalloc(&bond_atom1_GPU,size_intb);
	    cudaMalloc(&bond_atom2_GPU,size_intb);
            cudaMalloc(&neigh1_GPU,size_int);
            cudaMalloc(&neigh2_GPU,size_int);
            cudaMalloc(&neigh3_GPU,size_int);
            cudaMalloc(&neigh4_GPU,size_int);
            cudaMalloc(&neigh5_GPU,size_int);
            cudaMalloc(&neigh6_GPU,size_int);
            cudaMalloc(&type_odd_even_GPU,size_int);
	  }
       }
       
       initial_cell();
       
     }
      
     void initial_cell()
     {
       int
       J,mone,i,j,k,a,b,p,mone_change,n_init,c,ii,a_atom,b_atom,a_lists,b_lists,d;
       double r1,r2,Xmid,Ymid,shipua,x1,y1,x2,y2,acof,bcof,ccof,x2new1,x2new2,y2new1,y2new2,r2new1,r2new2,x0,y0;

      for(mone=1; mone<=num_atoms; mone++)
       {
         VX[mone]=0.0;
         VY[mone]=0.0;
         Y[mone]=Y0[mone];
         X[mone]=X0[mone];
	 for (J=1; J<=6; J++)
	 {
	   neigh[mone][J]=neigh0[mone][J];
	 }
       }
      for(mone=1; mone<=num_bonds; mone++)
       {
        bond_atom[mone][1]=bond_atom0[mone][1];
	bond_atom[mone][2]=bond_atom0[mone][2];
       }     
      if(cpu_gpu==1)
      {
	for(mone=0;mone<max_nx*max_ny*fac_threads*7;mone++)
	{
	  neigh_small_grid[mone]=neigh_small_grid0[mone];
	}
	for(mone=0;mone<max_nx_b*max_ny_b*threadsPerBlock;mone++)
	{
	  bond_atom_small1[mone]=bond_atom0_small1[mone];
	  bond_atom_small2[mone]=bond_atom0_small2[mone];	  
	}
      } 

      n_init=(int)((double)(num_bonds)*0.4);
      mone_change=0;
      while(mone_change <n_init)
      {
        p=min((int)(drand48()*(double)(num_bonds)+1.0),num_bonds);
        a=bond_atom[p][1];
        b=bond_atom[p][2];
        if((a>0)&&(b>0))
        {
          mone_change++;
	  bond_atom[p][1]=-1;
          bond_atom[p][2]=-1;
          for(j=1; j<=6; j++)
          {
            if(neigh[a][j]==b)
            {
              neigh[a][j]=-1;
            }
            if(neigh[b][j]==a)
            {
              neigh[b][j]=-1;
            }
          }
        }
      }
      
      Xmid=A_lat*double(T)/2.0+1e-5;
      Ymid=0.0+1e-5;
      for(i=1; i<=num_bonds; i++)
      {
        a=bond_atom[i][1];
        b=bond_atom[i][2];
        if((a>0)&&(b>0))
        {
          r1=sqrt((X[a]-Xmid)*(X[a]-Xmid)+(Y[a]-Ymid)*(Y[a]-Ymid));
          r2=sqrt((X[b]-Xmid)*(X[b]-Xmid)+(Y[b]-Ymid)*(Y[b]-Ymid));
          if((r1<=r_ball)&&(r2<=r_ball))
          {
            bond_atom[i][1]=-1;
            bond_atom[i][2]=-1;
            for(j=1; j<=6; j++)
            {
              if(neigh[a][j]==b)
              {
        	neigh[a][j]=-1;
              }
              if(neigh[b][j]==a)
              {
        	neigh[b][j]=-1;
              }
            }
          }
        }
      }
      for(i=1; i<=num_bonds; i++)
      {
        a=bond_atom[i][1];
        b=bond_atom[i][2];
        if((a>0)&&(b>0))
        {
          r1=sqrt((X[a]-Xmid)*(X[a]-Xmid)+(Y[a]-Ymid)*(Y[a]-Ymid));
          r2=sqrt((X[b]-Xmid)*(X[b]-Xmid)+(Y[b]-Ymid)*(Y[b]-Ymid));
	  if((r1<=r_ball)&&(r2>r_ball))
          {
            for(j=1; j<=6; j++)
            {
              if(neigh[a][j]!=b)
              {
		for(mone=1;mone<=num_bonds;mone++)
		{
		  if(((bond_atom[mone][1]==neigh[a][j])&&(bond_atom[mone][2]==a))||((bond_atom[mone][2]==neigh[a][j])&&(bond_atom[mone][1]==a)))
		  {
		    c=bond_atom[mone][1];
		    d=bond_atom[mone][2];
		    if((c>0)&&(d>0))
                    {
		      bond_atom[mone][1]=-1;
		      bond_atom[mone][2]=-1;
		      for(k=1; k<=6; k++)
                      {
                        if(neigh[c][k]==d)
                        {
        	          neigh[c][k]=-1;
                        }
                        if(neigh[d][k]==c)
                        {
        	          neigh[d][k]=-1;
                        }
		      }
                    }
		  }
		}
        	neigh[a][j]=-1;
              }
            }
	    x1=X[b];
	    y1=Y[b];
	    x2=X[a];
	    y2=Y[a];
	    x0=Xmid;
	    y0=Ymid;
	    shipua=(y2-y1)/(x2-x1);
	    acof=(1.0+shipua*shipua);
	    bcof=-2.0*(x0+x1*shipua*shipua-shipua*y1+shipua*y0);
	    ccof=x0*x0+shipua*shipua*x1*x1+2.0*shipua*x1*y0-2.0*shipua*x1*y1+(y1-y0)*(y1-y0)-r_ball*r_ball;
	    x2new1=(-bcof+sqrt(bcof*bcof-4.0*acof*ccof))/2.0/acof;
	    x2new2=(-bcof-sqrt(bcof*bcof-4.0*acof*ccof))/2.0/acof;
	    y2new1=shipua*(x2new1-x1)+y1;
	    y2new2=shipua*(x2new2-x1)+y1;
	    r2new1=sqrt((x2new1-X[b])*(x2new1-X[b])+(y2new1-Y[b])*(y2new1-Y[b]));
    	    r2new2=sqrt((x2new2-X[b])*(x2new2-X[b])+(y2new2-Y[b])*(y2new2-Y[b]));
	    if(r2new1<r2new2)
    	    {
   	      X[a]=x2new1;
    	      Y[a]=y2new1;
	      A_lat0[i]=r2new1;
    	    }
    	    else
    	    {
    	      X[a]=x2new2;
    	      Y[a]=y2new2;
	      A_lat0[i]=r2new2;
    	    }
          }
	  if((r2<=r_ball)&&(r1>r_ball))
          {
            for(j=1; j<=6; j++)
            {
              if(neigh[b][j]!=a)
              {
		for(mone=1;mone<=num_bonds;mone++)
		{
		  if(((bond_atom[mone][1]==neigh[b][j])&&(bond_atom[mone][2]==b))||((bond_atom[mone][2]==neigh[b][j])&&(bond_atom[mone][1]==b)))
		  {
		    c=bond_atom[mone][1];
		    d=bond_atom[mone][2];
		    if((c>0)&&(d>0))
                    {
		      bond_atom[mone][1]=-1;
		      bond_atom[mone][2]=-1;
		      for(k=1; k<=6; k++)
                      {
                        if(neigh[c][k]==d)
                        {
        	          neigh[c][k]=-1;
                        }
                        if(neigh[d][k]==c)
                        {
        	          neigh[d][k]=-1;
                        }
		      }
                    }
		  }
		}
        	neigh[b][j]=-1;
              }
            }
	    x1=X[a];
	    y1=Y[a];
	    x2=X[b];
	    y2=Y[b];
	    x0=Xmid;
	    y0=Ymid;
	    shipua=(y2-y1)/(x2-x1);
	    acof=(1.0+shipua*shipua);
	    bcof=-2.0*(x0+x1*shipua*shipua-shipua*y1+shipua*y0);
	    ccof=x0*x0+shipua*shipua*x1*x1+2.0*shipua*x1*y0-2.0*shipua*x1*y1+(y1-y0)*(y1-y0)-r_ball*r_ball;
	    x2new1=(-bcof+sqrt(bcof*bcof-4.0*acof*ccof))/2.0/acof;
	    x2new2=(-bcof-sqrt(bcof*bcof-4.0*acof*ccof))/2.0/acof;
	    y2new1=shipua*(x2new1-x1)+y1;
	    y2new2=shipua*(x2new2-x1)+y1;
	    r2new1=sqrt((x2new1-X[a])*(x2new1-X[a])+(y2new1-Y[a])*(y2new1-Y[a]));
    	    r2new2=sqrt((x2new2-X[a])*(x2new2-X[a])+(y2new2-Y[a])*(y2new2-Y[a]));
	    if(r2new1<r2new2)
    	    {
   	      X[b]=x2new1;
    	      Y[b]=y2new1;
	      A_lat0[i]=r2new1;
    	    }
    	    else
    	    {
    	      X[b]=x2new2;
    	      Y[b]=y2new2;
	      A_lat0[i]=r2new2;      
    	    }
          }
        }
      }
      if(cpu_gpu==1)
      {
	for(c=0;c<max_nx_b*max_ny_b;c++)
 	{
	  for(ii=0;ii<bond_id[c];ii++)
	  {
	    a=bond_atom_small1[c * threadsPerBlock + ii];
	    b=bond_atom_small2[c * threadsPerBlock + ii];
	    a_atom=atom_list_bonds[c * threadsPerBlock + a];
	    b_atom=atom_list_bonds[c * threadsPerBlock + b];
	    
	    mone=bond_list[c*threadsPerBlock + ii];
	    if(bond_atom[mone][1]==-1)
	    {
	      bond_atom_small1[c * threadsPerBlock + ii]=-1;
	      bond_atom_small2[c * threadsPerBlock + ii]=-1;
	      a_lists=neigh_lists[a_atom];
              b_lists=neigh_lists[b_atom];
	      for(j=1; j<=6; j++)
	      {       
	        if(neigh_small_grid_big[a_lists+j]==neigh_small_grid_big[b_lists])neigh_small_grid[a_lists+j]=-1; 
	        if(neigh_small_grid_big[b_lists+j]==neigh_small_grid_big[a_lists])neigh_small_grid[b_lists+j]=-1;	   
	      }
	    }
	  }
	}
      }     

     if(cpu_gpu==0)
      {
        for(mone=1; mone<=num_atoms; mone++)
        {
         Y_PLOT[mone]=(int)(((Y[mone]-YMIN+A_lat/2)/(DY_PLOT)-Y_CV)*(double)(winHeight)*0.95);
         X_PLOT[mone]=(int)(((X[mone]-XMIN+A_lat/2)/(DX_PLOT)-X_CV)*(double)(winWidth)*0.95);
        }
      }
      if(cpu_gpu==1)
       {
          /* Copy vectors from host memory to device memory*/
          cudaMemcpy(VX_GPU, VX, size, cudaMemcpyHostToDevice);
          cudaMemcpy(VY_GPU, VY, size, cudaMemcpyHostToDevice);
          cudaMemcpy(X_GPU, X, size, cudaMemcpyHostToDevice);
          cudaMemcpy(Y_GPU, Y, size, cudaMemcpyHostToDevice);
          cudaMemcpy(A_lat0_GPU, A_lat0, sizeb, cudaMemcpyHostToDevice);
          
	  cudaMemcpy(neigh_small_grid_GPU, neigh_small_grid, max_nx*max_ny*fac_threads*7 * sizeof(int) , cudaMemcpyHostToDevice);
          cudaMemcpy(bond_atom_small1_GPU, bond_atom_small1, max_nx_b*max_ny_b*threadsPerBlock * sizeof(int) , cudaMemcpyHostToDevice);
          cudaMemcpy(bond_atom_small2_GPU, bond_atom_small2, max_nx_b*max_ny_b*threadsPerBlock * sizeof(int) , cudaMemcpyHostToDevice);
	  
	  cudaMemcpy(X,X_GPU, size, cudaMemcpyDeviceToHost);
          cudaMemcpy(Y,Y_GPU, size, cudaMemcpyDeviceToHost);
          CALC_UXY_PLOT<<<blocksPerGrid, threadsPerBlock>>>(X_PLOT_GPU,Y_PLOT_GPU,X_GPU,Y_GPU,DX_PLOT,DY_PLOT,X_CV,Y_CV,YMIN,XMIN,A_lat,winHeight,winWidth,Na);
	  cudaMemcpy(X_PLOT, X_PLOT_GPU, size_int, cudaMemcpyDeviceToHost);
          cudaMemcpy(Y_PLOT, Y_PLOT_GPU, size_int, cudaMemcpyDeviceToHost);
	  
	  if(shared==0)
	  {
            for(mone=1;mone<=num_bonds;mone++)
	    {
	      temp_int2[mone]=bond_atom[mone][1];
	    }
	    cudaMemcpy(bond_atom1_GPU, temp_int2, size_intb, cudaMemcpyHostToDevice);
	    for(mone=1;mone<=num_bonds;mone++)
	    {
	      temp_int2[mone]=bond_atom[mone][2];
	    }
            cudaMemcpy(bond_atom2_GPU, temp_int2, size_intb, cudaMemcpyHostToDevice);


	    for(mone=1;mone<=num_atoms;mone++)
	    {
	      temp_int1[mone]=neigh[mone][1];
	    }
            cudaMemcpy(neigh1_GPU, temp_int1, size_int, cudaMemcpyHostToDevice);
	    for(mone=1;mone<=num_atoms;mone++)
	    {
	      temp_int1[mone]=neigh[mone][2];
	    }
            cudaMemcpy(neigh2_GPU, temp_int1, size_int, cudaMemcpyHostToDevice);
	    for(mone=1;mone<=num_atoms;mone++)
	    { 
	      temp_int1[mone]=neigh[mone][3];
	    }
            cudaMemcpy(neigh3_GPU, temp_int1, size_int, cudaMemcpyHostToDevice);
          
            cudaMemcpy(type_odd_even_GPU, type_odd_even, size_int, cudaMemcpyHostToDevice);
	    for(mone=1;mone<=num_atoms;mone++)
	    {
	      temp_int1[mone]=neigh[mone][4];
	    }
            cudaMemcpy(neigh4_GPU, temp_int1, size_int, cudaMemcpyHostToDevice);
	    for(mone=1;mone<=num_atoms;mone++)
	    {
	      temp_int1[mone]=neigh[mone][5];
	    }
            cudaMemcpy(neigh5_GPU, temp_int1, size_int, cudaMemcpyHostToDevice);
	    for(mone=1;mone<=num_atoms;mone++)
	    {
	      temp_int1[mone]=neigh[mone][6];
	    }
            cudaMemcpy(neigh6_GPU, temp_int1, size_int, cudaMemcpyHostToDevice);
	  }
       }
	
	 Redraw (winWidth, winHeight);
	 calc_all_energy();
         sprintf(tmp,"Total Energy: %.2lf\nRadial Energy: %.2lf\n3-body Energy: %.2lf\nkinetic_energy: %.2lf\nIts: %d\n",
          energy[3]+energy[4],energy[1],energy[2],energy[4],redr); /* on screen*/
         XmTextSetString (wText2,tmp);

     }

      void initial_gpu_lists()
      {
	int c,max_temp,max_temp2,i_reduce,flag;
	double XMAX_BOND,XMIN_BOND,YMAX_BOND,YMIN_BOND,xfac,yfac,xfac2,yfac2;
	
	i_reduce=0;
	flag=0;
	while(flag!=99)
	{
	  flag=99;
	  max_nx=(-XMIN+XMAX+EPS)/(BSZX-i_reduce)/((XMAX-XMIN)/(double)(T))+1;
	  max_ny=(-YMIN+YMAX+EPS)/(BSZY)/((YMAX-YMIN)/(double)(2*N+2))+1;
	  xfac=(double)(T);
	  yfac=(double)(2*N+2);
	  xfac2=(double)(T)*sqrt(3.0);
	  yfac2=(double)(2*N+2)*sqrt(3.0);
	  XMIN_BOND=(X[1]+X[T+1])/2.0;
	  XMAX_BOND=(X[T]+X[T-1])/2.0;
	  YMAX_BOND=Y[num_atoms];
	  YMIN_BOND=Y[1];
	
	  initial_gpu_lists_atoms(i_reduce,xfac,yfac);
	  max_temp=0;
	  max_temp2=0;
	  for(c=0;c<max_nx*max_ny;c++)
	  {
	    max_temp=IMAX(max_temp,atom_id2[c]);
	    max_temp2=IMAX(max_temp2,atom_id[c]);
	  }
          if(max_temp>fac_threads)
	  {  
	    i_reduce++;
	    flag=0;
	    /*printf("max_temp=%d, fac_threads=%d !!!\n",max_temp,fac_threads);
	    nrerror("CRUCIAL BUG IN GPU ATOM LISTS");*/
	  }
	  if(max_temp2>threadsPerBlock)
	  {  
	    if(flag==99)i_reduce++;
	    flag=0;
	    /*printf("max_temp2=%d, threadsPerBlock=%d !!!\n",max_temp2,threadsPerBlock);
	    nrerror("CRUCIAL BUG IN GPU ATOM LISTS");*/
	  }
	}
	
	NMat=max_nx*max_ny*threadsPerBlock;
	blocksPerGridMat= (NMat + threadsPerBlock - 1) / threadsPerBlock;
	
	i_reduce=0;
	flag=0;
	while(flag!=99)
	{
	  flag=99;
	  initial_gpu_lists_bonds(i_reduce,xfac2,yfac2,XMIN_BOND,XMAX_BOND,YMIN_BOND,YMAX_BOND);
	  max_temp=0;
 	  for(c=0;c<max_nx_b*max_ny_b;c++)
	  {
	    max_temp=IMAX(max_temp,bond_id[c]);
	  }
          if(max_temp>threadsPerBlock)
	  {  
	    i_reduce++;
	    flag=0;
	    /*printf("max_temp=%d, threadsPerBlock=%d !!!\n",max_temp,threadsPerBlock);
	    nrerror("CRUCIAL BUG IN GPU BOND LISTS");*/
	  }
	}
		
	cudaMalloc(&atom_id_GPU, max_nx*max_ny * sizeof(int));
        cudaMalloc(&atom_id2_GPU, max_nx*max_ny * sizeof(int));
        cudaMalloc(&atom_list_GPU, max_nx*max_ny*fac_threads * sizeof(int));
        cudaMalloc(&type_odd_even_lists_GPU, max_nx*max_ny*fac_threads * sizeof(int));
        cudaMalloc(&neigh_small_grid_GPU, max_nx*max_ny*fac_threads*7 * sizeof(int));
        cudaMalloc(&neigh_small_grid_big_GPU, max_nx*max_ny*fac_threads*7 * sizeof(int));
	cudaMalloc(&neigh_lists_GPU, max_nx*max_ny*fac_threads * sizeof(int));
	cudaMalloc(&bond_id_GPU, max_nx_b*max_ny_b * sizeof(int));
        cudaMalloc(&bond_list_GPU, max_nx_b*max_ny_b*threadsPerBlock * sizeof(int));
	cudaMalloc(&num_atoms_in_bonds_GPU, max_nx_b*max_ny_b * sizeof(int));
        cudaMalloc(&bond_atom_small1_GPU, max_nx_b*max_ny_b*threadsPerBlock * sizeof(int));
        cudaMalloc(&bond_atom_small2_GPU, max_nx_b*max_ny_b*threadsPerBlock * sizeof(int));
        cudaMalloc(&atom_list_bonds_GPU, max_nx_b*max_ny_b*threadsPerBlock * sizeof(int));
       
        cudaMemcpy(atom_id_GPU, atom_id, max_nx*max_ny * sizeof(int) , cudaMemcpyHostToDevice);
        cudaMemcpy(atom_id2_GPU, atom_id2, max_nx*max_ny * sizeof(int) , cudaMemcpyHostToDevice);
        cudaMemcpy(atom_list_GPU, atom_list, max_nx*max_ny*fac_threads * sizeof(int) , cudaMemcpyHostToDevice);
        cudaMemcpy(type_odd_even_lists_GPU, type_odd_even_lists, max_nx*max_ny*fac_threads * sizeof(int) , cudaMemcpyHostToDevice);
        cudaMemcpy(neigh_small_grid_GPU, neigh_small_grid, max_nx*max_ny*fac_threads*7 * sizeof(int) , cudaMemcpyHostToDevice);
        cudaMemcpy(neigh_small_grid_big_GPU, neigh_small_grid_big, max_nx*max_ny*fac_threads*7 * sizeof(int) , cudaMemcpyHostToDevice);
        cudaMemcpy(neigh_lists_GPU, neigh_lists, max_nx*max_ny*fac_threads * sizeof(int) , cudaMemcpyHostToDevice);
        cudaMemcpy(bond_id_GPU, bond_id, max_nx_b*max_ny_b * sizeof(int) , cudaMemcpyHostToDevice);
        cudaMemcpy(bond_list_GPU, bond_list, max_nx_b*max_ny_b*threadsPerBlock * sizeof(int) , cudaMemcpyHostToDevice);
        cudaMemcpy(num_atoms_in_bonds_GPU, num_atoms_in_bonds, max_nx_b*max_ny_b * sizeof(int) , cudaMemcpyHostToDevice);
        cudaMemcpy(bond_atom_small1_GPU, bond_atom_small1, max_nx_b*max_ny_b*threadsPerBlock * sizeof(int) , cudaMemcpyHostToDevice);
        cudaMemcpy(bond_atom_small2_GPU, bond_atom_small2, max_nx_b*max_ny_b*threadsPerBlock * sizeof(int) , cudaMemcpyHostToDevice);
        cudaMemcpy(atom_list_bonds_GPU, atom_list_bonds, max_nx_b*max_ny_b*threadsPerBlock * sizeof(int) , cudaMemcpyHostToDevice);
      }

      void initial_gpu_lists_atoms(int i_reduce,double xfac,double yfac)
      {
	int mone,j,d,mone2,flag,c;

	atom_list = (int*)malloc(max_nx*max_ny*fac_threads * sizeof(int));
	type_odd_even_lists = (int*)malloc(max_nx*max_ny*fac_threads * sizeof(int));
	atom_id = (int*)malloc(max_nx*max_ny * sizeof(int));
	atom_id2 = (int*)malloc(max_nx*max_ny * sizeof(int));
	neigh_small_grid = (int*)malloc(max_nx*max_ny*fac_threads*7 * sizeof(int));
	neigh_small_grid_big = (int*)malloc(max_nx*max_ny*fac_threads*7 * sizeof(int));
	neigh_lists = (int*)malloc(max_nx*max_ny*fac_threads * sizeof(int));
	neigh_small_grid0 = (int*)malloc(max_nx*max_ny*fac_threads*7 * sizeof(int));
	
	for(mone=0;mone<max_nx*max_ny;mone++)
	{
	  atom_id[mone]=0;
	  atom_id2[mone]=0;
	}
	for(mone=0;mone<max_nx*max_ny*fac_threads;mone++)
	{
	  atom_list[mone]=0;
	}
	for(mone=0;mone<max_nx*max_ny*fac_threads*7;mone++)
	{
	  neigh_small_grid[mone]=0;
	}
	for (mone=1;mone<=num_atoms;mone++)
	{  
          block_x=(X[mone]+EPS)/(BSZX-i_reduce)/((XMAX-XMIN)/xfac);
          block_y=(-YMIN+Y[mone]+EPS)/BSZY/((YMAX-YMIN)/yfac);
          c=block_y*(max_nx)+block_x;
	  atom_list[c*fac_threads + atom_id[c]]=mone;
	  type_odd_even_lists[c*fac_threads + atom_id[c]]=type_odd_even[mone];
	  neigh_lists[mone]=c*fac_threads*7 + atom_id[c]*7;
	  neigh_small_grid[c*fac_threads*7 + atom_id[c]*7]=atom_id[c];
	  neigh_small_grid_big[c*fac_threads*7 + atom_id[c]*7]=mone;
	  atom_id[c]++;
	}
	
	for(c=0;c<max_nx*max_ny;c++)
	{
	  atom_id2[c]=atom_id[c];
	  for(mone=0;mone<atom_id[c];mone++)
	  {
	    for(j=1; j<=6; j++)
	    {
	      d=neigh[atom_list[c*fac_threads + mone]][j];
	      flag=1;
	      for(mone2=0;mone2<atom_id2[c];mone2++)
	      {
	        if(d==atom_list[c*fac_threads + mone2])
	        {
	          neigh_small_grid[c*fac_threads*7 + mone*7 + j]=mone2;
	          neigh_small_grid_big[c*fac_threads*7 + mone*7 + j]=d;
	          flag=0;
	          break;
	        }
	      }
	      if(flag==1)
	      {
	        if(d!=-1)
	        {
	          atom_list[c*fac_threads + atom_id2[c]]=d;
	          neigh_small_grid[c*fac_threads*7 + mone*7 + j]=atom_id2[c];
	          neigh_small_grid_big[c*fac_threads*7 + mone*7 + j]=d;
	          atom_id2[c]++;
	        }
	        else
	        {
	          neigh_small_grid[c*fac_threads*7+ mone*7 + j]=-1;
	          neigh_small_grid_big[c*fac_threads*7 + mone*7 + j]=-1;		  
	        }
	      }
	    }
	  }
	}		

	for(mone=0;mone<max_nx*max_ny*fac_threads*7;mone++)
	{
	  neigh_small_grid0[mone]=neigh_small_grid[mone];
	}
     }

      void initial_gpu_lists_bonds(int i_reduce,double xfac2,double yfac2,double XMIN_BOND,double XMAX_BOND,double YMIN_BOND,double YMAX_BOND)
      {
	int mone,c,atom_of_bond1,atom_of_bond2,flag1,flag2,mone2;
	double x,y;

	max_nx_b=int((-XMIN_BOND+XMAX_BOND+EPS)/(double)(BSZX-i_reduce)/((XMAX_BOND-XMIN_BOND)/xfac2))+1;
	max_ny_b=int((-YMIN_BOND+YMAX_BOND+EPS)/(double)(BSZY-1)/((YMAX_BOND-YMIN_BOND)/yfac2))+1;
	
	bond_list = (int*)malloc(max_nx_b*max_ny_b*threadsPerBlock * sizeof(int));
	bond_id = (int*)malloc(max_nx_b*max_ny_b * sizeof(int));
	num_atoms_in_bonds = (int*)malloc(max_nx_b*max_ny_b * sizeof(int));
	atom_list_bonds_small = (int*)malloc(max_nx_b*max_ny_b*threadsPerBlock * sizeof(int));
	atom_list_bonds = (int*)malloc(max_nx_b*max_ny_b*threadsPerBlock * sizeof(int));
	bond_atom_small1 = (int*)malloc(max_nx_b*max_ny_b*threadsPerBlock * sizeof(int));
	bond_atom_small2 = (int*)malloc(max_nx_b*max_ny_b*threadsPerBlock * sizeof(int));
	bond_atom0_small1 = (int*)malloc(max_nx_b*max_ny_b*threadsPerBlock * sizeof(int));
	bond_atom0_small2 = (int*)malloc(max_nx_b*max_ny_b*threadsPerBlock * sizeof(int));
	
	NMat_b=max_nx_b*max_ny_b*threadsPerBlock;
	blocksPerGridMat_b= (NMat_b + threadsPerBlock - 1) / threadsPerBlock;
	
	for(mone=0;mone<max_nx_b*max_ny_b;mone++)
	{
	  bond_id[mone]=0;
	  num_atoms_in_bonds[mone]=0;
	}
	for(mone=0;mone<max_nx_b*max_ny_b*threadsPerBlock;mone++)
	{
	  bond_list[mone]=0;
	  atom_list_bonds_small[mone]=0;
	  atom_list_bonds[mone]=0;
	  bond_atom_small1[mone]=0;
	  bond_atom_small2[mone]=0;	  
	}

	for (mone=1;mone<=num_bonds;mone++)
	{  
          x=(X[bond_atom[mone][1]]+X[bond_atom[mone][2]])/2.0;
          y=(Y[bond_atom[mone][1]]+Y[bond_atom[mone][2]])/2.0;
	  block_x=IMIN(int((-XMIN_BOND+x+(0.7-0.5)*A_lat)/(double)(BSZX-i_reduce)/((XMAX_BOND-XMIN_BOND)/xfac2)),max_nx_b-1);
          block_y=IMIN(int((-YMIN_BOND+y+(0.4-0.5)*A_lat)/(double)(BSZY-1)/((YMAX_BOND-YMIN_BOND)/yfac2)),max_ny_b-1);
	  c=block_y*(max_nx_b)+block_x;
	  bond_list[c*threadsPerBlock + bond_id[c]]=mone;
	  bond_id[c]++;
	}
	
	for(c=0;c<max_nx_b*max_ny_b;c++)
	{
	  for(mone=0;mone<bond_id[c];mone++)
	  {
	    atom_of_bond1=bond_atom[bond_list[c*threadsPerBlock + mone]][1];
	    atom_of_bond2=bond_atom[bond_list[c*threadsPerBlock + mone]][2];
	    flag1=1;
	    flag2=1;
	    for(mone2=0;mone2<num_atoms_in_bonds[c];mone2++)
	    {
	      if(atom_of_bond1==atom_list_bonds[c*threadsPerBlock + mone2])
	      {
	        bond_atom_small1[c*threadsPerBlock+ mone]=atom_list_bonds_small[c*threadsPerBlock + mone2];
	        flag1=0;
	      } 	      
	      if(flag1==0)break;
	    }
	    for(mone2=0;mone2<num_atoms_in_bonds[c];mone2++)
	    { 
	      if(atom_of_bond2==atom_list_bonds[c*threadsPerBlock + mone2])
	      {
	        bond_atom_small2[c*threadsPerBlock+ mone]=atom_list_bonds_small[c*threadsPerBlock + mone2];
	        flag2=0;
	      }
	      if(flag2==0)break;
	    }
	    if(flag1==1)
	    {
	      atom_list_bonds_small[c*threadsPerBlock+num_atoms_in_bonds[c]]=num_atoms_in_bonds[c];
	      atom_list_bonds[c*threadsPerBlock+num_atoms_in_bonds[c]]=atom_of_bond1;
	      bond_atom_small1[c*threadsPerBlock+ mone]=atom_list_bonds_small[c*threadsPerBlock+num_atoms_in_bonds[c]]; 	 
	      num_atoms_in_bonds[c]++;
	    }
	    if(flag2==1)
	    {
	      atom_list_bonds_small[c*threadsPerBlock+num_atoms_in_bonds[c]]=num_atoms_in_bonds[c];
	      atom_list_bonds[c*threadsPerBlock+num_atoms_in_bonds[c]]=atom_of_bond2;
	      bond_atom_small2[c*threadsPerBlock+ mone]=atom_list_bonds_small[c*threadsPerBlock+num_atoms_in_bonds[c]]; 	
	      num_atoms_in_bonds[c]++;
	    }
	  }
	}
	
	for(mone=0;mone<max_nx_b*max_ny_b*threadsPerBlock;mone++)
	{
	  bond_atom0_small1[mone]=bond_atom_small1[mone];
	  bond_atom0_small2[mone]=bond_atom_small2[mone];	  
	}		
      }


      void time_step()
      {
       int mone;
       
       if(redr==0) clock_gettime(CLOCK_MONOTONIC, &start);
       clock_gettime(CLOCK_MONOTONIC, &start2);
       if(cpu_gpu==0)
       {
	 CALC_FORCE();
         CALC_U_AND_V();
	 time_md+=DT;
             
         DY_PLOT=sqrt(3.0)*A_lat/2.0*((double)(zoom_size)/1.515152-1.0);
         DX_PLOT=(double)(zoom_size)*A_lat;
         Y_CV=(double)(zoom_y-1)/100.0 *(double)(2*N+2)/((double)(zoom_size)/1.515152);
         X_CV=(double)(zoom_x-1)/100.0 *(double)(T)/((double)(zoom_size)+1.0);
         if (redr%update==0)
         {
	   for(mone=1; mone<=num_atoms; mone++)
           {
            Y_PLOT[mone]=(int)(((Y[mone]-YMIN+A_lat/2)/(DY_PLOT)-Y_CV)*(double)(winHeight)*0.95);
            X_PLOT[mone]=(int)(((X[mone]-XMIN+A_lat/2)/(DX_PLOT)-X_CV)*(double)(winWidth)*0.95);
           }
	 }  
         if (redr%100==0)
         {
           calc_all_energy();
         }
         energy[4]=0.0;
         for(mone=1; mone<=num_atoms; mone++)
         {
           energy[4]+=0.5*(VX[mone]*VX[mone]+VY[mone]*VY[mone]);
         }         
       }
       else
       { 
	 ZEROING_A_GPU<<<blocksPerGrid, threadsPerBlock>>>(AX_GPU, AY_GPU, Na);
      
	 CALC_FORCE_GPU<<<blocksPerGridMat_b, threadsPerBlock>>>(AX_GPU,AY_GPU,VX_GPU,VY_GPU,X_GPU,Y_GPU,bond_atom_small1_GPU,bond_atom_small2_GPU,ETA,A_lat0_GPU,
	                                                      bond_id_GPU,num_atoms_in_bonds_GPU,atom_list_bonds_GPU,bond_list_GPU,md_cg_mode);
	 if(k_teta>1.0e-5)
	 {
	   CALC_3_BODY_FORCE_GPU<<<blocksPerGridMat, threadsPerBlock>>>(AX_GPU,AY_GPU,X_GPU,Y_GPU,k_teta,
			       atom_list_GPU,atom_id_GPU,atom_id2_GPU,neigh_small_grid_GPU,type_odd_even_lists_GPU);
	 }
	 CALC_U_AND_V_GPU<<<blocksPerGrid, threadsPerBlock>>>(AX_GPU, AY_GPU, VX_GPU,VY_GPU,X_GPU, Y_GPU, n_safa_GPU, Na, DT);	 
         time_md+=DT;
        
         if (redr%100==0)
         {
           calc_all_energy_GPU();
         }
	 if (redr%update==0)
         {
	   DY_PLOT=sqrt(3.0)*A_lat/2.0*((double)(zoom_size)/1.515152-1.0);
           DX_PLOT=(double)(zoom_size)*A_lat;
           Y_CV=(double)(zoom_y-1)/100.0 *(double)(2*N+2)/((double)(zoom_size)/1.515152);    
           X_CV=(double)(zoom_x-1)/100.0 *(double)(T)/((double)(zoom_size)+1.0);    
	   CALC_UXY_PLOT<<<blocksPerGrid, threadsPerBlock>>>(X_PLOT_GPU,Y_PLOT_GPU,X_GPU,Y_GPU,DX_PLOT,DY_PLOT,X_CV,Y_CV,YMIN,XMIN,A_lat,winHeight,winWidth,Na);
	   cudaMemcpy(X_PLOT, X_PLOT_GPU, size_int, cudaMemcpyDeviceToHost);
           cudaMemcpy(Y_PLOT, Y_PLOT_GPU, size_int, cudaMemcpyDeviceToHost);

           calc_all_energy_GPU();
         }

         energy[4]=0.0;	 
         cudaMemcpy(energy_GPU, energy,size4, cudaMemcpyHostToDevice);
	 CALC_KINETIC_ENERGY<<<blocksPerGrid, threadsPerBlock>>>(VX_GPU,VY_GPU,energy_GPU,Na);
	 cudaMemcpy(energy,energy_GPU,size4, cudaMemcpyDeviceToHost);    
       }
      
       sprintf(tmp,"Total Energy: %.2lf\nRadial Energy: %.2lf\n3-body Energy: %.2lf\nkinetic_energy: %.2lf\nIts: %d\n",
          energy[3]+energy[4],energy[1],energy[2],energy[4],redr); /* on screen*/
  	XmTextSetString (wText2,tmp);
        	  
	 clock_gettime(CLOCK_MONOTONIC, &finish2);
	 elapsed2 += (finish2.tv_sec-start2.tv_sec);
         elapsed2 += (finish2.tv_nsec-start2.tv_nsec)*1.0e-9;
      }
                 
      void stretch()
      {
      int i,mone,update_stretch,ind;
      double shipua,x1new,x2new,y1new,y2new,r1new,r2new,r1,Xmid,Ymid,x_par_old;
      
      update_stretch=50;
      time_md=0.0;
      for(mone=1; mone<=num_atoms; mone++)
       {
         VX[mone]=0.0;
         VY[mone]=0.0;
       }
      
      x_par_old=r_ball;
      for (ind=1; ind<=update_stretch; ind++)
      {
        x_par=r_ball-(double)(ind)/((double)(update_stretch))*r_ball*0.3;
    	Xmid=A_lat*double(T)/2.0+1e-5;
    	Ymid=0.0+1e-5;	

	for(i=1; i<=num_atoms; i++)
    	{    	    
	    r1=sqrt((X[i]-Xmid)*(X[i]-Xmid)+(Y[i]-Ymid)*(Y[i]-Ymid));
    	    if(fabs(r1-x_par_old)<=1e-10)
    	    {
    	      shipua=(Y[i]-Ymid)/(X[i]-Xmid);
    	      x1new=x_par/sqrt(1.0+shipua*shipua)+Xmid;
    	      x2new=-x_par/sqrt(1.0+shipua*shipua)+Xmid;
    	      y1new=shipua*(x1new-Xmid)+Ymid;
    	      y2new=shipua*(x2new-Xmid)+Ymid;
    	      r1new=sqrt((x1new-X[i])*(x1new-X[i])+(y1new-Y[i])*(y1new-Y[i]));
    	      r2new=sqrt((x2new-X[i])*(x2new-X[i])+(y2new-Y[i])*(y2new-Y[i]));
	      if(r1new<r2new)
    	      {
    		X[i]=x1new;
    		Y[i]=y1new;
    	      }
    	      else
    	      {
    		X[i]=x2new;
    		Y[i]=y2new;	      
    	      }
	       n_safa[i]=1;
    	    }
    	}

        DY_PLOT=sqrt(3.0)*A_lat/2.0*((double)(zoom_size)/1.515152-1.0);
        DX_PLOT=(double)(zoom_size)*A_lat;
        Y_CV=(double)(zoom_y-1)/100.0 *(double)(2*N+2)/((double)(zoom_size)/1.515152);    
        X_CV=(double)(zoom_x-1)/100.0 *(double)(T)/((double)(zoom_size)+1.0);     
        if(cpu_gpu==0)
        {
          for(mone=1; mone<=num_atoms; mone++)
          {
            Y_PLOT[mone]=(int)(((Y[mone]-YMIN+A_lat/2)/(DY_PLOT)-Y_CV)*(double)(winHeight)*0.95);
            X_PLOT[mone]=(int)(((X[mone]-XMIN+A_lat/2)/(DX_PLOT)-X_CV)*(double)(winWidth)*0.95);
          }
        }
        else
        {
           cudaMemcpy(X_GPU, X, size, cudaMemcpyHostToDevice);
           cudaMemcpy(Y_GPU, Y, size, cudaMemcpyHostToDevice);
	   cudaMemcpy(VX_GPU, VX, size, cudaMemcpyHostToDevice);
           cudaMemcpy(VY_GPU, VY, size, cudaMemcpyHostToDevice);
	   CALC_UXY_PLOT<<<blocksPerGrid, threadsPerBlock>>>(X_PLOT_GPU,Y_PLOT_GPU,X_GPU,Y_GPU,DX_PLOT,DY_PLOT,X_CV,Y_CV,YMIN,XMIN,A_lat,winHeight,winWidth,Na);
	   cudaMemcpy(X_PLOT, X_PLOT_GPU, size_int, cudaMemcpyDeviceToHost);
           cudaMemcpy(Y_PLOT, Y_PLOT_GPU, size_int, cudaMemcpyDeviceToHost);
	   cudaMemcpy(n_safa_GPU, n_safa, size_int, cudaMemcpyHostToDevice);
        }
	 	      	 
	 calc_all_energy();
         sprintf(tmp,"Total Energy: %.2lf\nRadial Energy: %.2lf\n3-body Energy: %.2lf\nkinetic_energy: %.2lf\nIts: %d\n",
          energy[3]+energy[4],energy[1],energy[2],energy[4],redr);  /*on screen*/
         XmTextSetString (wText2,tmp);
	 
	 Redraw (winWidth, winHeight);
         x_par_old=x_par;
	 }  
      }

void conjugate_gradient()
{
        int i,its;
        double ftolf;
  
        ftolf=TOLF; 
     
        if(cpu_gpu==0)
	{
          if(running==1)
	  {
	    clock_gettime(CLOCK_MONOTONIC, &start);
            its=frprmn(ftolf);
            clock_gettime(CLOCK_MONOTONIC, &finish);
            elapsed = (finish.tv_sec-start.tv_sec);
            elapsed += (finish.tv_nsec-start.tv_nsec)*1.0e-9;
            printf("conjugate-gradient time = %lf sec\n",elapsed);
            printf("Using conjugate-gradient algorithm, its=%d, energy=%lf\n",its,energy[3]);
	    YMAX=Y[(2*N+2)*T];
            XMIN=X[1];
            XMAX=DMAX(X[(2*N+2)*T],X[(2*N+1)*T]);   
            DY_PLOT=sqrt(3.0)*A_lat/2.0*((double)(zoom_size)/1.515152-1.0);
            DX_PLOT=(double)(zoom_size)*A_lat;
            Y_CV=(double)(zoom_y-1)/100.0 *(double)(2*N+2)/((double)(zoom_size)/1.515152);	
            X_CV=(double)(zoom_x-1)/100.0 *(double)(T)/((double)(zoom_size)+1.0);  
            
            for(i=1; i<=num_atoms; i++)
            {
              Y_PLOT[i]=(int)(((Y[i]-YMIN+A_lat/2)/(DY_PLOT)-Y_CV)*(double)(winHeight)*0.95);
              X_PLOT[i]=(int)(((X[i]-XMIN+A_lat/2)/(DX_PLOT)-X_CV)*(double)(winWidth)*0.95);
            }
            Redraw (winWidth, winHeight);
	  }
	}
        if(cpu_gpu==1)
	  {
          
	  if(running==1)
	  {
	    g = (double*)malloc(size2);
	    h = (double*)malloc(size2);
	    xi = (double*)malloc(size2);
	    p = (double*)malloc(size2);
	    cudaMalloc(&g_GPU, size2);
	    cudaMalloc(&h_GPU, size2);
	    cudaMalloc(&xi_GPU, size2);
	    cudaMalloc(&p_GPU, size2);
	    pcom = (double*)malloc(size2);
	    xicom = (double*)malloc(size2);
            cudaMalloc(&pcom_GPU, size2);
	    cudaMalloc(&xicom_GPU, size2);
	    xt = (double*)malloc(size2);
            cudaMalloc(&xt_GPU, size2);
	    cudaMalloc(&gg_dgg_GPU, 2* sizeof(double));

	  
	    clock_gettime(CLOCK_MONOTONIC, &start);
            its=frprmn_GPU(ftolf);
            clock_gettime(CLOCK_MONOTONIC, &finish);
            elapsed = (finish.tv_sec-start.tv_sec);
            elapsed += (finish.tv_nsec-start.tv_nsec)*1.0e-9;
            printf("conjugate-gradient time = %lf sec\n",elapsed);
	    printf("Using conjugate-gradient algorithm, its=%d, energy=%lf\n",its,energy[3]);
	    DY_PLOT=sqrt(3.0)*A_lat/2.0*((double)(zoom_size)/1.515152-1.0);
            DX_PLOT=(double)(zoom_size)*A_lat;
            Y_CV=(double)(zoom_y-1)/100.0 *(double)(2*N+2)/((double)(zoom_size)/1.515152);    
            X_CV=(double)(zoom_x-1)/100.0 *(double)(T)/((double)(zoom_size)+1.0);    
	    CALC_UXY_PLOT<<<blocksPerGrid, threadsPerBlock>>>(X_PLOT_GPU,Y_PLOT_GPU,X_GPU,Y_GPU,DX_PLOT,DY_PLOT,X_CV,Y_CV,YMIN,XMIN,A_lat,winHeight,winWidth,Na);
	    cudaMemcpy(X_PLOT, X_PLOT_GPU, size_int, cudaMemcpyDeviceToHost);
            cudaMemcpy(Y_PLOT, Y_PLOT_GPU, size_int, cudaMemcpyDeviceToHost);
            Redraw (winWidth, winHeight);
	    cudaMemcpy(X,X_GPU, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(Y,Y_GPU, size, cudaMemcpyDeviceToHost);
	    cudaFree(g_GPU);
	    cudaFree(h_GPU);
	    cudaFree(xi_GPU);
	    cudaFree(p_GPU);
	    free(g);
	    free(h);
	    free(xi);
	    free(p);
	    free(xt);
	    cudaFree(pcom_GPU);
	    cudaFree(xt_GPU);
	    cudaFree(xicom_GPU);
	    cudaFree(gg_dgg_GPU);
	    free(pcom);
	    free(xicom);
	  }   
	}
}

int frprmn(double ftol)
{
	void linmin(double p[], double xi[]),dlinmin(double p[], double xi[]);
	int j,its,i,method_nr;
	double gg,gam,fp,dgg;
	double *g,*h,*xi,*p;

        method_nr=0;

	g=dvector(1,2*num_atoms);
	h=dvector(1,2*num_atoms);
	xi=dvector(1,2*num_atoms);
	p=dvector(1,2*num_atoms);
	calc_all_energy();
	fp=energy[3];
	CALC_FORCE();
	for(i=1;i<=num_atoms;i++)
	{
	  if(n_safa[i]==0)
 	  {
	    p[2*i-1]=X[i];
	    p[2*i]=Y[i];
	    xi[2*i-1]=-AX[i];
	    xi[2*i]=-AY[i];
	  }
	}
	for (j=1;j<=2*num_atoms;j++) {
		g[j] = -xi[j];
		xi[j]=h[j]=g[j];
	}
	for (its=1;its<=ITMAX;its++) {
		if(method_nr==0)
		{
		  linmin(p,xi);
		}
                if(method_nr==1)
		{
		  dlinmin(p,xi);
		}
		if (2.0*fabs(fret-fp) <= ftol*(fabs(fret)+fabs(fp)+EPS)) {
			free_dvector(xi,1,2*num_atoms);
			free_dvector(h,1,2*num_atoms);
			free_dvector(g,1,2*num_atoms);
			return its;
		}     		       
		calc_all_energy();
		fp=energy[3];
		if ((its%update==0)&&(its!=0)) 
                {  
		  DY_PLOT=sqrt(3.0)*A_lat/2.0*((double)(zoom_size)/1.515152-1.0);
     		  DX_PLOT=(double)(zoom_size)*A_lat;
     		  Y_CV=(double)(zoom_y-1)/100.0 *(double)(2*N+2)/((double)(zoom_size)/1.515152);    
     		  X_CV=(double)(zoom_x-1)/100.0 *(double)(T)/((double)(zoom_size)+1.0);  
     		     
     		  for(i=1; i<=num_atoms; i++)
     		  {
     		    Y_PLOT[i]=(int)(((Y[i]-YMIN+A_lat/2)/(DY_PLOT)-Y_CV)*(double)(winHeight)*0.95);
     		    X_PLOT[i]=(int)(((X[i]-XMIN+A_lat/2)/(DX_PLOT)-X_CV)*(double)(winWidth)*0.95);
     		  }
		  Redraw (winWidth, winHeight);
		  sprintf(tmp,"Total Energy: %.2lf\nRadial Energy: %.2lf\n3-body Energy: %.2lf\nkinetic_energy: %.2lf\nIts: %d\n",
                      energy[3]+energy[4],energy[1],energy[2],energy[4],its); /* on screen*/
                  XmTextSetString (wText2,tmp);
		}
		CALC_FORCE();
	        for(i=1;i<=num_atoms;i++)
	        {
	          xi[2*i-1]=-AX[i];
	          xi[2*i]=-AY[i];
	        }
		dgg=gg=0.0;
		for (j=1;j<=num_atoms;j++) {
			if(n_safa[j]==0)
			{
			  gg += g[2*j-1]*g[2*j-1];
			  dgg += (xi[2*j-1]+g[2*j-1])*xi[2*j-1];
			  gg += g[2*j]*g[2*j];
			  dgg += (xi[2*j]+g[2*j])*xi[2*j];
			}
		}
		if (gg == 0.0) {
			free_dvector(xi,1,2*num_atoms);
			free_dvector(h,1,2*num_atoms);
			free_dvector(g,1,2*num_atoms);
			return its;
		}
		gam=dgg/gg;
		for (j=1;j<=num_atoms;j++) {
			if(n_safa[j]==0)
			{
			  g[2*j-1] = -xi[2*j-1];
			  xi[2*j-1]=h[2*j-1]=g[2*j-1]+gam*h[2*j-1];
			  g[2*j] = -xi[2*j];
			  xi[2*j]=h[2*j]=g[2*j]+gam*h[2*j];
			}
		}
	}
	nrerror("Too many iterations in frprmn");
  return its;
}

int ncom;

void linmin(double p[], double xi[])
{
	double brent(double ax, double bx, double cx,
		double (*f)(double), double tol, double *xmin);
	double f1dim(double x);
	void mnbrak(double *ax, double *bx, double *cx, double *fa, double *fb,
		double *fc, double (*func)(double));
	int j,i;
	double xx,xmin,fx,fb,fa,bx,ax;

	ncom=2*num_atoms;
	pcom=dvector(1,num_atoms*2);
	xicom=dvector(1,num_atoms*2);
	for (j=1;j<=num_atoms*2;j++) {
		pcom[j]=p[j];
		xicom[j]=xi[j];
	}
	ax=0.0;
	xx=1.0;
	mnbrak(&ax,&xx,&bx,&fa,&fx,&fb,f1dim);
	fret=brent(ax,xx,bx,f1dim,TOL,&xmin);
	for (j=1;j<=2*num_atoms;j++) {
		xi[j] *= xmin;
		p[j] += xi[j];
	}
	for(i=1;i<=num_atoms;i++)
	{
	  if(n_safa[i]==0)
	  {
	    X[i]=p[2*i-1];
	    Y[i]=p[2*i];
	  }
	}	
	free_dvector(xicom,1,2*num_atoms);
	free_dvector(pcom,1,2*num_atoms);
}

void dlinmin(double p[], double xi[])
{
	double dbrent(double ax, double bx, double cx,
		double (*f)(double), double (*df)(double), double tol, double *xmin);
	double f1dim(double x);
	double df1dim(double x);
	void mnbrak(double *ax, double *bx, double *cx, double *fa, double *fb,
		double *fc, double (*func)(double));
	int j,i;
	double xx,xmin,fx,fb,fa,bx,ax;

	ncom=2*num_atoms;
	pcom=dvector(1,2*num_atoms);
	xicom=dvector(1,2*num_atoms);
	for (j=1;j<=num_atoms*2;j++) {
		pcom[j]=p[j];
		xicom[j]=xi[j];
	}
	ax=0.0;
	xx=1.0;
	mnbrak(&ax,&xx,&bx,&fa,&fx,&fb,f1dim);
	fret=dbrent(ax,xx,bx,f1dim,df1dim,TOL,&xmin);
	for (j=1;j<=2*num_atoms;j++) {
		xi[j] *= xmin;
		p[j] += xi[j];
	}
	for(i=1;i<=num_atoms;i++)
	{
	  if(n_safa[i]==0)
	  {
	    X[i]=p[2*i-1];
	    Y[i]=p[2*i];
	  }
	}
	free_dvector(xicom,1,2*num_atoms);
	free_dvector(pcom,1,2*num_atoms);
}


double brent(double ax, double bx, double cx, double (*f)(double), double tol,
	double *xmin)
{
	int iter;
	double a,b,d,etemp,fu,fv,fw,fx,p,q,r,tol1,tol2,u,v,w,x,xm;
	double e=0.0;

	a=(ax < cx ? ax : cx);
	b=(ax > cx ? ax : cx);
	x=w=v=bx;
	fw=fv=fx=(*f)(x);
	for (iter=1;iter<=ITMAX;iter++) {
		xm=0.5*(a+b);
		tol2=2.0*(tol1=tol*fabs(x)+ZEPS);
		if (fabs(x-xm) <= (tol2-0.5*(b-a))) {
			*xmin=x;
			return fx;
		}
		if (fabs(e) > tol1) {
			r=(x-w)*(fx-fv);
			q=(x-v)*(fx-fw);
			p=(x-v)*q-(x-w)*r;
			q=2.0*(q-r);
			if (q > 0.0) p = -p;
			q=fabs(q);
			etemp=e;
			e=d;
			if (fabs(p) >= fabs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x))
				d=CGOLD*(e=(x >= xm ? a-x : b-x));
			else {
				d=p/q;
				u=x+d;
				if (u-a < tol2 || b-u < tol2)
					d=SIGN(tol1,xm-x);
			}
		} else {
			d=CGOLD*(e=(x >= xm ? a-x : b-x));
		}
		u=(fabs(d) >= tol1 ? x+d : x+SIGN(tol1,d));
		fu=(*f)(u);
		if (fu <= fx) {
			if (u >= x) a=x; else b=x;
			SHFT(v,w,x,u)
			SHFT(fv,fw,fx,fu)
		} else {
			if (u < x) a=u; else b=u;
			if (fu <= fw || w == x) {
				v=w;
				w=u;
				fv=fw;
				fw=fu;
			} else if (fu <= fv || v == x || v == w) {
				v=u;
				fv=fu;
			}
		}
	}
	nrerror("Too many iterations in brent");
	*xmin=x;
	return fx;
}

double dbrent(double ax, double bx, double cx, double (*f)(double),
	double (*df)(double), double tol, double *xmin)
{
	int iter,ok1,ok2;
	double a,b,d,d1,d2,du,dv,dw,dx,e=0.0;
	double fu,fv,fw,fx,olde,tol1,tol2,u,u1,u2,v,w,x,xm;

	a=(ax < cx ? ax : cx);
	b=(ax > cx ? ax : cx);
	x=w=v=bx;
	fw=fv=fx=(*f)(x);
	dw=dv=dx=(*df)(x);
	for (iter=1;iter<=ITMAX;iter++) {
		xm=0.5*(a+b);
		tol1=tol*fabs(x)+ZEPS;
		tol2=2.0*tol1;
		if (fabs(x-xm) <= (tol2-0.5*(b-a))) {
			*xmin=x;
			return fx;
		}
		if (fabs(e) > tol1) {
			d1=2.0*(b-a);
			d2=d1;
			if (dw != dx) d1=(w-x)*dx/(dx-dw);
			if (dv != dx) d2=(v-x)*dx/(dx-dv);
			u1=x+d1;
			u2=x+d2;
			ok1 = (a-u1)*(u1-b) > 0.0 && dx*d1 <= 0.0;
			ok2 = (a-u2)*(u2-b) > 0.0 && dx*d2 <= 0.0;
			olde=e;
			e=d;
			if (ok1 || ok2) {
				if (ok1 && ok2)
					d=(fabs(d1) < fabs(d2) ? d1 : d2);
				else if (ok1)
					d=d1;
				else
					d=d2;
				if (fabs(d) <= fabs(0.5*olde)) {
					u=x+d;
					if (u-a < tol2 || b-u < tol2)
						d=SIGN(tol1,xm-x);
				} else {
					d=0.5*(e=(dx >= 0.0 ? a-x : b-x));
				}
			} else {
				d=0.5*(e=(dx >= 0.0 ? a-x : b-x));
			}
		} else {
			d=0.5*(e=(dx >= 0.0 ? a-x : b-x));
		}
		if (fabs(d) >= tol1) {
			u=x+d;
			fu=(*f)(u);
		} else {
			u=x+SIGN(tol1,d);
			fu=(*f)(u);
			if (fu > fx) {
				*xmin=x;
				return fx;
			}
		}
		du=(*df)(u);
		if (fu <= fx) {
			if (u >= x) a=x; else b=x;
			MOV3(v,fv,dv, w,fw,dw)
			MOV3(w,fw,dw, x,fx,dx)
			MOV3(x,fx,dx, u,fu,du)
		} else {
			if (u < x) a=u; else b=u;
			if (fu <= fw || w == x) {
				MOV3(v,fv,dv, w,fw,dw)
				MOV3(w,fw,dw, u,fu,du)
			} else if (fu < fv || v == x || v == w) {
				MOV3(v,fv,dv, u,fu,du)
			}
		}
	}
	nrerror("Too many iterations in routine dbrent");
	return 0.0;
}

extern int ncom;
extern double *pcom,*xicom;

double f1dim(double x)
{
	int j,i;
	double f,*xt;

	xt=dvector(1,ncom);
	for (j=1;j<=ncom/2;j++)
	{  
	  if(n_safa[j]==0)
	  {
	    xt[2*j-1]=pcom[2*j-1]+x*xicom[2*j-1];
	    xt[2*j]=pcom[2*j]+x*xicom[2*j];
	  }
	}
	for(i=1;i<=num_atoms;i++)
	{
	  if(n_safa[i]==0)
	  {
	    X[i]=xt[2*i-1];
	    Y[i]=xt[2*i];
	  }
	}
	calc_all_energy();
	f=energy[3];
	
	for(i=1;i<=num_atoms;i++)
	{
	  if(n_safa[i]==0)
	  {
	    X[i]=pcom[2*i-1];
	    Y[i]=pcom[2*i];
	  }
	}
	
	free_dvector(xt,1,ncom);
	return f;
}

double df1dim(double x)
{
	int j,i;
	double df1=0.0;
	double *xt,*df;

	xt=dvector(1,ncom);
	df=dvector(1,ncom);
	for (j=1;j<=ncom/2;j++)
	{
	  if(n_safa[j]==0)
	  {
	    xt[2*j-1]=pcom[2*j-1]+x*xicom[2*j-1];
	    xt[2*j]=pcom[2*j]+x*xicom[2*j];
	  }
	}
	for(i=1;i<=num_atoms;i++)
	{
	  if(n_safa[i]==0)
	  {
	    X[i]=xt[2*i-1];
	    Y[i]=xt[2*i];
	  }
	}
	CALC_FORCE();
	for(i=1;i<=num_atoms;i++)
	{
	  if(n_safa[i]==0)
 	  {
	    df[2*i-1]=-AX[i];
	    df[2*i]=-AY[i];
	  }
	}
	for (j=1;j<=ncom;j++) df1 += df[j]*xicom[j];
	for(i=1;i<=num_atoms;i++)
	{
	  if(n_safa[i]==0)
	  {
	    X[i]=pcom[2*i-1];
	    Y[i]=pcom[2*i];
	  }
	}
	free_dvector(df,1,ncom);
	free_dvector(xt,1,ncom);
	return df1;
}

void mnbrak(double *ax, double *bx, double *cx, double *fa, double *fb, double *fc,
	double (*func)(double))
{
	double ulim,u,r,q,fu,dum;


	*fa=(*func)(*ax);
	*fb=(*func)(*bx);
	if (*fb > *fa) {
		SHFT(dum,*ax,*bx,dum)
		SHFT(dum,*fb,*fa,dum)
	}
	*cx=(*bx)+GOLD*(*bx-*ax);
	*fc=(*func)(*cx);
	while (*fb > *fc) {
		r=(*bx-*ax)*(*fb-*fc);
		q=(*bx-*cx)*(*fb-*fa);
		u=(*bx)-((*bx-*cx)*q-(*bx-*ax)*r)/(2.0*SIGN(FMAX(fabs(q-r),1.0e-20),q-r));
		ulim=(*bx)+GLIMIT*(*cx-*bx);
		if ((*bx-u)*(u-*cx) > 0.0) {
			fu=(*func)(u);
			if (fu < *fc) {
				*ax=(*bx);
				*bx=u;
				*fa=(*fb);
				*fb=fu;
				return;
			} else if (fu > *fb) {
				*cx=u;
				*fc=fu;
				return;
			}
			u=(*cx)+GOLD*(*cx-*bx);
			fu=(*func)(u);
		} else if ((*cx-u)*(u-ulim) > 0.0) {
			fu=(*func)(u);
			if (fu < *fc) {
				SHFT(*bx,*cx,u,*cx+GOLD*(*cx-*bx))
				SHFT(*fb,*fc,fu,(*func)(u))
			}
		} else if ((u-ulim)*(ulim-*cx) >= 0.0) {
			u=ulim;
			fu=(*func)(u);
		} else {
			u=(*cx)+GOLD*(*cx-*bx);
			fu=(*func)(u);
		}
		SHFT(*ax,*bx,*cx,u)
		SHFT(*fa,*fb,*fc,fu)
	}
}

int frprmn_GPU(double ftol)
{
	void linmin_GPU();
	int its;
	double gam,fp;
			
	calc_all_energy_GPU();
	fp=energy[3];
	ZEROING_A_GPU<<<blocksPerGrid, threadsPerBlock>>>(AX_GPU, AY_GPU, Na);
      
	if(shared==1)
	{
	  CALC_FORCE_GPU<<<blocksPerGridMat_b, threadsPerBlock>>>(AX_GPU,AY_GPU,VX_GPU,VY_GPU,X_GPU,Y_GPU,bond_atom_small1_GPU,bond_atom_small2_GPU,ETA,A_lat0_GPU,
	                                                      bond_id_GPU,num_atoms_in_bonds_GPU,atom_list_bonds_GPU,bond_list_GPU,md_cg_mode);
	  if(k_teta>1.0e-5)
	  {
	     CALC_3_BODY_FORCE_GPU<<<blocksPerGridMat, threadsPerBlock>>>(AX_GPU,AY_GPU,X_GPU,Y_GPU,k_teta,
			       atom_list_GPU,atom_id_GPU,atom_id2_GPU,neigh_small_grid_GPU,type_odd_even_lists_GPU);
	  }
	}
	if(shared==0)
	{
	  CALC_FORCE_no_shared_GPU<<<blocksPerGrid_b, threadsPerBlock>>>(AX_GPU,AY_GPU,VX_GPU,VY_GPU,X_GPU,Y_GPU,Nb,bond_atom1_GPU,bond_atom2_GPU,ETA,A_lat0_GPU,md_cg_mode);
	  if(k_teta>1.0e-5)
	  {
	    CALC_3_BODY_FORCE_no_shared_GPU<<<blocksPerGrid, threadsPerBlock>>>(AX_GPU,AY_GPU,X_GPU,Y_GPU,Na,k_teta,type_odd_even_GPU,
			       neigh1_GPU,neigh2_GPU,neigh3_GPU,neigh4_GPU,neigh5_GPU,neigh6_GPU);
	  }
	}
	INITIALIZING_P_XI_GPU<<<blocksPerGrid, threadsPerBlock>>>(AX_GPU, AY_GPU, X_GPU, Y_GPU, n_safa_GPU, p_GPU,xi_GPU, Na);
	
	blocksPerGrid2 = (2*Na + threadsPerBlock - 1) / threadsPerBlock;
	
	INITIALIZING_G_H_XI_GPU<<<blocksPerGrid2, threadsPerBlock>>>(xi_GPU, g_GPU, h_GPU, 2*Na);
	
	for (its=1;its<=ITMAX;its++) {
		
		linmin_GPU();
		
		if (2.0*fabs(fret-fp) <= ftol*(fabs(fret)+fabs(fp)+EPS)) {
			return its;
		}
		calc_all_energy_GPU();
		fp=energy[3];
		if ((its%update==0)&&(its!=0)) 
                {
		  DY_PLOT=sqrt(3.0)*A_lat/2.0*((double)(zoom_size)/1.515152-1.0);
           	  DX_PLOT=(double)(zoom_size)*A_lat;
           	  Y_CV=(double)(zoom_y-1)/100.0 *(double)(2*N+2)/((double)(zoom_size)/1.515152);    
           	  X_CV=(double)(zoom_x-1)/100.0 *(double)(T)/((double)(zoom_size)+1.0);    
           	  CALC_UXY_PLOT<<<blocksPerGrid, threadsPerBlock>>>(X_PLOT_GPU,Y_PLOT_GPU,X_GPU,Y_GPU,DX_PLOT,DY_PLOT,X_CV,Y_CV,YMIN,XMIN,A_lat,winHeight,winWidth,Na);
           	  cudaMemcpy(X_PLOT, X_PLOT_GPU, size_int, cudaMemcpyDeviceToHost);
           	  cudaMemcpy(Y_PLOT, Y_PLOT_GPU, size_int, cudaMemcpyDeviceToHost); 
                  Redraw (winWidth, winHeight);
		  sprintf(tmp,"Total Energy: %.2lf\nRadial Energy: %.2lf\n3-body Energy: %.2lf\nkinetic_energy: %.2lf\nIts: %d\n",
                      energy[3]+energy[4],energy[1],energy[2],energy[4],its); /* on screen*/
                  XmTextSetString (wText2,tmp);
                }
		ZEROING_A_GPU<<<blocksPerGrid, threadsPerBlock>>>(AX_GPU, AY_GPU, Na);
      
	        if(shared==1)
     	  	{
     	          CALC_FORCE_GPU<<<blocksPerGridMat_b, threadsPerBlock>>>(AX_GPU,AY_GPU,VX_GPU,VY_GPU,X_GPU,Y_GPU,bond_atom_small1_GPU,bond_atom_small2_GPU,ETA,A_lat0_GPU,
     	        						      bond_id_GPU,num_atoms_in_bonds_GPU,atom_list_bonds_GPU,bond_list_GPU,md_cg_mode);
     	          if(k_teta>1.0e-5)
     	          {
     	             CALC_3_BODY_FORCE_GPU<<<blocksPerGridMat, threadsPerBlock>>>(AX_GPU,AY_GPU,X_GPU,Y_GPU,k_teta,
     	        		       atom_list_GPU,atom_id_GPU,atom_id2_GPU,neigh_small_grid_GPU,type_odd_even_lists_GPU);
     	          }
     	  	}
     	  	if(shared==0)
     	  	{
     	          CALC_FORCE_no_shared_GPU<<<blocksPerGrid_b, threadsPerBlock>>>(AX_GPU,AY_GPU,VX_GPU,VY_GPU,X_GPU,Y_GPU,Nb,bond_atom1_GPU,bond_atom2_GPU,ETA,A_lat0_GPU,md_cg_mode);
     	          if(k_teta>1.0e-5)
     	          {
     	            CALC_3_BODY_FORCE_no_shared_GPU<<<blocksPerGrid, threadsPerBlock>>>(AX_GPU,AY_GPU,X_GPU,Y_GPU,Na,k_teta,type_odd_even_GPU,
     	        		       neigh1_GPU,neigh2_GPU,neigh3_GPU,neigh4_GPU,neigh5_GPU,neigh6_GPU);
     	          }
     	  	}

	        CALCULATING_XI_GPU<<<blocksPerGrid, threadsPerBlock>>>(AX_GPU, AY_GPU,xi_GPU, Na);
			
		gg_dgg[0]=0.0;
		gg_dgg[1]=0.0;
		cudaMemcpy(gg_dgg_GPU, gg_dgg, 2*sizeof(double), cudaMemcpyHostToDevice); 
		CALCULATING_GG_DGG_GPU<<<blocksPerGrid, threadsPerBlock>>>(gg_dgg_GPU, g_GPU, xi_GPU, n_safa_GPU, Na);
		cudaMemcpy(gg_dgg, gg_dgg_GPU, 2*sizeof(double), cudaMemcpyDeviceToHost); 
		
		if (gg_dgg[0] == 0.0) {
			return its;
		}
		gam=gg_dgg[1]/gg_dgg[0];
		
		CALCULATING_G_XI_GPU<<<blocksPerGrid, threadsPerBlock>>>(xi_GPU, g_GPU, h_GPU, n_safa_GPU, Na, gam);
	}
	nrerror("Too many iterations in frprmn_GPU");
  return its;
}


void linmin_GPU()
{
	double brent(double ax, double bx, double cx,
		double (*f)(double), double tol, double *xmin);
	double f1dim_GPU(double x);
	void mnbrak(double *ax, double *bx, double *cx, double *fa, double *fb,
		double *fc, double (*func)(double));
	double xx,xmin,fx,fb,fa,bx,ax;

	ncom=2*num_atoms;
	
	CALCULATING_PCOM_XICOM_GPU<<<blocksPerGrid2, threadsPerBlock>>>(xi_GPU, xicom_GPU, p_GPU, pcom_GPU, 2*Na);
	
	ax=0.0;
	xx=1.0;
	mnbrak(&ax,&xx,&bx,&fa,&fx,&fb,f1dim_GPU);
	fret=brent(ax,xx,bx,f1dim_GPU,TOL,&xmin);
	
	CALCULATING_XI_P_GPU<<<blocksPerGrid2, threadsPerBlock>>>(xi_GPU, p_GPU, 2*Na, xmin);
	
	
	CALCULATING_X_Y_GPU<<<blocksPerGrid, threadsPerBlock>>>(X_GPU, Y_GPU, p_GPU, n_safa_GPU, Na);
	
}

double f1dim_GPU(double x)
{
	double f;
	
	CALCULATING_XT_DF_GPU<<<blocksPerGrid, threadsPerBlock>>>(pcom_GPU, xicom_GPU, xt_GPU, n_safa_GPU, Na, x);
        CALCULATING_X_Y_GPU<<<blocksPerGrid, threadsPerBlock>>>(X_GPU, Y_GPU, xt_GPU, n_safa_GPU, Na);
	
	calc_all_energy_GPU();
	f=energy[3];
	
	CALCULATING_X_Y_GPU<<<blocksPerGrid, threadsPerBlock>>>(X_GPU, Y_GPU, pcom_GPU, n_safa_GPU, Na);
	
	return f;
}

double** Make2DDoubleArray(int arraySizeX, int arraySizeY)
{  
  /*in the code, please use: double** myArray = Make2DDoubleArray(nx, ny); */
  double** theArray;  
  theArray = (double**) malloc(arraySizeX*sizeof(double*));  
  for (int i = 0; i < arraySizeX; i++)  
     theArray[i] = (double*) malloc(arraySizeY*sizeof(double));  
     return theArray;
}  

int** Make2DIntArray(int arraySizeX, int arraySizeY)
{  
  /*in the code, please use: int** myArray = Make2DIntArray(nx, ny); */
  int** theArray;  
  theArray = (int**) malloc(arraySizeX*sizeof(int*));  
  for (int i = 0; i < arraySizeX; i++)  
     theArray[i] = (int*) malloc(arraySizeY*sizeof(int));  
     return theArray;
}

int*** Make3DIntArray(int arraySizeX, int arraySizeY, int arraySizeZ)
{  
  /*in the code, please use: int** myArray = Make3DIntArray(nx, ny, nz); */
  int*** theArray;  
  theArray = (int***) malloc(arraySizeX*sizeof(int*));  
  for (int i = 0; i < arraySizeX; i++)
     theArray[i] =  Make2DIntArray(arraySizeY, arraySizeZ);
     return theArray;
}  
