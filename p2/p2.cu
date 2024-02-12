// ahgada Amay Haresh Gada 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define XI  -M_PI/4.0
#define XF  M_PI/4.0

#define THREADS 32

//Function declarations
double  fn(double);
void    print_function_data(int, double *, double *, double *);
int     main(int, char**);

// function that we have to integrate
double fn(double x){
  return atan(x);
}

// the function that runs on device -> each thread handles computation for 1 xc[i]
__global__ void integrate(int *N, double *inf, double *yc, double *h) {
    int idx = ( blockIdx.x * blockDim.x ) + threadIdx.x + 1;
    if(idx <= *N){
      double area;
      if(idx>=2) area = ( yc[idx] + yc[idx-1] ) / 2 * *h;
      else area = 0.0;
      inf[idx] = area;
    }
}

int main(int argc, char *argv[]){

    int NGRID;
    if(argc > 1) NGRID = atoi(argv[1]);
    else{
            printf("Please specify the number of grid points.\n");
            exit(0);
    }

    int         i;
    double  h;

    double *inf = (double *)malloc(sizeof(double) * (NGRID + 1) );
    double  *xc = (double *)malloc(sizeof(double)* (NGRID + 1));
    double  *yc = (double*)malloc(sizeof(double) * (NGRID + 1));
    
    //construct grid of x axis
    for (i = 1; i <= NGRID ; i++)
    {
            xc[i] = XI + (XF - XI) * (double)(i - 1)/(double)(NGRID - 1);
    }
    
    int  imin, imax;  

    imin = 1;
    imax = NGRID;
    
    // get the y value of the origin function, yc array is used for output
    // should not use for computing on GPU
    for( i = imin; i <= imax; i++ )
    {
            yc[i] = fn(xc[i]);
    }

    inf[1] = 0.0;
    h = (XF - XI) / (NGRID - 1);
 
    int *N_d;             // device copy of NGRIDS
    double *inf_d;        // device copy of inf
    double *yc_d;         // device copy of yc
    double *h_d;          // device copy of h

    int BLOCKS = NGRID/THREADS + (NGRID%THREADS==0?0:1);    // NUMBER OF BLOCKS IN THE GRID

    // allocating space on device for variables
    cudaMalloc( (void **) &N_d, sizeof(int) * 1 );
    cudaMalloc( (void **) &h_d, sizeof(double) * 1 );    
    cudaMalloc( (void **) &inf_d, sizeof(double)*(NGRID+1));
    cudaMalloc( (void **) &yc_d, sizeof(double)*(NGRID+1));
    
    // copying the value of NGRIDS to device
    cudaMemcpy(N_d, &NGRID, sizeof(int) * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(h_d, &h, sizeof(double) * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(yc_d, yc, sizeof(double) * (NGRID + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(inf_d, inf, sizeof(double) * (NGRID + 1), cudaMemcpyHostToDevice);

    //calling the function on device
    integrate <<<BLOCKS,THREADS>>> (N_d, inf_d, yc_d, h_d);
    
    // copying back inf to host from device. inf now holds areas at individual grid points on the x axis
    cudaMemcpy(inf, inf_d, sizeof(double) * (NGRID + 1), cudaMemcpyDeviceToHost);

    // aggregating individual areas to get final integral
    for(int k=2; k<=NGRID; k++){
      inf[k] += inf[k-1];
    }

    // printing the data into a file.dat
    print_function_data(NGRID, &xc[1], &yc[1], &inf[1]);

    // freeing all the GPU memory
    cudaFree(N_d);
    cudaFree(inf_d);
    cudaFree(yc_d);
    cudaFree(h_d);

    // freeing all the host memory
    free(xc);
    free(yc);
    free(inf);

    return 0;  
} 

//prints out the function and its derivative to a file
void print_function_data(int np, double *x, double *y, double *dydx)
{
        int   i;

        char filename[1024];
        sprintf(filename, "fn-%d.dat", np);

        FILE *fp = fopen(filename, "w");

        for(i = 0; i < np; i++)
        {
                fprintf(fp, "%f %f %f\n", x[i], y[i], dydx[i]);
        }

        fclose(fp);
}