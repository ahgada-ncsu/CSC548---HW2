//Group info:
//mshaikh2 Mushtaq Ahmed Shaikh
//ahgada Amay Gada


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "mpi.h"

#define _USE_MATH_DEFINES

#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0

#define MAX_PSZ 10
#define TSCALE 1.0
#define VSQR 0.1

__host__ __device__ double f(double, double);

void init(double *u, double *pebbles, int n, int n_dec, int rank);
void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t);
int tpdt(double *t, double dt, double end_time);
void print_heatmap(const char *filename, double *u, int n, double h, int n_dec, int rank, int size);
void init_pebbles(double *p, int pn, int n);

void update_edge_values(double *arr, int npoints, int n_dec, int rank, int size);

void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int n_dec, int rank, int size);

extern void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads, int n_dec, int rank, int size);
void evolve9pt(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t, int n_dec, int rank);

int main(int argc, char *argv[])
{

    // Initializations for MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(argc != 5)
    {
        printf("Usage: %s npoints npebs time_finish nthreads \n",argv[0]);
        return 0;
    }

    int     npoints   = atoi(argv[1]);
    int     npebs     = atoi(argv[2]);
    double  end_time  = (double)atof(argv[3]);
    int     nthreads  = atoi(argv[4]);

    // N decomposed -> dividing everything into rows
    int n_dec = npoints/size;
    if(npoints%size != 0 && rank == size-1) n_dec += npoints%size; 

    // area of each grid now equals n_dec * npoints + (4 * npoints) -> extra space for data coming from other ranks
    int 	  narea	    = (n_dec * (npoints)) + (4*npoints);

    double *u_i0, *u_i1;
    double *u_cpu, *u_gpu, *pebs;
    double h;

    double elapsed_cpu, elapsed_gpu;
    struct timeval cpu_start, cpu_end, gpu_start, gpu_end;
    
    u_i0 = (double*)malloc(sizeof(double) * narea);
    u_i1 = (double*)malloc(sizeof(double) * narea);
    pebs = (double*)malloc(sizeof(double) * npoints * npoints);
    u_cpu = (double*)malloc(sizeof(double) * narea);
    u_gpu = (double*)malloc(sizeof(double) * narea);

    h = (XMAX - XMIN)/npoints;

    if(rank == 0) init_pebbles(pebs, npebs, npoints);             //initialize pebbles in rank 0.
    
    MPI_Bcast(pebs, npoints*npoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);  // Broadcast pebs to all other ranks.

    // initializing u_i0 and u_i1 based on the received pebbles (Note that the init function has been changed to accomodate the decomposition of the matrix)
    init(u_i0, pebs, npoints, n_dec, rank);               
    init(u_i1, pebs, npoints, n_dec, rank);

    // Writing the initial heatmap to lake_i_rank.dat (Note that the print_heatmap function has also been changed to accomodate the decomposition of the matrix)
    char file_name[100];
    sprintf(file_name, "lake_i_%d.dat", rank);
    print_heatmap(file_name, u_i0, npoints, h, n_dec, rank, size);

    // running for cpu
    gettimeofday(&cpu_start, NULL);
    run_cpu(u_cpu, u_i0, u_i1, pebs, npoints, h, end_time, n_dec, rank, size);
    gettimeofday(&cpu_end, NULL);
    elapsed_cpu = ((cpu_end.tv_sec + cpu_end.tv_usec * 1e-6)-(
                    cpu_start.tv_sec + cpu_start.tv_usec * 1e-6));
    printf("CPU took %f seconds\n", elapsed_cpu);

    // running for GPU
    gettimeofday(&gpu_start, NULL);
    run_gpu(u_gpu, u_i0, u_i1, pebs, npoints, h, end_time, nthreads, n_dec, rank, size);
    gettimeofday(&gpu_end, NULL);
    elapsed_gpu = ((gpu_end.tv_sec + gpu_end.tv_usec * 1e-6)-(
                    gpu_start.tv_sec + gpu_start.tv_usec * 1e-6));
    printf("GPU took %f seconds\n", elapsed_gpu);

    // writing the heatmap per rank
    sprintf(file_name, "lake_f_%d.dat", rank);
    print_heatmap(file_name, u_cpu, npoints, h, n_dec, rank, size);

    // free memory
    free(u_i0);
    free(u_i1);
    free(pebs);
    free(u_cpu);
    free(u_gpu);

    // Ending MPI
    MPI_Finalize();
    return 0;  // changed to 0, because MPI threw error otherwise
}

// This sole function is responsible for communication between nodes that updates the heatmap in each rank. This function gets top 2*n and bottom 2*n values from the bottom and top rank of the process.
void update_edge_values(double *arr, int npoints, int n_dec, int rank, int size){
  if(rank%2 == 0){
    //send to next and prev
    if(rank+1 < size) MPI_Send(&arr[npoints*n_dec], 2*npoints, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
    if(rank-1 >= 0)    MPI_Send(&arr[2*npoints], 2*npoints, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD);

    if(rank-1 >= 0)    MPI_Recv(arr, 2*npoints, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if(rank+1 < size) MPI_Recv(&arr[(2*npoints) + (npoints*n_dec)], 2*npoints, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }else{
    // recv from next and prev
    if(rank-1 >= 0)    MPI_Recv(arr, 2*npoints, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if(rank+1 < size) MPI_Recv(&arr[(2*npoints) + (npoints*n_dec)], 2*npoints, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if(rank+1 < size) MPI_Send(&arr[npoints*n_dec], 2*npoints, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
    if(rank-1 >= 0)    MPI_Send(&arr[2*npoints], 2*npoints, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD);
  }
}


void run_cpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int n_dec, int rank, int size)
{
  double *un, *uc, *uo;
  double t, dt;

  int narea = (n_dec * (n)) + (4*n);

  un = (double*)malloc(sizeof(double) * narea);
  uc = (double*)malloc(sizeof(double) * narea);
  uo = (double*)malloc(sizeof(double) * narea);

  memcpy(uo, u0, sizeof(double) * narea);
  memcpy(uc, u1, sizeof(double) * narea);

  t = 0.;
  dt = h / 2.;

  while(1)
  {
    update_edge_values(uo, n, n_dec, rank, size);                   // updating the heatmap here
    update_edge_values(uc, n, n_dec, rank, size);                   // updating the heatmap here
    evolve9pt(un, uc, uo, pebbles, n, h, dt, t, n_dec, rank);       // evolving

    memcpy(uo, uc, sizeof(double) * narea);
    memcpy(uc, un, sizeof(double) * narea);
    if(!tpdt(&t,dt,end_time)) break;
  }
  
  memcpy(u, un, sizeof(double) * narea);	
}

// initializes pebbles at rank 0. No MPI modification
void init_pebbles(double *p, int pn, int n)
{
  int i, j, k, idx;
  int sz;

  srand( time(NULL) );
  memset(p, 0, sizeof(double) * n * n);

  for( k = 0; k < pn ; k++ )
  {
    i = rand() % (n - 4) + 2;
    j = rand() % (n - 4) + 2;
    sz = rand() % MAX_PSZ;
    idx = j + i * n;
    p[idx] = (double) sz;
  }
}


int tpdt(double *t, double dt, double tf)
{
  if((*t) + dt > tf) return 0;
  (*t) = (*t) + dt;
  return 1;
}

// added certain calculations to parallelly initialize the heatmap with pebbles.
void init(double *u, double *pebbles, int n, int n_dec, int rank)
{
  int i, j, idx, uidx;

  for(i = 0; i < n_dec ; i++){
    for(j = 0; j < n ; j++){
      idx = rank*n*n_dec + i*n + j;
      uidx = idx%(n*n_dec);
      u[2*n + uidx] = f(pebbles[idx], 0.0);
    }
  }
}

// evolving
void evolve(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t)
{
  int i, j, idx;

  for( i = 0; i < n; i++)
  {
    for( j = 0; j < n; j++)
    {
      idx = j + i * n;

      if( i == 0 || i == n - 1 || j == 0 || j == n - 1)
      {
        un[idx] = 0.;
      }
      else
      {
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *((uc[idx-1] + uc[idx+1] + 
                    uc[idx + n] + uc[idx - n] - 4 * uc[idx])/(h * h) + f(pebbles[idx],t));
      }
    }
  }
}

// evolving with 9 points
void evolve9pt(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t, int n_dec, int rank)
{
  int i, j, idx, ii, jj, oidx;

  for( i = 0; i < n_dec; i++)
  {
    for( j = 0; j < n; j++)
    {
      idx = j + i*n + 2*n;
      oidx = rank*n*n_dec + i*n + j;  //original index in matrix
      jj = oidx % n;
      ii = oidx / n;

      if( ii == 0 || ii == n - 1 || jj == 0 || jj == n - 1 || ii==1 || ii==n-2 || jj==1 || jj==n-2)
      {
        un[idx] = 0.;
      }
      else
      {
        un[idx] = 2*uc[idx] - uo[idx] + VSQR *(dt * dt) *
        (( uc[idx-1] + uc[idx+1] + uc[idx-n] + uc[idx+n] + 
        0.25*(uc[idx-n-1] + uc[idx-n+1] + uc[idx+n-1] + uc[idx+n+1]) +
        0.125*(uc[idx-1-1] + uc[idx+1+1] + uc[idx-n-n] + uc[idx+n+n]) -
        5.5 * uc[idx])/(h * h) + f(pebbles[oidx],t));
      }
    }
  }
}

// printing heatmap -> modified this as well
void print_heatmap(const char *filename, double *u, int n, double h, int n_dec, int rank, int size)
{
  int i, j, idx, k;

  k = n/size;

  FILE *fp = fopen(filename, "w");  

  for( i = 0; i < n_dec; i++ )
  {
    for( j = 0; j < n; j++ )
    {
      idx = j + i*n;
      fprintf(fp, "%f %f %f\n", (i+rank*k)*h, j*h, u[2*n+idx]);
    }
  }
  
  fclose(fp);
}