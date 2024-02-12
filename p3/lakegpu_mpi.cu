//Group info:
//mshaikh2 Mushtaq Ahmed Shaikh
//ahgada Amay Gada


#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define __DEBUG

#define CUDA_CALL( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_CHK_ERR() __cudaCheckError(__FILE__,__LINE__)

/**************************************
* void __cudaSafeCall(cudaError err, const char *file, const int line)
* void __cudaCheckError(const char *file, const int line)
*
* These routines were taken from the GPU Computing SDK
* (http://developer.nvidia.com/gpu-computing-sdk) include file "cutil.h"
**************************************/

inline void __cudaSafeCall( cudaError err, const char *file, const int line ){
#ifdef __DEBUG

#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
              file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
  } while ( 0 );
#pragma warning( pop )
#endif  // __DEBUG
  return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef __DEBUG
#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
  do
  {
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment if not needed.
    /*err = cudaThreadSynchronize();
    if( cudaSuccess != err )
    {
      fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }*/
  } while ( 0 );
#pragma warning( pop )
#endif // __DEBUG
  return;
}
#define TSCALE 1.0
#define VSQR 0.1
// Function prototypes
//double f(double p, double t);
__host__ __device__ double f(double p, double t) {
  return -expf(-TSCALE * t) * p;
}

int tpdt(double *t, double dt, double tf);
void update_edge_values(double *arr, int npoints, int n_dec, int rank, int size);


__global__ void evolve_gpu(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t, int n_dec, int rank)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int ii,jj,oidx;

    if (i >= n || j >= n_dec)
      return;

    // printf("I: %d J: %d\n", i, j);

    int idx = i + j*n + 2*n;
    oidx = rank*n*n_dec + j*n + i;  //original index in matrix
    jj = oidx % n;
    ii = oidx / n;

    if (ii == 0 || ii == n - 1 || jj == 0 || jj == n - 1 || ii == 1 || ii == n - 2 || jj == 1 || jj == n - 2){
        un[idx] = 0.;
    }
    else{
        un[idx] = 2 * uc[idx] - uo[idx] + VSQR * (dt * dt) *
                                              ((uc[idx - 1] + uc[idx + 1] + uc[idx - n] + uc[idx + n] +
                                                0.25 * (uc[idx - n - 1] + uc[idx - n + 1] + uc[idx + n - 1] + uc[idx + n + 1]) +
                                                0.125 * (uc[idx - 1 - 1] + uc[idx + 1 + 1] + uc[idx - n - n] + uc[idx + n + n]) -
                                                5.5 * uc[idx]) / (h * h) +
                                               f(pebbles[oidx], t));
        
    }
}

void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads, int n_dec, int rank, int size){
	cudaEvent_t kstart, kstop;
	float ktime;

  int narea = (n_dec * (n)) + (4*n);
        
	/* HW2: Define your local variables here */
  // int nblocks = (n/nthreads);

  int block1 = (n/nthreads) + (n%nthreads==0?0:1);
  int block2 = (n/n_dec) + (n%n_dec==0?0:1);

  // printf("BLCOKS : %d, %d", block1, block2);

  double *un, *uc, *uo, *d_pebbles, *uc_h, *uo_h;

  uc_h = (double*)malloc(sizeof(double) * narea);
  uo_h = (double*)malloc(sizeof(double) * narea);
  memcpy(uo_h, u0, sizeof(double) * narea);
  memcpy(uc_h, u1, sizeof(double) * narea);

  cudaMalloc((void **)&un, sizeof(double) * narea);
  cudaMalloc((void **)&uc, sizeof(double) * narea);
  cudaMalloc((void **)&uo, sizeof(double) * narea);
  cudaMalloc((void **)&d_pebbles, sizeof(double) * narea);

        /* Set up device timers */  
	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaEventCreate(&kstart));
	CUDA_CALL(cudaEventCreate(&kstop));

	/* HW2: Add CUDA kernel call preperation code here */
  dim3 blocks(block1, block2);
  dim3 threadsPerBlock(nthreads, nthreads);

	/* Start GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstart, 0));

  cudaMemcpy(d_pebbles, pebbles, sizeof(double) * narea, cudaMemcpyHostToDevice);

	/* HW2: Add main lake simulation loop here */
	double t = 0.;
  double dt = h / 2.;
  while (1)
  {   
      update_edge_values(uc_h, n, n_dec, rank, size);
      update_edge_values(uo_h, n, n_dec, rank, size);

      cudaMemcpy(uo, uo_h, sizeof(double) * narea, cudaMemcpyHostToDevice);
      cudaMemcpy(uc, uc_h, sizeof(double) * narea, cudaMemcpyHostToDevice);

      evolve_gpu<<<blocks, threadsPerBlock>>>(un, uc, uo, d_pebbles, n, h, dt, t, n_dec, rank);

      cudaMemcpy(uo, uc, sizeof(double) * narea, cudaMemcpyDeviceToDevice);
      cudaMemcpy(uc, un, sizeof(double) * narea, cudaMemcpyDeviceToDevice);

      cudaMemcpy(uo_h, uo, sizeof(double) * narea, cudaMemcpyDeviceToHost);
      cudaMemcpy(uc_h, uc, sizeof(double) * narea, cudaMemcpyDeviceToHost);

//      printf("Here in this loop for gpu \n");
      if (!tpdt(&t, dt, end_time))
          break;
  }
        /* Stop GPU computation timer */
  cudaMemcpy(u, un, sizeof(double) * narea, cudaMemcpyDeviceToHost);

	CUDA_CALL(cudaEventRecord(kstop, 0));
	CUDA_CALL(cudaEventSynchronize(kstop));
	CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
	printf("GPU computation: %f msec\n", ktime);

	/* HW2: Add post CUDA kernel call processing and cleanup here */
  cudaFree(un);
  cudaFree(uc);
  cudaFree(uo);
  cudaFree(d_pebbles);
	/* timer cleanup */
	CUDA_CALL(cudaEventDestroy(kstart));
	CUDA_CALL(cudaEventDestroy(kstop));
}