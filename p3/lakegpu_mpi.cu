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


__global__ void evolve_gpu(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t, int n_dec, int rank, int size)
{ 
    // getting i and j from gpu thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int ii,jj,oidx;
    int k = n/size;

    // printf("blockDim: %d, %d\n", blockIdx.x, blockIdx.y);

    // restricting invalid indices
    if (i >= n || j >= n_dec)
      return;

    // same as in cpu evolve
    int idx = i + j*n + 2*n;
    oidx = rank*n*k + j*n + i;  //original index in matrix
    // printf("OIDX: %d\n", oidx);
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

// code for running on gpu
void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads, int n_dec, int rank, int size){
	cudaEvent_t kstart, kstop;
	float ktime;

  int narea = (n_dec * (n)) + (4*n);  // get area for malloc later

  int block1 = (n/nthreads) + (n%nthreads==0?0:1); // size of 1 block
  int block2 = (n_dec/nthreads) + (n_dec%nthreads==0?0:1);       // size of the second block

  // printf("BLOCKS: %d, %d\n", block1, block2);

  double *un, *uc, *uo, *d_pebbles, *uc_h, *uo_h;

  uc_h = (double*)malloc(sizeof(double) * narea);
  uo_h = (double*)malloc(sizeof(double) * narea);
  memcpy(uo_h, u0, sizeof(double) * narea);
  memcpy(uc_h, u1, sizeof(double) * narea);

  // allocating on device
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

  // copying to device
  cudaMemcpy(d_pebbles, pebbles, sizeof(double) * narea, cudaMemcpyHostToDevice);

	/* HW2: Add main lake simulation loop here */
	double t = 0.;
  double dt = h / 2.;
  while (1)
  {   
      // updating edge values
      update_edge_values(uc_h, n, n_dec, rank, size);
      update_edge_values(uo_h, n, n_dec, rank, size);

      // copying updated data to device
      cudaMemcpy(uo, uo_h, sizeof(double) * narea, cudaMemcpyHostToDevice);
      cudaMemcpy(uc, uc_h, sizeof(double) * narea, cudaMemcpyHostToDevice);

      // evolving on device
      evolve_gpu<<<blocks, threadsPerBlock>>>(un, uc, uo, d_pebbles, n, h, dt, t, n_dec, rank, size);

      // some internal on device copying
      cudaMemcpy(uo, uc, sizeof(double) * narea, cudaMemcpyDeviceToDevice);
      cudaMemcpy(uc, un, sizeof(double) * narea, cudaMemcpyDeviceToDevice);

      // copyin heatmap back on host for updation
      cudaMemcpy(uo_h, uo, sizeof(double) * narea, cudaMemcpyDeviceToHost);
      cudaMemcpy(uc_h, uc, sizeof(double) * narea, cudaMemcpyDeviceToHost);


      if (!tpdt(&t, dt, end_time))
          break;
  }
        /* Stop GPU computation timer */
  
  // copying back from host to device
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