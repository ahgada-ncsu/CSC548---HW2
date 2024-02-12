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
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
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


__global__ void evolve_gpu(double *un, double *uc, double *uo, double *pebbles, int n, double h, double dt, double t)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n || j >= n)
      return;

    int idx = j + i * n;

    if (i == 0 || i == n - 1 || j == 0 || j == n - 1 || i == 1 || i == n - 2 || j == 1 || j == n - 2)
    {
        un[idx] = 0.;
    }
    else
    {
        un[idx] = 2 * uc[idx] - uo[idx] + VSQR * (dt * dt) *
                                              ((uc[idx - 1] + uc[idx + 1] + uc[idx - n] + uc[idx + n] +
                                                0.25 * (uc[idx - n - 1] + uc[idx - n + 1] + uc[idx + n - 1] + uc[idx + n + 1]) +
                                                0.125 * (uc[idx - 1 - 1] + uc[idx + 1 + 1] + uc[idx - n - n] + uc[idx + n + n]) -
                                                5.5 * uc[idx]) / (h * h) +
                                               f(pebbles[idx], t));
        
    }
}

void run_gpu(double *u, double *u0, double *u1, double *pebbles, int n, double h, double end_time, int nthreads)
{
	cudaEvent_t kstart, kstop;
	float ktime;
        
	/* HW2: Define your local variables here */
  int nblocks = (n/nthreads) + (n%nthreads==0?0:1);
  double *un, *uc, *uo, *d_pebbles;
  cudaMalloc((void **)&un, sizeof(double) * n * n);
  cudaMalloc((void **)&uc, sizeof(double) * n * n);
  cudaMalloc((void **)&uo, sizeof(double) * n * n);
  cudaMalloc((void **)&d_pebbles, sizeof(double) * n * n);
        /* Set up device timers */  
	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaEventCreate(&kstart));
	CUDA_CALL(cudaEventCreate(&kstop));

	/* HW2: Add CUDA kernel call preperation code here */
  dim3 blocks(nblocks, nblocks);
  dim3 threadsPerBlock(nthreads, nthreads);


	/* Start GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstart, 0));

  cudaMemcpy(uo, u0, sizeof(double) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(uc, u1, sizeof(double) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pebbles, pebbles, sizeof(double) * n * n, cudaMemcpyHostToDevice);

	/* HW2: Add main lake simulation loop here */
	double t = 0.;
  double dt = h / 2.;
  while (1)
  {
      evolve_gpu<<<blocks, threadsPerBlock>>>(un, uc, uo, d_pebbles, n, h, dt, t);

      cudaMemcpy(uo, uc, sizeof(double) * n * n, cudaMemcpyDeviceToDevice);
      cudaMemcpy(uc, un, sizeof(double) * n * n, cudaMemcpyDeviceToDevice);
//      printf("Here in this loop for gpu \n");
      if (!tpdt(&t, dt, end_time))
          break;
  }
        /* Stop GPU computation timer */
  cudaMemcpy(u, un, sizeof(double) * n * n, cudaMemcpyDeviceToHost);

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
