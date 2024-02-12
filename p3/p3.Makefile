lake: lakegpu.cu lake.cu
	nvcc lakegpu.cu lake.cu -o lake -O3 -lm -Wno-deprecated-gpu-targets -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64

lake-mpi: lake_mpi.cu  lakegpu_mpi.cu
	nvcc -c lakegpu_mpi.cu -O3 -lm -Wno-deprecated-gpu-targets -I${MPI_DIR}/include -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64
	nvcc -c lake_mpi.cu -O3 -lm -Wno-deprecated-gpu-targets -I${MPI_DIR}/include -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64
	mpicc lakegpu_mpi.o lake_mpi.o -o lake_mpi -O3 -lm -L${CUDA_HOME}/lib64 -lcudart -lstdc++

clean: 
	rm -f lake *.o

clean_dat:
	rm -f lake *.dat
	rm -f lake *.png