# // ahgada Amay Haresh Gada 

# Compiler and flags
CC = nvcc

p1make:
	$(CC) p2.cu -o p2 -O3 -lm -Wno-deprecated-gpu-targets -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64

