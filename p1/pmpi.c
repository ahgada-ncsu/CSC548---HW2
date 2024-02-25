#include <stdio.h>
#include "mpi.h"
#include <stdlib.h>

int *matrix;

// Profile for MPI Init
int MPI_Init (int *argc, char ***argv) {
    int err = PMPI_Init(argc, argv);

    //getting the size
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //allocating space to matrix so it can store the count of sends per rank
    matrix = (int *)malloc(sizeof(int) * size);
    
    //initializing the matrix to all 0s
    for(int j=0; j<size; j++) matrix[j] = 0;

    return err;
}


// Profile for MPI_Send
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm){
    // increenting the send count to destination in the local matrix
    matrix[dest] += 1;
    return PMPI_Send(buf, count, datatype, dest, tag, comm);
}


// Profile for MPI_Isend
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request){
    // increenting the send count to destination in the local matrix
    matrix[dest] += 1;
    return PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
}


// Profile for MPI_Finalize
int MPI_Finalize(){
    // getting the size
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // getting the rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // allocating space to store the entire 2d matrix at rank 0
    int *matrix_at_zero;
    if(rank==0){
        matrix_at_zero = (int *)malloc(sizeof(int) * size * size);
    }

    // do MPIGather to gather the entire 2d matrix from all the processes to rank 
    MPI_Gather(matrix, size, MPI_INT, matrix_at_zero, size, MPI_INT, 0, MPI_COMM_WORLD);

    // write the 2d matrix at rank 0 to a file "matrix.data"
    if(rank==0){
        FILE *f = fopen("matrix.data", "w");
        for(int i=0; i<size; i++){
            for(int j=0; j<size; j++){
                fprintf(f, "%d ", matrix_at_zero[i*size+j]);
            }
            fprintf(f, "\n");
        }
    }

    int err = PMPI_Finalize();
    return err;
}
