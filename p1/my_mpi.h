// // ahgada Amay Haresh Gada
#ifndef MY_MPI_H_
#define MY_MPI_H_
#include <stdio.h>
#include <stdlib.h>

// definitions
#define MPI_Datatype int	
#define MPI_DOUBLE 8
#define MPI_INT 4
#define MPI_CHAR 1	
#define MPI_Status int
#define MPI_STATUS_IGNORE 0

// holds the hostname and port for each process in the Communicator
typedef struct{
    char hostname[5];
    int port;
}Host;

// my structure for the communicator -> holds all important things for each process in the communicator
typedef struct {
    int size;
    int rank;		
 	int sockfd;
    int port;
	char *hostname;	
    Host *Hostlist;
}MPI_Comm;

// defining a variable for the WORLD Communicator
extern MPI_Comm MPI_COMM_WORLD;	

// Function APIs for MPI
int MPI_Comm_size ( MPI_Comm comm, int *size );	
int MPI_Comm_rank ( MPI_Comm comm, int *rank );	
int MPI_Init( int *argc, char **argv[] );	
int MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm); 
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status); 
int MPI_Finalize();
int MPI_Barrier(MPI_Comm comm);

#endif