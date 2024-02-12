// ahgada Amay Haresh Gada 

#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>
#include <string.h>
#include <netinet/in.h>
#include <unistd.h>
#include <sys/types.h>
#include <netdb.h>
#include "my_mpi.h"

MPI_Comm MPI_COMM_WORLD;	

// template for all socket errors for debugging
void error(const char *msg){
    perror(msg);
    exit(1);
}

// creates and binds socket at port defined. 
int create_socket(int port){
    int sockfd = -1;
    struct sockaddr_in serv_addr;
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) error("ERROR opening socket");
    int optval = 1; //SO_SNDBUF
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)) < 0) error("setsockopt");
    struct timeval timeout;
    timeout.tv_sec = 5;
    timeout.tv_usec = 0;
    if(setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) error("setsockopt failed");
    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(port);
    if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) error("ERROR on binding");
    listen(sockfd,5);
    return sockfd; //returns the socket file descriptor for communication
}

// allows creation of socket connections from process calling the function to another host at defined port 
int connect_to_process(char *hostname, int port){
    int sockfd;
    struct sockaddr_in serv_addr;
    struct hostent *server;
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)  error("ERROR opening socket");
    server = gethostbyname(hostname);
    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr, (char *)&serv_addr.sin_addr.s_addr, server->h_length);
    serv_addr.sin_port = htons(port);
    while (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0);
    return sockfd; // returns client file descriptor for communication
}


// used for initial communication in MPI_Init. Sends message to process at defined hostname and port
int send_msg(char *message, char *hostname, int port){
    int n = -1;
    int clientfd = connect_to_process(hostname, port);
    n = write(clientfd,message,strlen(message)); //write to file descripter -> equivalent to sending data to receiever.
    return(clientfd); // close connection
}


// used for initial communication in MPI_Init. Receives message from any process trying to communicate with the bound server.
int recv_msg(int server_fd, char **msg){
    socklen_t clilen;
    struct sockaddr_in cli_addr;
    clilen = sizeof(cli_addr);
    char buffer[1024];
    int newsockfd = accept(server_fd, (struct sockaddr *) &cli_addr, &clilen); // accept connection
    if (newsockfd < 0) error("ERROR on accept");
    bzero(buffer,1024);
    int n = read(newsockfd,buffer,1023);                                       // read the message into buffer
    if (n < 0) error("ERROR reading from socket");
    *msg = (char *) malloc(n);
    memcpy(*msg, buffer, n);
    return(newsockfd);
}


// global socket information stores hostlist information. This function parses the string and populates the Hostlist.
void break_global_socket_info(char *gsi){
    char *token;
    token = strtok(gsi, "\n,");
    while (token != NULL) {
        char *host = token;
        token = strtok(NULL, "\n,");
        int rank = atoi(token);
        token = strtok(NULL, "\n,");
        int port = atoi(token);
        token = strtok(NULL, "\n,");
        strcpy(MPI_COMM_WORLD.Hostlist[rank].hostname + strlen(MPI_COMM_WORLD.Hostlist[rank].hostname), host);
        MPI_COMM_WORLD.Hostlist[rank].port = port;       //   ''
    }
}


// My implementation of MPI_Init
int MPI_Init(int *argc, char **argv[]){
    // initialize mpi variables from environment
    MPI_COMM_WORLD.rank = atoi(getenv("MYMPI_RANK"));
    MPI_COMM_WORLD.size = atoi(getenv("MYMPI_NTASKS"));
    MPI_COMM_WORLD.hostname = getenv("MYMPI_NODENAME");
    char *NodeZeroHN = getenv("MYMPI_NODEZERO");

    // each process creates a socket connection and starts litening on port 1024 + Rank
    int sockfd = create_socket(1024 + MPI_COMM_WORLD.rank);
    if(sockfd<0) error("ERROR IN CREATING SERVER");
    MPI_COMM_WORLD.sockfd = sockfd;
    MPI_COMM_WORLD.port = 1024 + MPI_COMM_WORLD.rank;

    //allocate space to Hostlist. Depends on nuber of processes.
    MPI_COMM_WORLD.Hostlist = (Host *)malloc(MPI_COMM_WORLD.size * sizeof(Host));
    
    // stores global socket information
    char global_socket_info[500] = "";
    char *nl = "\n";

    // each rank sends socket information to rank 0. The message it sends is constrcted here.
    char rank_s[2];
    char port_s[6];
    char message[30] = "";
    strcat(message, MPI_COMM_WORLD.hostname);
    sprintf(rank_s, "%d", MPI_COMM_WORLD.rank);
    sprintf(port_s, "%d", MPI_COMM_WORLD.port);
    strcat(message, ",");
    strcat(message, rank_s);
    strcat(message, ",");
    strcat(message, port_s);

    // Rank 0 receives message from all other ranks. All other ranks, send message to rank 0.  -- The message has socket information [LOGICAL GATHER]
    if(MPI_COMM_WORLD.rank == 0){  
        for(int i=1; i<MPI_COMM_WORLD.size; i++){
            char *msg; 
            int fd = recv_msg(MPI_COMM_WORLD.sockfd, &msg);
            close(fd);
            strcat(global_socket_info, msg);
            strcat(global_socket_info, nl);
            free(msg);
        } 
        strcat(global_socket_info, message);
    }else{
        int fd = send_msg(message, NodeZeroHN, 1024);
        close(fd);
    }

    // Rank 0 sends global socoket information to all processes. [LOGICAL BROADVAST]
    if(MPI_COMM_WORLD.rank == 0){
        char gsi_copy[500];
        strcpy(gsi_copy, global_socket_info);
        break_global_socket_info(gsi_copy);
        for(int i=1; i<MPI_COMM_WORLD.size; i++){
            int fd = send_msg(global_socket_info, MPI_COMM_WORLD.Hostlist[i].hostname, MPI_COMM_WORLD.Hostlist[i].port);
            close(fd);
        }
    }else{
        char *msg; 
        int fd = recv_msg(MPI_COMM_WORLD.sockfd, &msg);
        close(fd);
        // printf("[RANK: %d] global socket info: %s\n", MPI_COMM_WORLD.rank, msg);
        break_global_socket_info(msg);
        free(msg);
    }
    
    sleep(3);  // sleep for 3 for safety

    return 0;
}


// My implementation of MPI_Send
/*
    - Sends header to receiver (rank).
    - sends data to receiver
*/
int MPI_Send(void *buffer, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm){
    char *host = comm.Hostlist[dest].hostname;
    int portno = comm.Hostlist[dest].port;
    int clientfd = connect_to_process(host, portno);
    int total=0;
    while(total<count*datatype){
        int n = write(clientfd, buffer+total, count*datatype);
        if(n<0) perror("SEND ERROR");
        if(n==0) break;
        total+=n;
    }
    close(clientfd);
    return(0);
}


// My impleentation of MPI_Recv
/*
    - Receives rank from sender (header)
    - To allow for acceptance from the correct source, I assume there is some sort of queue inside MPI_Recv. I am not implementing the queue right now for the sake of siplicity and taking into account the time given. Assuming corect order of receiving.
    - Receive data from sender.
*/
int MPI_Recv(void *buffer, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status){
    //receive header (or check if message already received in queue)
    socklen_t clilen;
    struct sockaddr_in cli_addr;
    clilen = sizeof(cli_addr);
    int newsockfd = accept(comm.sockfd, (struct sockaddr *) &cli_addr, &clilen); // accept connection

    if (newsockfd < 0) error("ERROR on accept");
    int total=0;
    while(total<count*datatype){
        int n = read(newsockfd, buffer+total, count*datatype);
        if(n<0) perror("RECV ERROR");
        total+=n;
        // printf("Recv total i: %d %d\n", total, count*datatype);
        if(n==0) break;
    }
    close(newsockfd);
    return 0;
}


// My implementation of MPI_Comm_size
int MPI_Comm_size ( MPI_Comm comm, int *size ){
    *size = comm.size;
    return 0;
}


// My implementation of MPI_Comm_rank
int MPI_Comm_rank ( MPI_Comm comm, int *rank ){
    *rank = comm.rank;
    return 0;
}

int MPI_Barrier(MPI_Comm comm){
    if(comm.rank != 0){
        char a[1] = "a";
        MPI_Send(a, 1, 1, 0, 1, comm);
    }else{
        char a[1];
        for(int j=1; j<comm.size; j++)
        MPI_Recv(a, 1, 1, j, 1, comm, MPI_STATUS_IGNORE);
    }

    return 0;
}

// My implementation of MPI_Finalize
int MPI_Finalize(){
    // MPI_Barrier(MPI_COMM_WORLD);
    close(MPI_COMM_WORLD.sockfd); // closes persistent socket connections
    return 0;
}