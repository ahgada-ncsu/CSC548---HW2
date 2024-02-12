// ahgada Amay Haresh Gada 

#include <stdio.h>
#include <stdlib.h>
#include "my_mpi.h"
#include <sys/times.h>
#include <math.h>


// Finds the min, max, mean, and std deviation for the given array arr[]
void find_stats(unsigned long arr[], int size, double *mean, double *min, double *max, double *std_dev){
    *min = 1000000, *max = -1000000;
    double sum = 0;
    for(int i=0; i<size; i++){
        if((double)arr[i]<*min) *min = (double) arr[i];
        if((double)arr[i]>*max) *max = (double) arr[i];
        sum += (double) arr[i];
    }
    *mean = sum/(double)size;
    sum = 0;
    for(int i=0; i<size; i++){
        sum += ((double)arr[i] - *mean) * ((double)arr[i] - *mean);    
    }
    double temp = sum/(double)size;
    *std_dev = sqrt(temp);
}

// the function that defines how to get minimum of two numbers
double min(double a, double b){
    if(a<b) return a;
    return b;
}

// the function that defines how to get maximum of two numbers
double max(double a, double b){
    if(a>b) return a;
    return b;
}


int main(int argc, char *argv[]){
    int numproc, my_rank, len;

    MPI_Init(&argc, &argv);

    //get number of processes in the communicator
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);

    //get my rank in the comm
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    //524288
    for(int s=8192*1; s<=524288; s*=2){ // iterating through message sizes
        
        // if(my_rank == 0) printf("S: %d\n", s);

        //each node craetes the message
        int *message = (int *) malloc(s*sizeof(int)); //alocating space for the sending message
        for(int i=0; i<s; i++) message[i] = i;

        int *data = (int *) malloc(s*sizeof(int)); // allocating space for the receving message
        int buf_size = s;

        int mid = numproc/2; // helps in differentiating between 2 nodes

        //benchamrking start
        struct timeval start;
        struct timeval end;
        unsigned long e_usec;

        unsigned long node1_arr[100]; // array to store results for RTT from node 1
        unsigned long node2_arr[100]; // can ignore. used to store RTT for node 2 because initially we were asked to plot 10 bars.

        for(int iter=0; iter<101; iter++){
            // Calculating Node 1 to Node 2 RTT
            if(my_rank < mid){
                //send to my_rank + mid
                gettimeofday(&start, 0);
                MPI_Send(message, buf_size, MPI_INT, my_rank+mid, 1, MPI_COMM_WORLD);
                MPI_Recv(data, buf_size, MPI_INT, my_rank+mid, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                gettimeofday(&end, 0);
                e_usec = ((end.tv_sec * 1000000) + end.tv_usec) - ((start.tv_sec * 1000000) + start.tv_usec);
                if(iter!=0) node1_arr[iter-1] = e_usec;
            }else{
                //recv from my_rank - mid
                MPI_Recv(data, buf_size, MPI_INT, my_rank-mid, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(data, buf_size, MPI_INT, my_rank-mid, 1, MPI_COMM_WORLD);
            }

            // Calculating Node 2 to Node 1 RTT (Can ignore now | used to store RTT for node 2 because initially we were asked to plot 10 bars)
            if(my_rank >= mid){
                gettimeofday(&start, 0);
                MPI_Send(message, buf_size, MPI_INT, my_rank-mid, 1, MPI_COMM_WORLD);
                MPI_Recv(data, buf_size, MPI_INT, my_rank-mid, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                gettimeofday(&end, 0);
                e_usec = ((end.tv_sec * 1000000) + end.tv_usec) - ((start.tv_sec * 1000000) + start.tv_usec);
                if(iter!=0) node2_arr[iter-1] = e_usec;
            }else{
                //recv from my_rank - mid
                MPI_Recv(data, buf_size, MPI_INT, my_rank+mid, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(data, buf_size, MPI_INT, my_rank+mid, 1, MPI_COMM_WORLD);
            }
        }


        // find max, min, mean, std dev for distribution
        double minn, maxx, mean, stddev;
        if(my_rank < mid) find_stats(node1_arr, 100, &mean, &minn, &maxx, &stddev);
        else find_stats(node2_arr, 100, &mean, &minn, &maxx, &stddev);
        // printf("MIN: %lf, MAX: %lf, MEAN: %lf, STDDEV: %lf\n", minn, maxx, mean, stddev);

        //stores stats in a transferrable array
        double stats_arr[4];
        stats_arr[0] = minn;
        stats_arr[1] = maxx;
        stats_arr[2] = mean;
        stats_arr[3] = stddev;

        sleep(2);        

        // Send all data to rank 0 for aggregation
        if(my_rank != 0 && my_rank<mid){
            MPI_Send(stats_arr, 4, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }else if(my_rank==0){
            double node1_min=minn, node1_max=maxx, node1_mean=mean, node1_std=stddev;
            for(int j=1; j<mid; j++){
                double temp[4];
                MPI_Recv(temp, 4, MPI_DOUBLE, j, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                node1_min = min(node1_min, temp[0]);
                node1_max = max(node1_max, temp[1]);
                node1_mean += temp[2];
                node1_std += temp[3];
            }

            //aggregating
            node1_mean/=(double)mid;
            node1_std/=(double)mid;

            if(numproc != 8) printf("MESSAGE_SIZE: %d                    min: %lf, max: %lf, mean: %lf, std: %lf\n", s/256, node1_min, node1_max, node1_mean, node1_std);
        }
        
        //hardcode for last case
        /*
            0 <-> 6
            1 <-> 7
            2 <-> 3
            4 <-> 5
        */
        if(numproc == 8){
            //Inter Process Node 1
            unsigned long inter_node1_arr[100];
            unsigned long inter_node2_arr[100];
            unsigned long intra_node1_arr[100];
            unsigned long intra_node2_arr[100];

            for(int iter=0; iter<101; iter++){
                if(my_rank < 2){
                    gettimeofday(&start, 0);
                    MPI_Send(message, buf_size, MPI_INT, my_rank+6, 1, MPI_COMM_WORLD);
                    MPI_Recv(data, buf_size, MPI_INT, my_rank+6, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    gettimeofday(&end, 0);
                    e_usec = ((end.tv_sec * 1000000) + end.tv_usec) - ((start.tv_sec * 1000000) + start.tv_usec);
                    if(iter!=0) inter_node1_arr[iter-1] = e_usec;
                }else if(my_rank > 5){
                    MPI_Recv(data, buf_size, MPI_INT, my_rank-6, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(data, buf_size, MPI_INT, my_rank-6, 1, MPI_COMM_WORLD);
                }else if(my_rank == 2){
                    gettimeofday(&start, 0);
                    MPI_Send(message, buf_size, MPI_INT, 3, 1, MPI_COMM_WORLD);
                    MPI_Recv(data, buf_size, MPI_INT, 3, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    gettimeofday(&end, 0);
                    e_usec = ((end.tv_sec * 1000000) + end.tv_usec) - ((start.tv_sec * 1000000) + start.tv_usec);
                    if(iter!=0) intra_node1_arr[iter-1] = e_usec;
                }else if(my_rank == 3){
                    MPI_Recv(data, buf_size, MPI_INT, 2, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(data, buf_size, MPI_INT, 2, 1, MPI_COMM_WORLD);
                }
            }
        

            // can ignore now .....
            for(int iter=0; iter<101; iter++){
                if(my_rank == 4){
                    gettimeofday(&start, 0);
                    MPI_Send(message, buf_size, MPI_INT, 5, 1, MPI_COMM_WORLD);
                    MPI_Recv(data, buf_size, MPI_INT, 5, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    gettimeofday(&end, 0);
                    e_usec = ((end.tv_sec * 1000000) + end.tv_usec) - ((start.tv_sec * 1000000) + start.tv_usec);
                    if(iter!=0) intra_node2_arr[iter-1] = e_usec;
                }else if(my_rank == 5){
                    MPI_Recv(data, buf_size, MPI_INT, 4, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(data, buf_size, MPI_INT, 4, 1, MPI_COMM_WORLD);
                }
            }


            //get stats
            if(my_rank < mid){
                if(my_rank < 2) find_stats(inter_node1_arr, 100, &mean, &minn, &maxx, &stddev);
                else if(my_rank == 2) find_stats(intra_node1_arr, 100, &mean, &minn, &maxx, &stddev);
            }else{
                // if(my_rank > 5) find_stats(inter_node2_arr, 100, &mean, &minn, &maxx, &stddev);
                if(my_rank == 4) find_stats(intra_node2_arr, 100, &mean, &minn, &maxx, &stddev);
            }

            double stats_arr[4];
            stats_arr[0] = minn;
            stats_arr[1] = maxx;
            stats_arr[2] = mean;
            stats_arr[3] = stddev;

            sleep(1);

            //Send results to 0 for aggregation
            if(my_rank != 0 && my_rank != 3 && my_rank != 5 && my_rank != 6 && my_rank != 7){
                MPI_Send(stats_arr, 4, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            }else if(my_rank == 0){
                double inter_node1_min=minn, inter_node1_max=maxx, inter_node1_mean=mean, inter_node1_std=stddev;
                double temp[4];
                MPI_Recv(temp, 4, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                inter_node1_min = min(inter_node1_min, temp[0]);
                inter_node1_max = max(inter_node1_max, temp[1]);
                inter_node1_mean += temp[2];
                inter_node1_std += temp[3];
                inter_node1_mean/=2.0;
                inter_node1_std/=2.0;

                double intra_node1_min=100000, intra_node1_max=-100000, intra_node1_mean=0, intra_node1_std=0;
                MPI_Recv(temp, 4, MPI_DOUBLE, 2, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                intra_node1_min = min(intra_node1_min, temp[0]);
                intra_node1_max = max(intra_node1_max, temp[1]);
                intra_node1_mean += temp[2];
                intra_node1_std += temp[3];

                // double inter_node2_min=100000, inter_node2_max=-100000, inter_node2_mean=0, inter_node2_std=0;
                // for(int j = 6; j<=7; j++){
                //     MPI_Recv(temp, 4, MPI_DOUBLE, j, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //     inter_node2_min = min(inter_node2_min, temp[0]);
                //     inter_node2_max = max(inter_node2_max, temp[1]);
                //     inter_node2_mean += temp[2];
                //     inter_node2_std += temp[3];
                // }
                // inter_node2_mean/=(double)2;
                // inter_node2_std/=(double)2;

                double intra_node2_min=100000, intra_node2_max=-100000, intra_node2_mean=0, intra_node2_std=0;
                MPI_Recv(temp, 4, MPI_DOUBLE, 4, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                intra_node2_min = min(intra_node2_min, temp[0]);
                intra_node2_max = max(intra_node2_max, temp[1]);
                intra_node2_mean += temp[2];
                intra_node2_std += temp[3];

                intra_node1_min = min(intra_node1_min, intra_node2_min);
                intra_node1_max = max(intra_node1_max, intra_node2_max);
                intra_node1_mean = (intra_node1_mean + intra_node2_mean)/2;
                intra_node1_std = (intra_node1_std + intra_node2_std)/2;

                printf("MESSAGE_SIZE: %d INTER NODE         min: %lf, max: %lf, mean: %lf, std: %lf\n", s/256, inter_node1_min, inter_node1_max, inter_node1_mean, inter_node1_std);
                printf("MESSAGE_SIZE: %d INTRA NODE         min: %lf, max: %lf, mean: %lf, std: %lf\n", s/256, intra_node1_min, intra_node1_max, intra_node1_mean, intra_node1_std);
                // printf("MESSAGE_SIZE: %d INTER NODE 2         min: %lf, max: %lf, mean: %lf, std: %lf\n", s/256, inter_node2_min, inter_node2_max, inter_node2_mean, inter_node2_std);
                // printf("MESSAGE_SIZE: %d INTRA NODE 2         min: %lf, max: %lf, mean: %lf, std: %lf\n", s/256, intra_node2_min, intra_node2_max, intra_node2_mean, intra_node2_std);
                printf("\n");

            }
        }
        // break;
    }

    //end of program
    MPI_Finalize();
}