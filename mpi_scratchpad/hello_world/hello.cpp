#include "mpi.h"
#include <string>
#include <iostream>
using namespace std;
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int proc_num;
    int current_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

    MPI_Status status;
    if (proc_num == 1)
    {
        printf("More than 1 processor is needed\n");
        MPI_Finalize();
        exit(0);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &current_proc);
    printf("Number of processors is: %d\n", proc_num);

    int manager_proc = 0;
    printf("Starting send process...\n");
    string text = "Hello world";
    int chunk_size = text.length() / (proc_num - 1);

    if (current_proc == manager_proc)
    {
        printf("ChunkSize = %d\n", chunk_size);
    }
    char confirmation[5];
    if (current_proc != manager_proc)
    {
        int start = (current_proc - 1) * chunk_size;
        int length = chunk_size;
        if (current_proc == (proc_num - 1))
        {
            length = text.length() - start;
        }
        //char chunk_to_send[length];
        printf("Sending %d characters from position %d from processor %d\n", length, start, current_proc);
        string chunk_to_send = text.substr(start, length);
        printf("Chunk to send is %s\n", chunk_to_send.c_str());

        MPI_Send(chunk_to_send.c_str(), chunk_to_send.size(), MPI_CHAR, manager_proc, current_proc, MPI_COMM_WORLD);

        MPI_Bcast(confirmation, 5, MPI_CHAR, manager_proc, MPI_COMM_WORLD);
        printf("Confirmation received on %d. Message: %s\n", current_proc, confirmation);
    }
    else
    {
        int received_length = text.length() + 1;
        char received[received_length];
        printf("Prepping to receive...\n");
        // int lengthOfLastChunk = text.length() - ((proc_num - 2) * chunk_size);
        // int maxBufferLength = max(chunk_size, lengthOfLastChunk);
        // char buffer[chunk_size];

        for (int i = 1; i < proc_num; i++)
        {
            MPI_Status status;
            printf("Receiving from %d...\n", i);
            int message_length;
            MPI_Probe(i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_CHAR, &message_length);
            printf("Expecting message with length: %d\n", message_length);
            int start = (i - 1) * chunk_size;

            char *buffer = (char *)malloc(sizeof(char) * (message_length + 1));
            MPI_Recv(buffer, message_length, MPI_CHAR, i, i, MPI_COMM_WORLD, &status);
            buffer[message_length] = '\0';
            printf("Received %s from %d...\n", buffer, i);
            for (int j = 0; j < message_length; j++)
            {
                received[start + j] = buffer[j];
            }
            free(buffer);
        }

        received[received_length - 1] = '\0';

        printf("OK, got it, I was told to say: %s\n", received);
        printf("Prepping to broadcast receival confirmation\n");
        char confirmation[] = "Done\0";
        MPI_Bcast(confirmation, 5, MPI_CHAR, current_proc, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}