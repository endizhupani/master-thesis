#include "mpi.h"
#include <string>
using namespace std;
int main(int argc, char *argv[])
{
    int proc_num;
    int current_proc;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

    if (proc_num == 1)
    {
        printf("More than 1 processor is needed\n");
        MPI_Finalize();
        exit(0);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &current_proc);
    printf("Number of processors is: %d\n Processor is: %d\n", proc_num, current_proc);

    int manager_proc = 0;
    printf("Starting send process...");
    string text = "Hello world";
    int chunk_size = text.length() / proc_num - 1;

    char received[text.length()];
    if (current_proc != manager_proc)
    {
        int start = (current_proc - 1) * chunk_size;
        int length = chunk_size;
        if (current_proc = proc_num)
        {
            length = text.length() - start - 1;
        }
        char *chunk_to_send;
        strncpy(chunk_to_send, text.substr(current_proc - 1, length).c_str(), length);
        printf("Sending %s", chunk_to_send);
        MPI_Send(chunk_to_send, length, MPI_CHAR, manager_proc, 0, MPI_COMM_WORLD);
    }
    else
    {
        printf("Prepping to receive...");
        char buffer[chunk_size];
        MPI_Status status;
        for (int i = 1; i < proc_num; i++)
        {
            MPI_Recv(buffer, chunk_size, MPI_CHAR, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            for (int j = 0; j < chunk_size; j++)
            {
                printf("received %c on %d from %d\n", buffer[j], current_proc, i);
            }
        }
    }

    // for (int i = 1; i < proc_num; i++)
    // {
    // }

    // printf("Prepping to broadcast message");
    MPI_Finalize();
    return 0;
}