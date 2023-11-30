#include <cstdio>
#include <cstdlib>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <assert.h>
#include <boost/sort/spreadsort/float_sort.hpp>
#define CAST_TYPE int
#define DATA_TYPE float

using namespace std;
using namespace boost::sort::spreadsort;

struct rightshift{
inline CAST_TYPE operator()(const DATA_TYPE &x, const unsigned offset) const {
    return float_mem_cast<DATA_TYPE, CAST_TYPE>(x) >> offset;
  }
};

struct lessthan { //greaterthan function is used to sort in descending order
  bool operator()(const DATA_TYPE &x, const DATA_TYPE &y) const {
    return x < y;
  }
};

signed cmp(const void *a, const void *b) {
    float fa = *(const float*) a;
    float fb = *(const float*) b;
    return (fa > fb) - (fa < fb);
}

signed main(signed argc, char **argv) {

    MPI_Init(&argc, &argv);

    signed rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = atoll(argv[1]);
    char *input_filename = argv[2];
    char *output_filename = argv[3];

    int additional_size = n % size, normal_size = n / size;;
    // task size for rank n-1, n, n+1
    int task_size_pre = (size > n) ? 1 : (rank - 1 < additional_size) ? normal_size + 1 : normal_size;
    int task_size = (size > n) ? 1 : ((rank < additional_size) ? normal_size + 1 : normal_size);
    int task_size_post = (size > n) ? 1 : (rank + 1 < additional_size) ? normal_size + 1 : normal_size;

    int start_pos = (size > n) ? (rank * task_size) : ((rank) ? ((rank * normal_size) + min(additional_size, rank)) : 0);
    int end_pos = start_pos + task_size - 1;

    if (rank >= n || rank >= size) task_size = 0;

    MPI_File input_file, output_file;

    float* Data = new float[task_size+5];
    float* temp_Data = new float[task_size+5];
    float* recv_Data = new float [task_size+5];
    
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    if (rank < n && rank < size) MPI_File_read_at(input_file, sizeof(float) * start_pos, Data, task_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    // sort each task first
    if (task_size > 1) boost::sort::spreadsort::float_sort(Data, Data + task_size);

    int sorted = 0, global_sorted = 0;

    if (size > n) size = n;

    //sent part of the data to decrease the send recv time
    int numerator = 5, denominator = 10;
    int task_size_send = max(1, (normal_size * numerator) / denominator);
    int recv_size = max(1, (normal_size * numerator) / denominator);

    int tt = (size / 2) + 1;
    while(!global_sorted) {
        sorted = 1, global_sorted = 0;
        
        // odd phase
        if (rank < n && rank < size) {
        if (rank & 1) {
            MPI_Sendrecv(Data, task_size_send, MPI_FLOAT, rank - 1, 0, recv_Data, recv_size, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (Data[0] < recv_Data[recv_size-1]) {
                sorted = 0;
                int i = task_size-1, j = recv_size-1, k = 0;
                while (i >= 0 && j >= 0 && k < task_size) {
                    if (Data[i] > recv_Data[j])
                        temp_Data[k++] = Data[i--];
                    else 
                        temp_Data[k++] = recv_Data[j--];
                }
                while (i >= 0 && k < task_size)
                    temp_Data[k++] = Data[i--];
                while (j >= 0 && k < task_size)
                    temp_Data[k++] = recv_Data[j--];
                for (int i = 0; i < task_size; ++i) 
                    Data[task_size-i-1] = temp_Data[i];
            }
        } 

        if (!(rank & 1) && (rank < size - 1)) {
            MPI_Sendrecv(Data + task_size - task_size_send, task_size_send, MPI_FLOAT, rank + 1, 0, recv_Data, recv_size, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (recv_Data[0] < Data[task_size - 1]) {
                sorted = 0;
                // merge Data and recv_Data
                // fiint Data first and then ret_Data
                int i = 0, j = 0, k = 0;
                while (i < task_size && j < recv_size&& k < task_size) {
                    if (Data[i] < recv_Data[j])
                        temp_Data[k++] = Data[i++];
                    else
                        temp_Data[k++] = recv_Data[j++];
                }
                while (i < task_size && k < task_size)
                    temp_Data[k++] = Data[i++];
                while (j < recv_size&& k < task_size)
                    temp_Data[k++] = recv_Data[j++];
                for (int i = 0; i < task_size; ++i)
                    Data[i] = temp_Data[i];
            }
        }


        // even phase
        if (rank && !(rank & 1)) {
            MPI_Sendrecv(Data, task_size_send, MPI_FLOAT, rank - 1, 0, recv_Data, recv_size, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (Data[0] < recv_Data[recv_size-1]) {
                sorted = 0;
                int i = task_size-1, j = recv_size-1, k = 0;
                while (i >= 0 && j >= 0 && k < task_size) {
                    if (Data[i] > recv_Data[j])
                        temp_Data[k++] = Data[i--];
                    else 
                        temp_Data[k++] = recv_Data[j--];
                }
                while (i >= 0 && k < task_size)
                    temp_Data[k++] = Data[i--];
                while (j >= 0 && k < task_size)
                    temp_Data[k++] = recv_Data[j--];
                for (int i = 0; i < task_size; ++i) 
                    Data[task_size-i-1] = temp_Data[i];
            }
        }

        if ((rank & 1) && (rank < size - 1)) {

            MPI_Sendrecv(Data + task_size - task_size_send, task_size_send, MPI_FLOAT, rank + 1, 0, recv_Data, recv_size, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (recv_Data[0] < Data[task_size-1]) {
                sorted = 0;
                // merge Data and recv_Data
                int i = 0, j = 0, k = 0;
                while (i < task_size && j < recv_size && k < task_size) {
                    if (Data[i] < recv_Data[j])
                        temp_Data[k++] = Data[i++];
                    else
                        temp_Data[k++] = recv_Data[j++];
                }
                while (i < task_size && k < task_size)
                    temp_Data[k++] = Data[i++];
                while (j < recv_size && k < task_size)
                    temp_Data[k++] = recv_Data[j++];
                for (int i = 0; i < task_size; ++i)
                    Data[i] = temp_Data[i];
            }
        }
        }

        MPI_Allreduce(&sorted, &global_sorted, 1, MPI_INT, MPI_BAND, MPI_COMM_WORLD);
        
    }

    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    if (rank < n && rank < size) MPI_File_write_at(output_file, sizeof(float) * start_pos, Data, task_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    MPI_Finalize();
    return 0;
}
