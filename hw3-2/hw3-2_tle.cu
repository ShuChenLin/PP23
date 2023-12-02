#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


//======================
#define DEV_NO 0
#define INF ((1 << 30) - 1)
cudaDeviceProp prop;

int n, m, N;  // Number of vertices, edges

int *Dist, *d_Dist;

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&N, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    // pad N to multiple 32
    n = (32 - (N % 32)) + N;

    // allocate memory for Dist
    int size = n * n * (sizeof(int));
    cudaMallocHost((void**)&Dist, size);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i * n + j] = 0;
            } else {
                Dist[i * n + j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * n + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < N; ++i) {
        // for (int j = 0; j < n; ++j) {
        //     if (Dist[i * n + j] < 0) Dist[i * n + j] = INF;
        // }
        fwrite(Dist + i * n, sizeof(int), N, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }



__global__ void BFW_phase1(int* d_Dist, int n, int r, int block_width) {
    int x = threadIdx.x, y = threadIdx.y;

    int i = x + (r * block_width);
    int j = y + (r * block_width);

    if (i >= n || j >= n) return;

    __shared__ int shared_Dist[32][32];
    shared_Dist[x][y] = d_Dist[i * n + j];
#pragma unroll 32
    for (int k = 0; k < block_width; ++k) {
        __syncthreads();
        if (shared_Dist[x][y] > shared_Dist[x][k] + shared_Dist[k][y]) {
            shared_Dist[x][y] = shared_Dist[x][k] + shared_Dist[k][y];
        }
    }

    d_Dist[i * n + j] = shared_Dist[x][y];
    __syncthreads();
}

__global__ void BFW_phase2(int* d_Dist, int n, int r, int block_width) {
    // should consider grid id
    if (blockIdx.x == r) return;

    int x = threadIdx.x, y = threadIdx.y;
    int i, j, temp_x, temp_y;
    if (blockIdx.y == 0) {
        i = x + (r * block_width);
        j = y + (blockIdx.x * block_width);
        temp_x = i;
        temp_y = y + (r * block_width);
    } else {
        i = x + (blockIdx.x * block_width);
        j = y + (r * block_width);
        temp_x = x + (r * block_width);
        temp_y = j;
    }

    if (i >= n || j >= n) return;
    bool flag = (temp_x >= n || temp_y >= n);

    __shared__ int shared_Dist[32][32];
    __shared__ int shared_Dist_pivot[32][32];

    shared_Dist[x][y] = d_Dist[i * n + j];
    if (!flag) shared_Dist_pivot[x][y] = d_Dist[temp_x * n + temp_y];

#pragma unroll 32
    for (int k = 0; k < block_width; ++k) {
        __syncthreads();
        if (blockIdx.y == 0) {
            if (shared_Dist[x][y] > shared_Dist_pivot[x][k] + shared_Dist[k][y]) {
                shared_Dist[x][y] = shared_Dist_pivot[x][k] + shared_Dist[k][y];
            }
        } else {
            if (shared_Dist[x][y] > shared_Dist[x][k] + shared_Dist_pivot[k][y]) {
                shared_Dist[x][y] = shared_Dist[x][k] + shared_Dist_pivot[k][y];
            }
        }
    }

    d_Dist[i * n + j] = shared_Dist[x][y];

    __syncthreads();
}


__global__ void BFW_phase3(int* d_Dist, int n, int r, int block_width) {
    if (blockIdx.x == r || blockIdx.y == r) return;

    int x = threadIdx.x, y = threadIdx.y;

    int i = x + (blockIdx.x * block_width);
    int j = y + (blockIdx.y * block_width);

    int x1 = x + (r * block_width);
    int y1 = y + (blockIdx.y * block_width);

    int x2 = x + (blockIdx.x * block_width);
    int y2 = y + (r * block_width);

    if (i >= n || j >= n) return;
    bool flag1 = (x1 >= n || y1 >= n);
    bool flag2 = (x2 >= n || y2 >= n);

    __shared__ int shared_Dist[32][32];
    __shared__ int shared_Dist_row[32][32];
    __shared__ int shared_Dist_col[32][32];

    shared_Dist[x][y] = d_Dist[i * n + j];
    if (!flag1) shared_Dist_row[x][y] = d_Dist[x1 * n + y1];
    if (!flag2) shared_Dist_col[x][y] = d_Dist[x2 * n + y2];
    __syncthreads();
#pragma unroll 32
    for (int k = 0; k < block_width; ++k) {
        if (shared_Dist[x][y] > shared_Dist_col[x][k] + shared_Dist_row[k][y]) {
            shared_Dist[x][y] = shared_Dist_col[x][k] + shared_Dist_row[k][y];
        }
    }

    d_Dist[i * n + j] = shared_Dist[x][y];

    __syncthreads();
}



int main(int argc, char* argv[]) {
    // assert(argc == 3);
    input(argv[1]);

    // cudaGetDeviceProperties(&prop, DEV_NO);
    // printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d", prop.maxThreasPerBlock, prop.sharedMemPerBlock);

    // use cuda progamming to complete blocked FW algorithm

    // allocate device memory
    int size = n * n * sizeof(int);
    cudaMalloc((void**)&d_Dist, size);

    // copy from host to device
    cudaMemcpy(d_Dist, Dist, size, cudaMemcpyHostToDevice);

    // set grid size and block size
    int block_width = 32;
    int block_num = n / 32;

    dim3 block_size(block_width, block_width);
    dim3 grid_size2(block_num, 2);
    dim3 grid_size3(block_num, block_num);


    for (int r = 0; r < block_num; ++r) {

        BFW_phase1<<<1, block_size>>>(d_Dist, n, r, block_width);
        BFW_phase2<<<grid_size2, block_size>>>(d_Dist, n, r, block_width);
        BFW_phase3<<<grid_size3, block_size>>>(d_Dist, n, r, block_width);
    }

    // copy from device to host
    cudaMemcpy(Dist, d_Dist, size, cudaMemcpyDeviceToHost);


    //block_FW(B);
    output(argv[2]);
    return 0;
}