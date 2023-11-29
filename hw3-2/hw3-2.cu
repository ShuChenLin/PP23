#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


//======================
#define DEV_NO 0
#define INF 1e9
cudaDeviceProp prop;

int n, m;  // Number of vertices, edges

int *Dist, *d_Dist;

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    // allocate memory for Dist
    Dist = (int*)malloc(n * n * sizeof(int));

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
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i * n + j] < 0) Dist[i * n + j] = INF;
        }
        fwrite(Dist + i * n, sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }



__global__ void BFW_phase1(int* d_Dist, int n, int r, int block_width, int block_size) {
    int i = threadIdx.x + (r * block_width);
    int j = threadIdx.y + (r * block_width);

    if (i >= n || j >= n) return;

    __shared__ int shared_Dist[32][32];
    shared_Dist[threadIdx.x][threadIdx.y] = d_Dist[i * n + j];
    __syncthreads();

    for (int k = 0; k < block_size; ++k) {
        if (shared_Dist[threadIdx.x][threadIdx.y] > shared_Dist[threadIdx.x][k] + shared_Dist[k][threadIdx.y]) {
            shared_Dist[threadIdx.x][threadIdx.y] = shared_Dist[threadIdx.x][k] + shared_Dist[k][threadIdx.y];
        }
        __syncthreads();
    }

    d_Dist[i * n + j] = shared_Dist[threadIdx.x][threadIdx.y];

}

__global__ void BFW_phase2(int* d_Dist, int n, int r, int block_width, int block_size) {
    // should consider grid id
    if (blockIdx.x == r) return;

    int x1 = threadIdx.x + (r * block_width);
    int y1 = threadIdx.y + (blockIdx.x * block_width);

    int x2 = threadIdx.x + (blockIdx.x * block_width);
    int y2 = threadIdx.y + (r * block_width);

    bool flag1 = (x1 >= n || y1 >= n);
    bool flag2 = (x2 >= n || y2 >= n);

    __shared__ int shared_Dist1[32][32];
    __shared__ int shared_Dist2[32][32];
    shared_Dist1[threadIdx.x][threadIdx.y] = d_Dist[x1 * n + y1];
    shared_Dist2[threadIdx.x][threadIdx.y] = d_Dist[x2 * n + y2];
    __syncthreads();

    for (int i = 0; i < block_size; ++i) {
        if (!flag1 && (shared_Dist1[threadIdx.x][threadIdx.y] > shared_Dist1[threadIdx.x][i] + shared_Dist1[i][threadIdx.y])) {
            shared_Dist1[threadIdx.x][threadIdx.y] = shared_Dist1[threadIdx.x][i] + shared_Dist1[i][threadIdx.y];
        }
        if (!flag2 && (shared_Dist2[threadIdx.x][threadIdx.y] > shared_Dist2[threadIdx.x][i] + shared_Dist2[i][threadIdx.y])) {
            shared_Dist2[threadIdx.x][threadIdx.y] = shared_Dist2[threadIdx.x][i] + shared_Dist2[i][threadIdx.y];
        }
        __syncthreads();
    }

    if (!flag1) d_Dist[x1 * n + y1] = shared_Dist1[threadIdx.x][threadIdx.y];
    if (!flag2) d_Dist[x2 * n + y2] = shared_Dist2[threadIdx.x][threadIdx.y];

}


__global__ void BFW_phase3(int* d_Dist, int n, int r, int block_width, int block_size) {
    if (blockId.x == r || blockId.y == r) return;

    int i = threadIdx.x + (blockIdx.x * block_width);
    int j = threadIdx.y + (blockIdx.y * block_width);

    if (i >= n || j >= n) return;

    __shared__ int shared_Dist[32][32];
    shared_Dist[threadIdx.x][threadIdx.y] = d_Dist[i * n + j];
    __syncthreads();

    for (int k = 0; k < block_size; ++k) {
        if (shared_Dist[threadIdx.x][threadIdx.y] > shared_Dist[threadIdx.x][k] + shared_Dist[k][threadIdx.y]) {
            shared_Dist[threadIdx.x][threadIdx.y] = shared_Dist[threadIdx.x][k] + shared_Dist[k][threadIdx.y];
        }
        __syncthreads();
    }
}



int main(int argc, char* argv[]) {
    assert(argc == 3);
    input(argv[1]);
    int B = 512;

    cudaGetDeviceProperties(&prop, DEV_NO);
    printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d", prop.maxThreasPerBlock, prop.sharedMemPerBlock);

    // use cuda progamming to complete blocked FW algorithm

    // allocate device memory
    int size = n * n * sizeof(int);
    cudaMalloc((void**)&d_Dist, size);

    // copy from host to device
    cudaMemcpy(d_Dist, Dist, size, cudaMemcpyHostToDevice);

    // set grid size and block size
    int block_width = 32;
    int block_num = ceil(n, block_width);

    dim3 block_size(block_width, block_width);
    dim3 grid_size2(block_num, 1);
    dim3 grid_size3(block_num, block_num);


    for (int r = 0; r < block_num; ++r) {
        BFW_phase1<<<1, block_size>>>(d_Dist, n, r, block_width, block_size);
        BFW_phase2<<<grid_size2, block_size>>>(d_Dist, n, r, block_width, block_size);
        BFW_phase3<<<grid_size3, block_size>>>(d_Dist, n, r, block_width, block_size);
    }


    //block_FW(B);
    output(argv[2]);
    return 0;
}