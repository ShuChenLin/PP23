#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


//======================
const int INF = ((1 << 30) - 1);
const int BLOCK = 50;
const int HALFBLOCK = 25;

void input(char* infile, int** Dist, int &n, int &m, int &N) {
    FILE* file = fopen(infile, "rb");
    fread(&N, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    // pad N to multiple BLOCK
    n = (N & (BLOCK-1)) ? ((BLOCK - (N & (BLOCK-1))) + N) : N;

    // allocate memory for Dist
    *Dist = (int*)malloc(n * n * sizeof(int));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                (*Dist)[i * n + j] = 0;
            } else {
                (*Dist)[i * n + j] = INF;
            }
        }
    }

    int *paris = (int*)malloc(m * 3 * sizeof(int));
    fread(paris, sizeof(int), m * 3, file);
    for (int i = 0; i < m; ++i) {
        (*Dist)[paris[i * 3] * n + paris[i * 3 + 1]] = paris[i * 3 + 2];
    }
    fclose(file);
}

void output(char* outFileName, int* Dist, int N, int n) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < N; ++i) {
        fwrite(Dist + i * n, sizeof(int), N, outfile);
    }
    fclose(outfile);
}

__global__ void BFW_phase1(int* d_Dist, size_t n, int r, int offset) {
    int x = threadIdx.y, y = threadIdx.x;
    int x1 = x + HALFBLOCK, y1 = y + HALFBLOCK;

    int i = x + offset, i1 = i + HALFBLOCK;
    int j = y + offset, j1 = j + HALFBLOCK;

    __shared__ int shared_Dist[BLOCK][BLOCK];
    shared_Dist[x][y] = d_Dist[i * n + j];
    shared_Dist[x][y1] = d_Dist[i * n + j1];
    shared_Dist[x1][y] = d_Dist[i1 * n + j];
    shared_Dist[x1][y1] = d_Dist[i1 * n + j1];

#pragma unroll 32
    for (int k = 0; k < BLOCK; ++k) {
        __syncthreads();
        shared_Dist[x][y] = min(shared_Dist[x][y], shared_Dist[x][k] + shared_Dist[k][y]);
        shared_Dist[x][y1] = min(shared_Dist[x][y1], shared_Dist[x][k] + shared_Dist[k][y1]);
        shared_Dist[x1][y] = min(shared_Dist[x1][y], shared_Dist[x1][k] + shared_Dist[k][y]);
        shared_Dist[x1][y1] = min(shared_Dist[x1][y1], shared_Dist[x1][k] + shared_Dist[k][y1]);
    }

    d_Dist[i * n + j] = shared_Dist[x][y];
    d_Dist[i * n + j1] = shared_Dist[x][y1];
    d_Dist[i1 * n + j] = shared_Dist[x1][y];
    d_Dist[i1 * n + j1] = shared_Dist[x1][y1];
}

__global__ void BFW_phase2(int* d_Dist, size_t n, int r, int offset) {
    // should consider grid id
    // if (blockIdx.x == r) return;
    int blockid = blockIdx.x + ((blockIdx.x >= r) ? 1 : 0);

    int x = threadIdx.y, y = threadIdx.x, x1 = x + HALFBLOCK, y1 = y + HALFBLOCK;

    int i = x + (blockid * BLOCK), i1 = i + HALFBLOCK;
    int j = y + offset, j1 = j + HALFBLOCK;

    int a = x + offset, a1 = a + HALFBLOCK;
    int b = y + (blockid * BLOCK), b1 = b + HALFBLOCK;


    __shared__ int shared_Dist1[BLOCK][BLOCK];
    __shared__ int shared_Dist2[BLOCK][BLOCK];
    __shared__ int shared_Dist_pivot[BLOCK][BLOCK];

    shared_Dist1[x][y] = d_Dist[i * n + j];
    shared_Dist1[x][y1] = d_Dist[i * n + j1];
    shared_Dist1[x1][y] = d_Dist[i1 * n + j];
    shared_Dist1[x1][y1] = d_Dist[i1 * n + j1];

    shared_Dist2[x][y] = d_Dist[a * n + b];
    shared_Dist2[x][y1] = d_Dist[a * n + b1];
    shared_Dist2[x1][y] = d_Dist[a1 * n + b];
    shared_Dist2[x1][y1] = d_Dist[a1 * n + b1];

    shared_Dist_pivot[x][y] = d_Dist[a * n + j];
    shared_Dist_pivot[x][y1] = d_Dist[a * n + j1];
    shared_Dist_pivot[x1][y] = d_Dist[a1 * n + j];
    shared_Dist_pivot[x1][y1] = d_Dist[a1 * n + j1];

    __syncthreads();
    
#pragma unroll 32
    for (int k = 0; k < BLOCK; ++k) {
        //__syncthreads();
        shared_Dist1[x][y] = min(shared_Dist1[x][y], shared_Dist1[x][k] + shared_Dist_pivot[k][y]);
        shared_Dist1[x][y1] = min(shared_Dist1[x][y1], shared_Dist1[x][k] + shared_Dist_pivot[k][y1]);
        shared_Dist1[x1][y] = min(shared_Dist1[x1][y], shared_Dist1[x1][k] + shared_Dist_pivot[k][y]);
        shared_Dist1[x1][y1] = min(shared_Dist1[x1][y1], shared_Dist1[x1][k] + shared_Dist_pivot[k][y1]);

        shared_Dist2[x][y] = min(shared_Dist2[x][y], shared_Dist_pivot[x][k] + shared_Dist2[k][y]);
        shared_Dist2[x][y1] = min(shared_Dist2[x][y1], shared_Dist_pivot[x][k] + shared_Dist2[k][y1]);
        shared_Dist2[x1][y] = min(shared_Dist2[x1][y], shared_Dist_pivot[x1][k] + shared_Dist2[k][y]);
        shared_Dist2[x1][y1] = min(shared_Dist2[x1][y1], shared_Dist_pivot[x1][k] + shared_Dist2[k][y1]);
    }

    d_Dist[i * n + j] = shared_Dist1[x][y];
    d_Dist[i * n + j1] = shared_Dist1[x][y1];
    d_Dist[i1 * n + j] = shared_Dist1[x1][y];
    d_Dist[i1 * n + j1] = shared_Dist1[x1][y1];

    d_Dist[a * n + b] = shared_Dist2[x][y];
    d_Dist[a * n + b1] = shared_Dist2[x][y1];
    d_Dist[a1 * n + b] = shared_Dist2[x1][y];
    d_Dist[a1 * n + b1] = shared_Dist2[x1][y1];

}


__global__ void BFW_phase3(int* d_Dist, size_t n, int r, int offset) {
    // if (blockIdx.x == r || blockIdx.y == r) return;

    int blockidx = blockIdx.x + ((blockIdx.x >= r) ? 1 : 0);
    int blockidy = blockIdx.y + ((blockIdx.y >= r) ? 1 : 0);

    int x = threadIdx.y, y = threadIdx.x;
    int x1 = x + HALFBLOCK, y1 = y + HALFBLOCK;

    int i = x + (blockidx * BLOCK);
    int j = y + (blockidy * BLOCK);
    int i1 = i + HALFBLOCK;
    int j1 = j + HALFBLOCK;

    int a = x + offset;
    int a1 = a + HALFBLOCK;
    int d = y + offset;
    int d1 = d + HALFBLOCK;

    __shared__ int shared_Dist_row[BLOCK][BLOCK];
    __shared__ int shared_Dist_col[BLOCK][BLOCK];

    shared_Dist_row[x][y] = d_Dist[a * n + j];
    shared_Dist_row[x][y1] = d_Dist[a * n + j1];
    shared_Dist_row[x1][y] = d_Dist[a1 * n + j];
    shared_Dist_row[x1][y1] = d_Dist[a1 * n + j1];

    shared_Dist_col[x][y] = d_Dist[i * n + d];
    shared_Dist_col[x][y1] = d_Dist[i * n + d1];
    shared_Dist_col[x1][y] = d_Dist[i1 * n + d];
    shared_Dist_col[x1][y1] = d_Dist[i1 * n + d1];

    __syncthreads();

    __shared__ int shared_Dist[BLOCK][BLOCK];
    shared_Dist[x][y] = d_Dist[i * n + j];
    shared_Dist[x][y1] = d_Dist[i * n + j1];
    shared_Dist[x1][y] = d_Dist[i1 * n + j];
    shared_Dist[x1][y1] = d_Dist[i1 * n + j1];

#pragma unroll 32
    for (int k = 0; k < BLOCK; ++k) {
        shared_Dist[x][y] = min(shared_Dist[x][y], shared_Dist_col[x][k] + shared_Dist_row[k][y]);
        shared_Dist[x][y1] = min(shared_Dist[x][y1], shared_Dist_col[x][k] + shared_Dist_row[k][y1]);
        shared_Dist[x1][y] = min(shared_Dist[x1][y], shared_Dist_col[x1][k] + shared_Dist_row[k][y]);
        shared_Dist[x1][y1] = min(shared_Dist[x1][y1], shared_Dist_col[x1][k] + shared_Dist_row[k][y1]);
    }

    d_Dist[i * n + j] = shared_Dist[x][y];
    d_Dist[i * n + j1] = shared_Dist[x][y1];
    d_Dist[i1 * n + j] = shared_Dist[x1][y];
    d_Dist[i1 * n + j1] = shared_Dist[x1][y1];

}



int main(int argc, char* argv[]) {
    int *Dist, *d_Dist;
    // assert(argc == 3);
    int n, m, N;
    input(argv[1], &Dist, n, m, N);

    // cudaGetDeviceProperties(&prop, DEV_NO);
    // printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d", prop.maxThreasPerBlock, prop.sharedMemPerBlock);

    // allocate device memory
    cudaSetDevice(0);
    // cudaMalloc((void**)&d_Dist, n * n * sizeof(int));

    // copy from host to device
    // cudaMemcpy(d_Dist, Dist, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaHostRegister(Dist, n * n * sizeof(int), cudaHostAllocDefault);
    size_t pitch;
    cudaMallocPitch((void**)&d_Dist, &pitch, n * sizeof(int), n);
    cudaMemcpy2D(d_Dist, pitch, Dist, n * sizeof(int), n * sizeof(int), n, cudaMemcpyHostToDevice);

    // set grid size and block size
    int block_num = n / BLOCK;

    dim3 block_size(HALFBLOCK, HALFBLOCK);
    dim3 grid_size1(1, 1);
    dim3 grid_size2(block_num-1, 1);
    dim3 grid_size3(block_num-1, block_num-1);


    for (int r = 0; r < block_num; ++r) {
        int offset = r * BLOCK;
        // BFW_phase1<<<grid_size1 block_size>>>(d_Dist, n, r, BLOCK, offset);
        // BFW_phase2<<<grid_size2, block_size>>>(d_Dist, n, r, BLOCK, offset);
        // BFW_phase3<<<grid_size3, block_size>>>(d_Dist, n, r, BLOCK, offset);
        BFW_phase1<<<grid_size1, block_size>>>(d_Dist, pitch/sizeof(int), r, offset);
        BFW_phase2<<<grid_size2, block_size>>>(d_Dist, pitch/sizeof(int), r, offset);
        BFW_phase3<<<grid_size3, block_size>>>(d_Dist, pitch/sizeof(int), r, offset);
    }

    // copy from device to host
    // cudaMemcpy(Dist, d_Dist, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy2D(Dist, n * sizeof(int), d_Dist, pitch, n * sizeof(int), n, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_Dist);


    //block_FW(B);
    output(argv[2], Dist, N, n);
    return 0;
}