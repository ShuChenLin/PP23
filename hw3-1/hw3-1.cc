#include <assert.h>
#include <stdio.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <omp.h>

using namespace std;

#define INF ((1 << 30) -1)
#define CHUNKSIZE 5

int Dist[6005][6005];

int main(int argc, char** argv) {
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int ncpus = CPU_COUNT(&cpu_set);

    //input file and output file
    assert(argc == 3);
    const char* infile = argv[1];
    const char* outfile = argv[2];


    int n, m;

    //read input file
    FILE* fp = fopen(infile, "r");
    fread(&n, sizeof(int), 1, fp);
    fread(&m, sizeof(int), 1, fp);


#pragma omp parallel num_threads(ncpus)
{
    #pragma omp for schedule(dynamic, CHUNKSIZE)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) Dist[i][j] = 0;
            else Dist[i][j] = INF;
        }
    }
}

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, fp);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(fp);


    //blocked floyd warshall

    int block_size = ncpus, rounds = (n + block_size - 1) / (block_size);

    #pragma omp parallel num_threads(ncpus)
    {
        for (int k = 0; k < rounds; ++k) {
            int start = k * block_size;
            int end = min(n, start + block_size);

            // phase 1: pivot block, on diagonal, self-updata
            for (int h = start; h < end; ++h) {
                #pragma omp for schedule(dynamic, CHUNKSIZE)
                for (int i = start; i < end; ++i) {
                    if (Dist[i][h] == INF) continue;
                    for (int j = start; j < end; ++j) {
                        if (Dist[h][j] == INF) continue;
                        Dist[i][j] = min(Dist[i][j], Dist[i][h] + Dist[h][j]);
                    }
                }
            }

//#pragma omp barrier // Wait for Phase 1 to finish before proceeding to Phase 2

            // phase 2: blocks with same row and column as pivot block, update by pivot block
            #pragma omp for schedule(dynamic, CHUNKSIZE)
            for (int i = 0; i < rounds; ++i) {
                if (i != k) {
                    int start_i = i * block_size, end_i = min(n, start_i + block_size);
                    for (int m = start; m < end; ++m) {
                        for (int h = start; h < end; ++h) {
                            if (Dist[h][m] != INF) {
                                for (int l = start_i; l < end_i; ++l) {
                                    if (Dist[m][l] == INF) continue;
                                    Dist[h][l] = min(Dist[h][l], Dist[h][m] + Dist[m][l]);
                                }
                            }

                            if (Dist[m][h] != INF) {
                                for (int l = start_i; l < end_i; ++l) {
                                    if (Dist[l][m] == INF) continue;
                                    Dist[l][h] = min(Dist[l][h], Dist[l][m] + Dist[m][h]);
                                }
                            }
                        }
                    }
                }
            }

//#pragma omp barrier // Wait for Phase 2 to finish before proceeding to Phase 3

            // phase 3: remain blocks, update by blocks in phase 2
            #pragma omp for schedule(dynamic, CHUNKSIZE)
            for (int i = 0; i < rounds; ++i) {
                for (int j = 0; j < rounds; ++j) {
                    if (i != k && j != k) {
                        int start_i = i * block_size, start_j = j * block_size;
                        int end_i = min(n, start_i + block_size), end_j = min(n, start_j + block_size);
                        for (int m = start; m < end; ++m) {
                            for (int h = start_i; h < end_i; ++h) {
                                if (Dist[h][m] == INF) continue;
                                for (int l = start_j; l < end_j; ++l) {
                                    if (Dist[m][l] == INF) continue;
                                    Dist[h][l] = min(Dist[h][l], Dist[h][m] + Dist[m][l]);
                                }
                            }
                        }
                    }
                }
            }
//#pragma omp barrier
        }
    }

    //write output file

    fp = fopen(outfile, "w");

    for (int i = 0; i < n; ++i) {
        fwrite(Dist[i], sizeof(int), n, fp);
    }

    fclose(fp);

    return 0;

}