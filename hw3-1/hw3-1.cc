#include <assert.h>
#include <stdio.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <omp.h>

using namespace std;

#define INF 2147483647
#define CHUNCKSIZE 1

int Dist[6005][6005];

int main(int argc, char** argv) {
    cout << "just start\n";
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int ncpus = CPU_COUNT(&cpu_set);

    //input file and output file
    assert(argc == 3);
    const char* infile = argv[1];
    const char* outfile = argv[2];

    cout << "input and output file\n";

    int n, m;

    //read input file
    FILE* fp = fopen(infile, "r");
    cout << "first check\n";
    fread(&n, sizeof(int), 1, fp);
    fread(&m, sizeof(int), 1, fp);

    cout << "this is n and m " << n << " " << m << "\n";

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) Dist[i][j] = 0;
            else Dist[i][j] = INF;
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, fp);
        cout << "pair " << pair[0] << " " << pair[1] << " " << pair[2] << "\n";
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(fp);

    cout << "hello\n";

    //blocked floyd warshall

    int block_size = ncpus, rounds = ceil(n / block_size);

    #pragma omp parallel num_threads(ncpus)
    {
        for (int k = 0; k < rounds; k++) {
            int start = k * block_size;
            int end = (k == rounds - 1) ? n : start + block_size;

#pragma omp for schedule(static, CHUNKSIZE) nowait
            // phase 1: pivot block, on diagonal, self-updata
            for (int h = start; h < end; h++) {
                for (int i = start; i < end; i++) {
                    for (int j = start; j < end; j++) {
                        Dist[i][j] = min(Dist[i][j], Dist[i][h] + Dist[h][j]);
                    }
                }
            }

#pragma omp barrier // Wait for Phase 1 to finish before proceeding to Phase 2

#pragma omp for schedule(static, CHUNKSIZE) nowait
            // phase 2: blocks with same row and column as pivot block, update by pivot block
            for (int i = 0; i < rounds; i++) {
                if (i != k) {
                    int start_i = i * block_size;
                    int end_i = (i == rounds - 1) ? n : start_i + block_size;
                    for (int j = start; j < end; j++) {
                        for (int h = start_i; h < end_i; h++) {
                            for (int l = start; l < end; l++) {
                                Dist[h][l] = min(Dist[h][l], Dist[h][j] + Dist[j][l]);
                                Dist[l][h] = min(Dist[l][h], Dist[l][j] + Dist[j][h]);
                            }
                        }
                    }
                }
            }

#pragma omp barrier // Wait for Phase 2 to finish before proceeding to Phase 3

#pragma omp for schedule(static, CHUNKSIZE) nowait
            // phase 3: remain blocks, update by blocks in phase 2
            for (int i = 0; i < rounds; i++) {
                if (i != k) {
                    int start_i = i * block_size;
                    int end_i = (i == rounds - 1) ? n : start_i + block_size;
                    for (int j = 0; j < rounds; j++) {
                        if (j != k) {
                            int start_j = j * block_size;
                            int end_j = (j == rounds - 1) ? n : start_j + block_size;
                            for (int h = start_i; h < end_i; h++) {
                                for (int l = start_j; l < end_j; l++) {
                                    for (int m = start; m < end; m++) {
                                        Dist[h][l] = min(Dist[h][l], Dist[h][m] + Dist[m][l]);
                                        Dist[l][h] = min(Dist[l][h], Dist[l][m] + Dist[m][h]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
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
