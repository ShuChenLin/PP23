#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	MPI_Init(&argc, &argv);
	int mpi_rank, mpi_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);


	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
    unsigned long long square_r = r * r;

	unsigned long long task_size = r / mpi_size;
	unsigned long long task_remain = r % mpi_size;
	unsigned long long start = mpi_rank * task_size + min(mpi_rank * 1ULL, task_remain);
	unsigned long long end = start + task_size + (mpi_rank < task_remain ? 1 : 0);

#pragma omp NUM_THREADS(4) reduction(+:pixels)
	{

    //if (r > 500000000) omp_set_num_threads(5000);
		
	#pragma omp parallel for reduction(+:pixels)
		for (unsigned long long x = start; x < end; x++) {
			unsigned long long y = ceil(sqrtl(square_r - x*x));
			pixels += y;
			//if (pixels > k) pixels %= k;
		}

	}

	if (mpi_rank == 0) {
		for (int i = 1; i < mpi_size; i++) {
			unsigned long long recv_pixels;
			MPI_Recv(&recv_pixels, 1, MPI_UNSIGNED_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			pixels += recv_pixels;
		}
        pixels %= k;
        printf("%llu\n", (4 * pixels) % k);
	} else {
		MPI_Send(&pixels, 1, MPI_UNSIGNED_LONG_LONG, 0, 0, MPI_COMM_WORLD);
	}
	MPI_Finalize();
}
