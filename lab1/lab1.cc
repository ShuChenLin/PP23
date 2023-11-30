#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
typedef unsigned long long ll;

int main(int argc, char** argv) {

    // Init
    int ret = MPI_Init(&argc, &argv);
    if (ret != MPI_SUCCESS) {
        printf("Error startingMPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, ret);
    }
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    //

	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
    
	ll r = atoll(argv[1]);
	ll k = atoll(argv[2]);
	ll pixels = 0;
    ll pixels_total = 0;

    ll task_size = r / size;
    ll task_start, task_end;
    if (rank == 0) task_start = 0, task_end = task_size + (r % size);
    else task_start = rank * task_size + (r % size), task_end = task_start + task_size;

	for (ll x = task_start; x < task_end; x++) {
		ll y = ceil(sqrtl(r*r - x*x));
		pixels += y;
		pixels %= k;
	}

    MPI_Reduce(&pixels, &pixels_total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("%llu\n", (4 * pixels_total) % k);
    }

    MPI_Finalize();
}
