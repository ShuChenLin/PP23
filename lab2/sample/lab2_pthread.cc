#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

typedef struct {
	unsigned long long r;
	unsigned long long start;
	unsigned long long end;
	unsigned long long k;
	unsigned long long pixels;
} args_t;

void* calculate(void* arg) {
	args_t* args = (args_t*) arg;
    unsigned long long pixels = 0, k = args->k, r = args->r * args->r;
	for (unsigned long long x = args->start; x < args->end; x++) {
		unsigned long long y = ceil(sqrtl(r - x*x));
        pixels += y;
        //if (pixels > k) pixels %= k;
        //pixels %= k;
	}
    args->pixels = pixels;
	return NULL;
}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long long ncpus = CPU_COUNT(&cpuset);

	pthread_t threads[8000];
	args_t tasks[8000];

	unsigned long long start_temp = 0, end_temp;
    unsigned long long temp_ncpus = ncpus;
	unsigned long long size_temp = r / temp_ncpus;
	unsigned long long more = r % temp_ncpus;

	
	for (unsigned long long x = 0; x < temp_ncpus; x++) {
        if (x >= more && size_temp == 0) break;
		tasks[x].r = r;
		tasks[x].start = start_temp;
		tasks[x].end = start_temp + size_temp + (x < more ? 1 : 0);
        start_temp = tasks[x].end;
		tasks[x].k = k;
		tasks[x].pixels = 0;
		pthread_create(&threads[x], NULL, calculate, &tasks[x]);
	}

	for (unsigned long long x = 0; x < temp_ncpus; x++) {
		pthread_join(threads[x], NULL);
		pixels += tasks[x].pixels;
	}
    pixels %= k;

	printf("%llu\n", (4 * pixels) % k);
}

