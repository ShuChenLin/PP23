#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <iostream>

using namespace std;

#define NUM_THREADS 8
#define CHUNCKSIZE 500
#define sll static_cast<long long>

int main(int argc, char** argv) {
        if (argc != 3) {
                fprintf(stderr, "must provide exactly 2 arguments!\n");
                return 1;
        }
        unsigned long long r = atoll(argv[1]);
        unsigned long long k = atoll(argv[2]);
        unsigned long long pixels = 0;
        unsigned long long square_r = r * r;

#pragma omp parallel num_threads(NUM_THREADS) reduction(+:pixels)
        {

        //if (r > 20000000) omp_set_num_threads(10000);
        int id = omp_get_thread_num(), thread_num = omp_get_num_threads();
        
        //unsigned long long size1 = (r >> 2) << 2;
        unsigned long long temp_pixels[10000];
        #pragma omp for schedule(dynamic, CHUNCKSIZE) nowait
         for (unsigned long long x = 0; x < r; x++) {
             unsigned long long y = ceil(sqrtl(square_r - x*x));
             //pixels += y;
             //pixels %= k;
             temp_pixels[id] += y;
         }

         for (unsigned long long x = 0; x < thread_num; x++) {
            pixels += temp_pixels[x];
         }
         pixels %= k;

        /*for (unsigned long long x = 0; x < size1; x += 2) {
            unsigned long long x_s[2] = {x, x+1};
            unsigned long long square_rs[2] = {square_r, square_r};
            __m128i temp_x = _mm_loadu_si128((__m128i*)x_s);
            __m128i temp_square = _mm_loadu_si128((__m128i*)square_rs);

            __m128i new_x = _mm_mul_epu32(temp_x, temp_x);

            __m128i final_x = _mm_sub_epi64(temp_square, new_x);

            unsigned long long y_s[2];

            _mm_store_si128((__m128i*)y_s, final_x);
            for (int i = 0; i < 2; i++) {
                //cout << y_s[i] << "this is x_s[" << x+i << "]\n";
                pixels += ceil(sqrtl(y_s[i]));
            }
            pixels %= k;

        }

        #pragma omp for schedule(guided, CHUNCKSIZE) nowait

        for (unsigned long long x = size1; x < r; x++) {
            unsigned long long y = ceil(sqrtl(square_r - x*x));
            pixels += y;
            pixels %= k;

        }*/
        }
        printf("%llu\n", (4 * pixels) % k);
}
