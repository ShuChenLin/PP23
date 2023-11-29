#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <pmmintrin.h>

void write_png(const char*, int, int, int, const int*);

typedef struct {
    double y_lu, x_rl, lower, left;
    int width, height, ncpus, start, iters, *image;
} args_t;

void* mandelbrot(void* arg) {
    args_t* args = (args_t*) arg;
    int start = args->start, height = args->height, width = args->width, iters = args->iters, ncpus = args->ncpus;
    double y_lu = args->y_lu, x_rl = args->x_rl, lower = args->lower, left = args->left;

    __m128d two_sse = _mm_set1_pd(2.0);
    __m128d zero_sse = _mm_set1_pd(0.0);
    __m128i one_sse = _mm_set1_epi32(1);

    for (int j = start; j < height; j += ncpus) {
        double y0 = j * y_lu + lower;
        int new_j = j * width;
        for (int i = 0; i < width-1; i += 2) {
            double x0 = i * x_rl + left;
            double x1 = (i+1) * x_rl + left;

            __m128d x0_sse = _mm_set_pd(x1, x0);
            __m128d y0_sse = _mm_set1_pd(y0);
            __m128d x_sse = _mm_set1_pd(0.0);
            __m128d y_sse = _mm_set1_pd(0.0);
            __m128d length_squared_sse = _mm_set1_pd(0.0);
            __m128i repeats_sse = _mm_set1_epi32(0);
            
            int done[2] = {0, 0}, final_repeats[2], repeats[4];

            while (!done[0] || !done[1]) {
                __m128d temp_sse = _mm_sub_pd(_mm_mul_pd(x_sse, x_sse), _mm_mul_pd(y_sse, y_sse));
                y_sse = _mm_add_pd(_mm_mul_pd(two_sse, _mm_mul_pd(x_sse, y_sse)), y0_sse);
                x_sse = _mm_add_pd(temp_sse, x0_sse);
                length_squared_sse = _mm_add_pd(_mm_mul_pd(x_sse, x_sse), _mm_mul_pd(y_sse, y_sse));
                repeats_sse = _mm_add_epi64(repeats_sse, one_sse);

                //store repeats_sse into repeats
                _mm_storeu_si128((__m128i*)repeats, repeats_sse);
                if (!done[0]) {
                    if (repeats[0] >= iters || length_squared_sse[0] >= 4.0) {
                        //printf("0 repeats %d %lf\n", repeats[0], length_squared_sse[0]);
                        final_repeats[0] = repeats[0];
                        done[0] = 1;
                    }
                }

                if (!done[1]) {
                    if (repeats[1] >= iters || length_squared_sse[1] >= 4.0) {
                        //printf("1 repeats %d %lf\n", repeats[1], length_squared_sse[0]);
                        final_repeats[1] = repeats[1];
                        done[1] = 1;
                    }
                }

            }

            args->image[new_j + i] = final_repeats[0];
            args->image[new_j + i + 1] = final_repeats[1];
            //printf("hello\n");
        }
        //printf("bye\n");

        if (width & 1) {

            double x0 = (width - 1) * x_rl + left;
            int repeats = 0;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            while (repeats < iters && length_squared < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            args->image[new_j + width - 1] = repeats;

        }
    }

    return NULL;
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int ncpus = CPU_COUNT(&cpu_set);
    printf("%d cpus available\n", ncpus);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    /* mandelbrot set */
    pthread_t threads[ncpus];
    args_t arg[ncpus];
    int start = 0;
    double y_ul = ((upper - lower) / height);
    double x_rl = ((right - left) / width);

    for (int i = 0; i < ncpus; ++i) {
        arg[i].lower = lower;
        arg[i].left = left;
        arg[i].y_lu = y_ul;
        arg[i].x_rl = x_rl;
        arg[i].width = width;
        arg[i].height = height;
        arg[i].start = start + i;
        arg[i].iters = iters;
        arg[i].image = image;
        arg[i].ncpus = ncpus;
        pthread_create(&threads[i], NULL, mandelbrot, (void*)&arg[i]);
    }

    for (int i = 0; i < ncpus; ++i) {
        pthread_join(threads[i], NULL);
    }

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);

    for (int y = height-1; y >= 0; --y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[y * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}
