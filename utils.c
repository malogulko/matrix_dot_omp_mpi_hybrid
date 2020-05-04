//
// Created by Malogulko, Alexey on 01/03/2020.
// This is just shared code for other matrix multipliers
//
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

// Enable this to show low-level debug messages
const bool DEBUG_LOGGER_ENABLED = false;
// Enable this to show info messages(e.g. input matrices\output matrices)
const bool INFO_LOGGER_ENABLED = false;
const int SQUARE = 2;
const double NUM_MAX = 10.0;

void log_debug(const char *fmt, ...) {
    if (DEBUG_LOGGER_ENABLED) {
        va_list argptr;
        va_start(argptr, fmt);
        vfprintf(stdout, fmt, argptr);
        va_end(argptr);
    }
}

void log_info(const char *fmt, ...) {
    if (INFO_LOGGER_ENABLED) {
        va_list argptr;
        va_start(argptr, fmt);
        vfprintf(stdout, fmt, argptr);
        va_end(argptr);
    }
}

void check_partition(int matrix_size, int num_partitions) {
    if (fmod((double) matrix_size * matrix_size, num_partitions) != 0.) {
        fprintf(stderr, "matrix must be dividable into square partitions\n");
        exit(1);
    }
}

void parse_matrix_size(int argc, char *argv[], int *size) {
    if (argc >= 2) {
        if (sscanf(argv[1], "%i", size) != 1) {
            fprintf(stderr, "error - not an integer");
            exit(1);
        }
    } else {
        printf("Please use <matrix size>");
        exit(1);
    }
}

void print_matrix_memory_stripe(const double *matrix, int x_size, int y_size) {
    log_debug("Matrix %dx%d memory stripe [\naddress: value\n", x_size, y_size);
    for (int i = 0; i < x_size * y_size; i++) {
        log_debug("%d : %f \n", i, *(matrix + i));
    }
    log_debug("]\n");
}

/**
 * Prints column-wise block
 */
void print_row_block(const double *block, int block_size) {
    for (int i = 0; i < block_size; i++) {
        log_debug("[ ");
        for (int j = 0; j < block_size; j++) {
            log_debug("%f ", *(block + i + j * block_size));
        }
        log_debug("]\n");
    }
}

/**
 * Prints a column of column-wise partitioned matrix(assumes there is only 1 column), e.g. a column of 4x4 for 2x2 block
 * size (effectively 4x2 matrix with tow 2x2 blocks) represented by:
 *
 * 01 02 03 04 05 06 07 08
 *
 * Will print as:
 *
 * 01 03
 * 02 04
 * 05 07
 * 06 08
 *
 */
void print_row_blocked_col(const double *col, int size, int block_size) {
    log_debug("Printing a column of column-wise partitioned matrix of size %dx%d column-wise blocks of size %dx%d\n",
              size, size, block_size, block_size);
    print_matrix_memory_stripe(col, size, block_size);
    for (int i = 0; i < (size / block_size); i++) {
        print_row_block(col + block_size * block_size * i, block_size);
    }
}

/**
 * Prints row of row-wise blocked matrix, e.g a row of 4x4 for 2x2 block size
 * (effectively 2x4 matrix with tow 2x2 blocks) represented by::
 *
 * 01 02 03 04 05 06 07 08
 *
 * Will print as:
 *
 * 01 02 05 06
 * 03 04 07 08
 */

void print_row_blocked_row(const double *row, int size, int block_size) {
    log_debug("Printing a row of row-wise partitioned matrix of size %dx%d col-wise blocks of size %dx%d\n",
              size, size, block_size, block_size);
    print_matrix_memory_stripe(row, size, block_size);
    for (int i = 0; i < (size / block_size); i++) {
        log_debug("[ ");
        for (int j = 0; j < (size / block_size); j++) {
            for (int k = 0; k < (size / block_size); ++k) {
                log_debug("%f ", *(row + i * block_size + j * size + k));
            }
        }
        log_debug("]\n");
    }
}

/**
 * Prints row-wise blocked matrix with row-wise blocks in them, e.g 4x4 matrix for 2x2 block represented by:
 *
 * 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16
 *
 * Will print as:
 *
 * 01 02 05 06
 * 03 04 07 08
 * 09 10 13 14
 * 11 12 15 16
 */
void print_matrix_blocked_rows(const double *matrix, int size, int block_size) {
    log_debug("Printing row-wise partitioned matrix of size %dx%d with row-wise blocks of size %dx%d\n",
              size, size, block_size, block_size);
    print_matrix_memory_stripe(matrix, size, size);
    for (int i = 0; i < (size / block_size); i++) { // Block row
        for (int bi = 0; bi < block_size; bi++) { // In-Block row
            log_info("[ ");
            for (int j = 0; j < (size / block_size); j++) { // Block column
                for (int bj = 0; bj < block_size; bj++) { // In-Block column
                    log_info("%f ",
                             *(matrix + j * block_size * block_size + i * size * block_size + bi * block_size + bj));
                }
            }
            log_info("]\n");
        }
    }
}

void set_start(struct timespec *start_time) {
    clock_gettime(CLOCK_MONOTONIC_RAW, start_time);
}

void add_time(struct timespec start_time, uint64_t *time_microseconds) {
    struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_time);
    *time_microseconds = +(end_time.tv_sec - start_time.tv_sec) * 1000000 +
                         (end_time.tv_nsec - start_time.tv_nsec) / 1000; // microseconds
}


/**
 * Prints column-wise blocked matrix, with column-wise blocks, e.g 4x4 matrix for 2x2 block represented by:
 *
 * 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16
 *
 * Will print as:
 *
 * 01 03 05 07
 * 02 04 06 08
 * 09 11 13 15
 * 10 12 14 16
**/
void print_matrix_blocked_cols_in_rows(const double *matrix, int size, int block_size) {
    log_debug("Printing row-wise partitioned matrix of size %dx%d with row-wise blocks of size %dx%d\n",
              size, size, block_size, block_size);
    print_matrix_memory_stripe(matrix, size, size);
    for (int i = 0; i < (size / block_size); i++) { // Block row
        for (int bi = 0; bi < block_size; bi++) { // In-Block row
            log_info("[ ");
            for (int j = 0; j < (size / block_size); j++) { // Block column
                for (int bj = 0; bj < block_size; bj++) { // In-Block column
                    log_info("%f ",
                             *(matrix + bj * block_size + i * size * block_size + bi + j * block_size * block_size));
                }
            }
            log_info("]\n");
        }
    }
}

void random_matrix(double *matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        *(matrix + i) = (double) i + 1;
    }
}

/**
 * Initializes memory for the matrix and fills it with zeros
 * @param size - total number of elements in the matrix
 */
double *malloc_zero_matrix(int size) {
    double *matrix = malloc(size * sizeof(double));
    for (int i = 0; i < size; ++i) {
        *(matrix + i) = 0;
    }
    return matrix;
}

/**
 * Allocates matrix and fills it with random
 */
void matrix_malloc_and_rand(double **matrix, int matrix_size) {
    int memory_size = matrix_size * matrix_size * sizeof(double);
    *matrix = malloc(memory_size);
    random_matrix(*matrix, matrix_size);
}
