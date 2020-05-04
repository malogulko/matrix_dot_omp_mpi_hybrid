//
// Created by Malogulko, Alexey on 26/04/2020.
//

#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdarg.h>
#include "utils.c"

int ROOT_NODE_RANK = 0;

/**
 * @param pos_id - column or row where this node belongs to
 * @param node_rank - this node's global rank
 * @param mpi_comm - pointer to the MPI_Comm structure
 * @param internal_node_rank - this node's rank in the new communicator, e.g. if you're splitting into rows,
 *                             this number will symbolize COLUMN in a ROW
 * @param internal_size - size of the new communicator, e.g if you're splitting into rows,
 *                        this number symbolizes size of the rows
 */
MPI_Comm split_mpi(int pos_id, int node_rank, int *internal_node_rank, int *internal_size) {
    MPI_Comm mpi_comm;
    MPI_Comm_split(MPI_COMM_WORLD, pos_id, node_rank, &mpi_comm);
    MPI_Comm_rank(mpi_comm, internal_node_rank);
    MPI_Comm_size(mpi_comm, internal_size);
    log_info("Splitting MPI_COMM_WORLD, original rank %d, color %d, new rank %d, new size %d\n",
             node_rank, pos_id, *internal_node_rank, *internal_size);
    return mpi_comm;
}

/**
 * Initializes node position in the grid + block size
 */
void init_grid(int matrix_size, int node_rank, int num_nodes, int *block_size, int *node_row, int *node_col) {
    // This number is the size of the square into which the matrix is divided
    int square_size = (int) sqrt(num_nodes);
    // Full length of the stripe of memory for partition
    *block_size = (matrix_size * matrix_size) / num_nodes;
    *node_row = (int) (node_rank / square_size);
    *node_col = node_rank % square_size;
    log_info("Grid initialized, matrix_size %d, node_rank %d, num_nodes %d, block_size %d, row/col %d, %d\n",
             matrix_size, node_rank, num_nodes, *block_size, *node_row, *node_col);
}

/**
 * Initializes Global MPI for this node
 */
void init_mpi(int *argc, char ***argv, int *num_nodes, int *node_rank) {
    MPI_Init(argc, argv);
    MPI_Comm_size(MPI_COMM_WORLD, num_nodes);
    log_info("Initialized MPI_COMM_WORLD for %d nodes\n", *num_nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, node_rank);
    log_info("Initialized MPI_COMM_WORLD for node with rank %d\n", *node_rank);
}

/**
 * Allocates matrix and fills it with random
 */
void matrix_malloc_and_rand(double **matrix, int matrix_size) {
    int memory_size = matrix_size * matrix_size * sizeof(double);
    *matrix = malloc(memory_size);
    random_matrix(*matrix, matrix_size);
}

/**
 * Generates random matrix and distributes it across the grid
 */
void generate_and_distribute_matrix(
        int node_rank,
        int matrix_size,
        double *block,
        int block_size,
        void (*print_fn)(const double *, int, int)
) {
    double *matrix;
    // Only root node knows an entire matrix
    if (node_rank == ROOT_NODE_RANK) {
        matrix_malloc_and_rand(&matrix, matrix_size);
        int block_width = (int) sqrt(block_size);
        print_fn(matrix, matrix_size, block_width);
    }
    // Distributing matrix across the grid
    MPI_Scatter(matrix, block_size, MPI_DOUBLE, block, block_size, MPI_DOUBLE, ROOT_NODE_RANK, MPI_COMM_WORLD);
    // No need to keep memory for the matrix anymore
    if (node_rank == ROOT_NODE_RANK) free(matrix);
}

/**
 *
 *
 * A: 01 02 03 04 represents:
 *
 * 01 02
 * 03 04
 *
 * B: 01 02 03 04 represents:
 *
 * 01 03
 * 02 04
 *
 */
void ijk(const double *matrix_a, const double *matrix_b, double *matrix_c, int matrix_size) {
    for (int i = 0; i < matrix_size; i++) { // Matrix A/C row
        for (int j = 0; j < matrix_size; j++) { // Matrix B/C col
            double *c_sum = matrix_c + i * matrix_size + j;
            for (int k = 0; k < matrix_size; k++) { // A - column, C - row
                *(c_sum) += *(matrix_a + i * matrix_size + k) * *(matrix_b + j * matrix_size + k);
            }
        }
    }
}

/**
 * Because stripes of matrices A and B are already a multiple of blocks, we can use blocking partitioning.
 *
 * @param row_block_a
 * @param col_block_b
 * @param matrix_size
 * @param block_c
 * @param block_size
 */
void local_compute(double *row_block_a, double *col_block_b, int matrix_size, double *block_c, int block_size) {
    int block_width = (int) sqrt(block_size);
    // K is a block number
    for (int k = 0; k < matrix_size / block_width; k += 1) {
        int offset = k * block_size; // Memory offset
        ijk(row_block_a + offset, col_block_b + offset, block_c, block_width);
    }
}

/**
 * The matrix representation is blocked by design, 4x4 matrix for 4-node cluster:
 *
 * 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16
 *
 * Matrices A and C are stored in col-wise format, with col-wise partitioning:
 *
 * 01 03 09 11
 * 02 04 10 12
 * 05 07 13 15
 * 06 08 14 16
 *
 * At the same time, matrix B is stored in row-wise format, with row-wise partitioning
 *
 * 01 02 05 06
 * 03 04 07 08
 * 09 10 13 14
 * 11 12 15 16
 *
 */

int main(int argc, char **argv) {
    int num_nodes, matrix_size, node_rank;
    parse_matrix_size(argc, argv, &matrix_size);
    struct timespec start, end, start_sync, end_sync;

    // Initializes MPI here
    init_mpi(&argc, &argv, &num_nodes, &node_rank);

    // Making sure partitioning is possible
    if (node_rank == ROOT_NODE_RANK) check_partition(matrix_size, num_nodes);

    // Calculating THIS node's position in the grid
    int block_size, node_row, node_col;
    init_grid(matrix_size, node_rank, num_nodes, &block_size, &node_row, &node_col);

    // Communicator for row
    int mpi_col_in_row_rank, mpi_row_size;
    MPI_Comm row_comm = split_mpi(node_row, node_rank, &mpi_col_in_row_rank, &mpi_row_size);

    // Communicator for column
    int mpi_row_in_col_rank, mpi_col_size;
    MPI_Comm col_comm = split_mpi(node_col, node_rank, &mpi_row_in_col_rank, &mpi_col_size);

    // Generating and distributing matrix A
    double *block_a = malloc(block_size * sizeof(double));
    if (node_rank == ROOT_NODE_RANK) log_info("Generating and distributing matrix A:\n");
    generate_and_distribute_matrix(node_rank, matrix_size, block_a, block_size, print_matrix_blocked_rows);
    if (node_rank == ROOT_NODE_RANK) log_info("Matrix A distributed successfully\n");

    // Generating and distributing matrix B
    double *block_b = malloc(block_size * sizeof(double));
    if (node_rank == ROOT_NODE_RANK) log_info("Generating and distributing matrix B:\n");
    generate_and_distribute_matrix(node_rank, matrix_size, block_b, block_size, print_matrix_blocked_cols_in_rows);
    if (node_rank == ROOT_NODE_RANK) log_info("Matrix B distributed successfully\n");

    // Gathering col-block of sub-matrices A column on every node in col
    double *block_row_a = malloc(block_size * mpi_col_size * sizeof(double));
    log_info("Start: Node(global) %d(grid %dx%d) synchronizing matrix A row %d\n", node_rank, node_row, node_col,
             node_col);
    MPI_Allgather(block_a, block_size, MPI_DOUBLE, block_row_a, block_size, MPI_DOUBLE, row_comm);
    free(block_a); // No need to keep memory for single block of A anymore
    log_info("Done: Node(global) %d(grid %dx%d) synchronizing matrix A row %d\n", node_rank, node_row, node_col,
             node_col);
    if (node_rank == ROOT_NODE_RANK) {
        log_info("Root node blocked A row\n");
        print_row_blocked_row(block_row_a, matrix_size, (int) sqrt(block_size));
    }

    // Gathering row-block of sub-matrices B on every node in row
    double *block_col_b = malloc(block_size * mpi_row_size * sizeof(double));
    log_info("Start: Node(global) %d(grid %dx%d) synchronizing matrix B col %d\n", node_rank, node_row, node_col,
             node_col);
    MPI_Allgather(block_b, block_size, MPI_DOUBLE, block_col_b, block_size, MPI_DOUBLE, col_comm);
    free(block_b); // No need to keep memory for single block of B anymore
    log_info("Done: Node(global) %d(grid %dx%d) synchronizing matrix B col %d\n", node_rank, node_row, node_col,
             node_col);
    if (node_rank == ROOT_NODE_RANK) {
        log_info("Root node blocked B col\n");
        print_row_blocked_col(block_col_b, matrix_size, (int) sqrt(block_size));
    }

    double *block_c = malloc_zero_matrix(block_size);
    local_compute(block_row_a, block_col_b, matrix_size, block_c, block_size);
    // No need to keep this in memory anymore
    free(block_row_a);
    free(block_col_b);
    if (node_rank == ROOT_NODE_RANK) {
        log_info("Root node block C\n");
        print_matrix_memory_stripe(block_c, (int) sqrt(block_size), (int) sqrt(block_size));
        print_row_block(block_c, (int) sqrt(block_size));
    }

    double *matrix_c;
    if (node_rank == ROOT_NODE_RANK) matrix_c = malloc(matrix_size * matrix_size * sizeof(double));
    MPI_Gather(block_c, block_size, MPI_DOUBLE, matrix_c, block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    free(block_c);
    if (node_rank == ROOT_NODE_RANK) {
        printf("Matrix C:\n");
        print_matrix_blocked_rows(matrix_c, matrix_size, (int) sqrt(block_size));
    }
    MPI_Finalize();
}