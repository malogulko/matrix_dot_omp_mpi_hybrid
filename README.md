## What?

This program is an example of a hybrid MPI+OpenMP matrix multiplication algorithm. In this particular implementation, 
MPI node get split into grid, where every block of the grid can be mapped to a block of the resulting matrix. OpenMP here
is only used for local computations, spawning <number of blocks in row/col> number of threads.

E.g. Imagine calculating product of two 4x4 matrices:

```
01 02 03 04          01 02 03 04
05 06 07 08    \/    05 06 07 08
09 10 11 12    /\    09 10 11 12
13 14 15 16          13 14 15 16
``` 

On 4-node cluster, this algorithm would split both matrices 4 blocks:

```
┌-------┬-------┐         ┌-------┬-------┐   
│ 01 02 │ 03 04 │         │ 01 02 │ 03 04 │ 
│ 05 06 │ 07 08 │         │ 05 06 │ 07 08 │      
├-------┼-------┤    \/   ├-------┼-------┤ 
│ 09 10 | 11 12 |    /\   │ 09 10 | 11 12 | 
| 13 14 | 15 16 │         | 13 14 | 15 16 │ 
└-------┴-------┘         └-------┴-------┘
```

Where every node would be responsible for calculating only one block of the resulting matrix, e.g. matrix c marked with
?? in the following diagram:

```
┌-------┬-------┐
│ ?? ?? │ 00 00 │
│ ?? ?? │ 00 00 │
├-------┼-------┤
│ 00 00 | 00 00 |
| 00 00 | 00 00 │
└-------┴-------┘
```

Is calculated with:

```
                           ┌-------┐
                           │ 01 02 │ 
┌-------┬-------┐          │ 05 06 │             
│ 01 02 │ 03 04 │          ├-------┤               
│ 05 06 │ 07 08 │          │ 09 10 |               
└-------┴-------┘          | 13 14 │
                           └-------┘               
```

Meaning that every node knows entire row and column of the input matrices. 
In this particular example, local computations are done in 2-parallel threads
(because 4x4 divided into 4 nodes would result into having 2x2 blocks). E.g.

```
Thread 1

┌-------┐       ┌-------┐      ┌-------┐
│ 01 02 │   \/  │ 01 02 │  ──  │ 05 11 │
│ 05 06 │   /\  │ 05 06 │  ──  │ 11 25 │
└-------┘       └-------┘      └-------┘

Thread 2
┌-------┐       ┌-------┐      ┌---------┐
│ 03 04 │   \/  │ 09 10 │  ──  │ 105 127 │
│ 07 08 │   /\  │ 13 14 │  ──  │ 143 173 │
└-------┘       └-------┘      └---------┘

Reducing into a block of matrix C(sequential)

┌-------┐      ┌---------┐       ┌---------┬-------┐
│ 05 11 │   +  │ 105 127 │  ──   │ 110 138 │ 00 00 │
│ 11 25 │      │ 143 173 │  ──   │ 144 198 │ 00 00 │
└-------┘      └---------┘       ├---------┼-------┤
                                 │  00 00  | 00 00 |
                                 |  00 00  | 00 00 │
                                 └---------┴-------┘
```

Once every node is done with calculations, the algorithm uses MPI to return the results back to the root node and print it there.


## Building

Linux:
```shell script
.build# mpicc -fopenmp -o matrix_dot_omp_mpi ../matrix_dot_omp_mpi.c -lm
```

cmake on Mac:
```shell script
.build# cmake ..
.build# make
```

## Running

```
                   ┌---------------------- Number of nodes
                   |                        ┌-- X size of square matrix
.build# mpirun -np 4 ../matrix_dot_omp_mpi 16
16;4;3;19
 | | |  |
 | | |  └--- Time spent in calculation and data transfer(microseconds)
 | | └------ Time spent in data transfer only(microseconds)
 | └-------- Number of nodes
 └---------- X size of square matrix
```

## Outputs and debug

There are 2 levels of debug info you can control in file utils.c:

```
// Enable this to show low-level debug messages
const bool DEBUG_LOGGER_ENABLED = false;
// Enable this to show info messages(e.g. input matrices\output matrices)
const bool INFO_LOGGER_ENABLED = false;
```

If you wish to enable the output, put "true" to the logger you want to enable and re-compile the program