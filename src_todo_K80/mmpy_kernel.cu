// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"

using namespace std;

#include <stdio.h>

#define SUBBLOCK_COLS (BLOCKTILE_N / BLOCKDIM_X)
#define SUBBLOCKS_ROWS (BLOCKTILE_M / BLOCKDIM_Y)

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B)
{
    // if (threadIdx.x == 0 && threadIdx.y == 0) {
        // printf("blockDim.x = %d, blockDim.y = %d\n", blockDim.x, blockDim.y);
    // }
    // const int x = blockDim.x;
    // const int y = blockDim.y;
    __shared__ _DOUBLE_ shared_A[BLOCKTILE_M][BLOCKTILE_K];
    __shared__ _DOUBLE_ shared_B[BLOCKTILE_K][BLOCKTILE_N];

    int matrix_size = N * N;
    int ty = threadIdx.y, tx = threadIdx.x;
    int by = blockIdx.y, bx = blockIdx.x;
    int I = by * BLOCKTILE_M + ty;
    int J = bx * BLOCKTILE_N + tx;
    // if (threadIdx.x != 0 && threadIdx.y != 0) {
        // printf("I = %d, J = %d\n", I, J);
        // printf("bx = %d, by = %d\n", bx, by);
    // }
    _DOUBLE_ Cij[SUBBLOCKS_ROWS][SUBBLOCK_COLS] = {0};

    #pragma unroll
    for (int kk = 0; kk < N; kk += BLOCKTILE_K) {

        /* 
         * Requirements:
         *
         * 1) Increase ILP by doing more work per thread. This reduces
         * the number of threads needed per thread-block and therefore
         * reduces the  amount of time spent waiting in the 
         * __syncthreads() barrier. By using independent instructions 
         * before and after the __syncthreads() barrier, we keep the
         * GPU busy despite the lower occupancy resulting from smaller
         * thread block sizes.
         *
         * 2) Global memory accesses should coalesce i.e. threads within a
         * warp must access sequential memory addresses aligned on a 128-
         * byte boundary.
         *
         * We achieve the above two goals by using smaller thread block
         * sizes (around 16x16 as provided in the starter code). However,
         * since the matrix block dimensions could be larger than the thread
         * block sizes, we can improve ILP by getting each thread to load
         * more than one element of matrices A and B. For example, using
         * 64x64 matrix block sizes with a thread block of 16x16 would mean
         * that each thread is tasked with now loading 4x4 elements of 
         * matrices A and B. But while doing this, we need to ensure that
         * global memory accesses are coalesced. Therefore, we split the
         * matrix block into smaller sub-blocks such that each thread loads
         * an element from each of these sublocks (essentially, the thread
         * block size acts as a stride for each thread to load elements).
         * This way, threads with consecutive thread IDs would access
         * sequential global memory addresses.
         */
        #pragma unroll        
        for (int i = 0; i < BLOCKTILE_M; i += BLOCKDIM_Y) {
            #pragma unroll
            for (int j = 0; j < BLOCKTILE_K; j += BLOCKDIM_X) {

                int A_index = (I + i) * N + kk + tx + j;

                shared_A[ty + i][tx + j] = (A_index < matrix_size) ? A[A_index] : 0;
            }
        }

        for (int i = 0; i < BLOCKTILE_K; i += BLOCKDIM_Y) {
            #pragma unroll
            for (int j = 0; j < BLOCKTILE_N; j += BLOCKDIM_X) {
                int B_index = (kk + ty + i) * N + J + j;

                shared_B[ty + i][tx + j] = (B_index < matrix_size) ? B[B_index] : 0;
            }
        }

        __syncthreads();

        /*
         * Shared memory accesses should be such that they don't cause bank conflicts.
         * To do this, we once again use striding across matrix sub-blocks so that
         * threads with consecutive IDs access different banks.
         */

        #pragma unroll
        for (int k = 0; k < BLOCKTILE_K; ++k) {
            #pragma unroll
            for (int i = 0; i < SUBBLOCKS_ROWS; ++i) {
                #pragma unroll
                for (int j = 0; j < SUBBLOCK_COLS; ++j) {
                    Cij[i][j] += shared_A[ty + i * BLOCKDIM_Y][k] * shared_B[k][tx + j * BLOCKDIM_X];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < SUBBLOCKS_ROWS; ++i) {
        #pragma unroll
        for (int j = 0; j < SUBBLOCK_COLS; ++j) {
            if (I + i * BLOCKDIM_Y < N && J + j * BLOCKDIM_X < N) {
                C[(I + i * BLOCKDIM_Y) * N + J + j * BLOCKDIM_X] = Cij[i][j];
            }
        }
    }
}
