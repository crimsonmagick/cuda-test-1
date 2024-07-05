import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule

# Define the CUDA kernel for matrix multiplication
mod = SourceModule("""
__global__ void MatrixMulKernel(float *A, float *B, float *C, int N) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < N) && (Col < N)) {
        float value = 0;
        for (int k = 0; k < N; ++k) {
            value += A[Row * N + k] * B[k * N + Col];
        }
        C[Row * N + Col] = value;
    }
}
""")

# Matrix size
N = 3

# Create random matrices A and B
A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)

# Allocate memory on the GPU
A_gpu = cuda.mem_alloc(A.nbytes)
B_gpu = cuda.mem_alloc(B.nbytes)
C_gpu = cuda.mem_alloc(A.nbytes)

# Copy the matrices to the GPU
cuda.memcpy_htod(A_gpu, A)
cuda.memcpy_htod(B_gpu, B)

# Get the kernel function from the compiled module
matrixmul = mod.get_function("MatrixMulKernel")

# Define the block and grid size
block_size = (N, N, 1)
grid_size = (1, 1, 1)

# Launch the kernel
matrixmul(A_gpu, B_gpu, C_gpu, np.int32(N), block=block_size, grid=grid_size)

# Create an empty array to store the result
C = np.empty_like(A)

# Copy the result from the GPU back to the CPU
cuda.memcpy_dtoh(C, C_gpu)

# Print the result
print("Matrix A:")
print(A)
print("Matrix B:")
print(B)
print("Result of matrix multiplication on GPU:")
print(C)
