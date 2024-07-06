import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import array


def load_kernel_file(filename):
    with open(filename, 'r') as file:
        return file.read()


# Compile the kernel code
mod = SourceModule(load_kernel_file('add_arrays.cu'))
add = mod.get_function("add")

a = array.array('i', [1, 2, 3, 4])
b = array.array('i', [5, 6, 7, 8])
c = array.array('i', [0, 0, 0, 0])

# Allocate memory on the GPU
a_gpu = cuda.mem_alloc(len(a) * a.itemsize)
b_gpu = cuda.mem_alloc(len(b) * b.itemsize)
c_gpu = cuda.mem_alloc(len(c) * c.itemsize)

# Transfer data from host (RAM) to the device (GPU VRAM)
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# Execute the kernel
add(a_gpu, b_gpu, c_gpu, block=(4, 1, 1))

# Transfer the result from device (GPU VRAM) to the host (RAM)
cuda.memcpy_dtoh(c, c_gpu)

print("Result:", c.tolist())
