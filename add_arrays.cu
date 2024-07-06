__global__ void add(int* a, int* b, int* c) {
  int idx = threadIdx.x;
  c[idx] = a[idx] + b[idx];
}
