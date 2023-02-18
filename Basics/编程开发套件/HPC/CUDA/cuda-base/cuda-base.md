
- [cuda-base](#cuda-base)
  - [编译简化demo](#编译简化demo)
  - [e01-HelloWorld](#e01-helloworld)
  - [e02-add](#e02-add)
  - [e02-device prop](#e02-device-prop)
  - [add-mul](#add-mul)
    - [benchmark n 100000](#benchmark-n-100000)


# cuda-base

## 编译简化demo


```python
import os 
class CudaAutoBuild(object):
    def __init__(self, root_dir = "/tmp/cuda-base"):
        self.root_dir = root_dir 
        assert root_dir.startswith("/tmp")
        os.makedirs(self.root_dir, exist_ok=True)
        os.system(f"rm -rf {os.path.join(self.root_dir, '*')}")


    def __enter__(self):
        return self
    
    def __exit__(self, *args, **kws):
        pass 
    
    def add_file(self, filename, filestr):
        with open(os.path.join(self.root_dir, filename), "w") as f:
            f.write(f"{filestr.strip()}\n\n")

    def build(self, cmd=None):
        if cmd is None:
            cmd = "nvcc hello.cu --generate-code arch=compute_50,code=sm_50 -o hello"
        os.system(f"cd {self.root_dir} && {cmd}")
    
    def exec(self, cmd=None):
        if cmd is None:
            cmd = "hello"
        os.system(f"cd {self.root_dir} && chmod +x {cmd} && ./{cmd}")
```


```python
with CudaAutoBuild() as cab:
    cab.add_file("hello.cu", r"""
#include <stdio.h>

__global__ void print_HelloWorld(void) {
    printf("Hello World! from thread [%d, %d] From device.\n", threadIdx.x, blockIdx.x);
}

int main() {
    printf("Hello World from host!\n");
    print_HelloWorld<<<1, 1>>>();

    cudaDeviceSynchronize();
    return 0;
}
    """)
    cab.build()
    cab.exec()


```

    Hello World from host!
    Hello World! from thread [0, 0] From device.


## e01-HelloWorld


```python
# create hello-world file
custr_HelloWorld = r"""
#include <stdio.h>

__global__ void print_HelloWorld(void) {
    printf("Hello World! from thread [%d, %d] From device.\n", threadIdx.x, blockIdx.x);
}

int main() {
    printf("Hello World from host!\n");
    print_HelloWorld<<<1, 1>>>();

    cudaDeviceSynchronize();
    return 0;
}
    """

!rm -rf build/*
!mkdir build
with open("build/hello-world.cu", "w") as f:
    f.write(custr_HelloWorld)

```

    mkdir: cannot create directory ‘build’: File exists



```python
# build
!nvcc build/hello-world.cu \
    --generate-code arch=compute_50,code=sm_50 \
    -o build/hello-world

```


```python
# exec
!chmod +x build/hello-world 
!build/hello-world
```

    Hello World from host!
    Hello World! from thread [0, 0] From device.



```python

```

## e02-add


```python
with CudaAutoBuild() as cab:
    cab.add_file("hello.cu", r"""
#include <stdio.h>


// Definition of kernel functin to add two variable
__global__ void gpu_add(int d_a, int d_b, int *d_c) {
    *d_c = d_a + d_b;
}

// main function
int main() {
    // Defining host variable to store answer
    int h_a = 125, h_b = 236;
    int h_c;

    // Defining device pointer
    int *d_c;

    // Allocating memory for device pointer
    cudaMalloc((void**)&d_c, sizeof(int));

    // Kernal call
    gpu_add<<<1, 1>>>(h_a, h_b, d_c);

    // Copy result from device memory to host memory
    cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d + %d = %d\n", h_a, h_b, h_c);

    // Free up memory
    cudaFree(d_c);

}
    """)
    cab.build()
    cab.exec()
```

    125 + 236 = 361



```python

```

## e02-device prop
- refer
  - https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp


```python
with CudaAutoBuild() as cab:
    cab.add_file("hello.cu", r"""
#include <stdio.h>

// main function
int main() {
    
    int deviceCount = -1;
    cudaGetDeviceCount(&deviceCount);
    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d. \n",
            device, deviceProp.major, deviceProp.minor
        );
        printf("\t name: %s.\n", deviceProp.name);
        printf("\t maxThreadsDim: (%d, %d, %d).\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("\t maxGridSize: (%d, %d, %d).\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("\t maxThreadsPerBlock: %d.\n", deviceProp.maxThreadsPerBlock);
    }
    return 0;    

}
    """)
    cab.build()
    cab.exec()
```

    Device 0 has compute capability 5.0. 
    	 name: NVIDIA GeForce GTX 960M.
    	 maxThreadsDim: (1024, 1024, 64).
    	 maxGridSize: (2147483647, 65535, 65535).
    	 maxThreadsPerBlock: 1024.



```python

```


```python

```

## add-mul
计算  
$y = a - \dfrac{b}{2} + c^2 $



```python
# cuda 
with CudaAutoBuild() as cab:
    cab.add_file("hello.cu", r"""
#include <stdio.h>

// Definition of kernel functin to add two variable
// y = a - b / 2 + c ** 2
__global__ void
arr_add(size_t n, float *d_a, float *d_b, float *d_c, float *d_y) {
    size_t tid = blockIdx.x;
    if (tid < n) {
        d_y[tid] = d_a[tid] - d_b[tid] / 2 + d_c[tid] * d_c[tid];
//        printf("idx a b c y: %05ld, %f, %f, %f, %f\n", tid, d_a[tid], d_b[tid],
//               d_c[tid], d_y[tid]);
    }
}


// main function
int main() {
    size_t n = 100000;
    float h_a[n], h_b[n], h_c[n], h_y[n];
    for (int i = 0; i < n; i++) {
        h_a[i] = (float) i;
        h_b[i] = (float) i;
        h_c[i] = (float) i;
        h_y[i] = 0;
    }

    float *d_a, *d_b, *d_c, *d_y;

    // Allocating memory
    cudaMalloc((void **) &d_a, n * sizeof(float));
    cudaMalloc((void **) &d_b, n * sizeof(float));
    cudaMalloc((void **) &d_c, n * sizeof(float));
    cudaMalloc((void **) &d_y, n * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, n * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel call
    arr_add<<<n, 1>>>(n, d_a, d_b, d_c, d_y);

    // Copy result
    cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();

    float sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += h_y[i];
    }
    // printf("sum: %f, ", sum);
    
    // Free
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_y);

    return 0;
}

    """)
    cab.build()
    cab.exec()
```


```python
%%timeit  
!/tmp/cuda-base/hello 
```

    190 ms ± 2.18 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
# cpp
with CudaAutoBuild() as cab:
    cab.add_file("hello.cu", r"""
#include <stdio.h>

// Main Program
// y = a - b / 2 + c ** 2
void arr_add(size_t n, const float *a, const float *b, const float *c, float *y) {
    for (size_t i = 0; i < n; ++i) {
        y[i] = a[i] - b[i] / 2 + c[i] * c[i];
    }
}


int main() {
    size_t n = 100000;
    float a[n], b[n], c[n], y[n];
    for (int i = 0; i < n; i++) {
        a[i] = (float)i;
        b[i] = (float)i;
        c[i] = (float)i;
        y[i] = 0;
    }

    arr_add(n, a, b, c, y);
    float sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += y[i];
    }
    // printf("sum: %f, ", sum);
}
    """)
    cab.build()
    cab.exec()
```


```python
%%timeit  
!/tmp/cuda-base/hello 
```

    122 ms ± 329 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)


### benchmark n 100000  
```txt
-------------------------------------------------------------
Benchmark                   Time             CPU   Iterations
-------------------------------------------------------------
bm_arr_add_cpu/1       351769 ns       351759 ns         1905
bm_arr_add_cpu/10     3320200 ns      3320090 ns          209
bm_arr_add_cpu/100   33445583 ns     33444053 ns           21


-------------------------------------------------------------
Benchmark                   Time             CPU   Iterations
-------------------------------------------------------------
bm_arr_add_gpu/1       502368 ns       502253 ns        10000
bm_arr_add_gpu/10     5023164 ns      5022271 ns         1000
bm_arr_add_gpu/100   50218850 ns     50204298 ns          100


----------------------------------------------------------------
Benchmark                      Time             CPU   Iterations
----------------------------------------------------------------
bm_arr_add_cpu/100         18050 ns        18046 ns        36805
bm_arr_add_cpu/1000      1690554 ns      1690532 ns          429
bm_arr_add_cpu/10000   177923426 ns    177912452 ns            4
bm_arr_add_cpu/100000 1.6662e+10 ns   1.6662e+10 ns            1

----------------------------------------------------------------
Benchmark                      Time             CPU   Iterations
----------------------------------------------------------------
bm_arr_add_gpu/100        287895 ns       287858 ns         2302
bm_arr_add_gpu/1000      5723494 ns      5723163 ns          138
bm_arr_add_gpu/10000   291581341 ns    291429314 ns            3
bm_arr_add_gpu/100000 2.7557e+10 ns   2.7556e+10 ns            1

```


```python

```

    -rwxr-xr-x 1 root root 794K  2月 18 21:16 /tmp/cuda-base/hello

