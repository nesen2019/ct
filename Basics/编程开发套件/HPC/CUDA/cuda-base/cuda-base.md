
- [cuda-base](#cuda-base)
  - [编译简化demo](#编译简化demo)
  - [e01-HelloWorld](#e01-helloworld)
  - [e02-add](#e02-add)




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
    int h_a = 3, h_b = 5;
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

    3 + 5 = 8



```python

```


```python

```
