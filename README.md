# hpcfem
Bottom-up development of high-performance finite element operators

## Compile
- Basic C++ Implementation
```bash
cd a64fx
module load llvm
clang++ macro.cpp --std=c++11 -o macro -Wall -g -fsanitize=address -fno-omit-frame-pointer
```
- Basic C++ Implementation
```bash
cd cuda
module load cuda
nvcc macro.cu --std=c++11 -lineinfo -o cargo -g -G -Xcompiler -fsanitize=address -fno-omit-frame-pointer -Wall  -Xcompiler -fopenmp 
```
