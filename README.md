# hpcfem
Bottom-up development of high-performance finite element operators

## Compile
```bash
cd a64fx
module load llvm
clang++ macro.cpp --std=c++11 -o macro -Wall -g -fsanitize=address -fno-omit-frame-pointer
```
