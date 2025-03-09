## Build && Run

```bash
source set_env.h
make
salloc -N 1 -n 6 mpirun ./apm 17 ./dna/line_chrY.fa ./dna/line_chrY.fa
```