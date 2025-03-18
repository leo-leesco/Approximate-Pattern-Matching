# Approximate Pattern Matching


<!--toc:start-->
- [Approximate Pattern Matching](#approximate-pattern-matching)
  - [Build && Run](#build-run)
  - [Test](#test)
<!--toc:end-->

## Build && Run

```bash
source set_env.sh
make
OMP_NUM_THREADS=12 salloc -N 1 -n 6 mpirun ./apm_cuda 25 ./dna/big_chrY.fa CCAGTTCCCTTCTGGAATTTAGGGGCCCTGGGACAGCCCTGTACATGAGC CATACCGATAACAACCACGAGCTAGTAAGCGCCGTCGCGCCAATAAATCT ACCCTCATTGGTCAGGTCCAGCGCATAGGGTAGGATAGGATCTGTACCAT
```

## Test

Runs on `big_chrY.fa` and 3 patterns.

```bash
source set_env.sh
make
cd test
chmod +x apm_*
python3 runner.py
```

Output:

```bash

Running Base Implementation...

Running Optimized Implementation...

===================================
          COMPARISON RESULTS       
===================================

Base Implementation Time: 59.9952 seconds
Optimized Implementation Time: 1.6439 seconds

✅ Results Match! The optimization is correct.

-----------------------------------
| Pattern | Base Matches | Optimized Matches |
-----------------------------------
| CCAGTTCCCTTCTGGAATTTAGGGGCCCTGGGACAGCCCTGTACATGAGC | 130016       | 130016            |
| CATACCGATAACAACCACGAGCTAGTAAGCGCCGTCGCGCCAATAAATCT | 224129       | 224129            |
| TATGCCACATGCCCGGAATTAGGTCTGTTACTCGTAGCAAACGTATGCGG | 289548       | 289548            |
-----------------------------------

Speedup Factor: 36.5
```
