## Build && Run

```bash
source set_env.h
make
salloc -N 1 -n 6 mpirun ./apm 17 ./dna/line_chrY.fa ./dna/line_chrY.fa
```

## Explanation

### MPI

# MPI Parallelization for Approximate Pattern Matching

To improve the efficiency of the approximate pattern matching algorithm, we implemented parallelization using **MPI (Message Passing Interface)**. The main idea is to **distribute the input text across multiple processes**, allowing each process to compute pattern matches in its assigned chunk independently. This significantly reduces the number of characters each process needs to check, improving performance, especially for large text files. Though it greatly improves performance, it doesn't significantly change the underlying logic, allowing for other optimization and parallelization techniques.

## Parallelization Strategy

### 1. Text Partitioning
- The input text is divided among `N` processes, where `N` is the number of available MPI ranks.
- Each process reads and fixes it's own contiguous **chunk** of the text for processing.

### 2. Handling Overlapping Boundaries
- Since pattern matching may require examining text beyond a process's assigned chunk (to detect matches at chunk boundaries), an **overlap of `max_pattern_length - 1` characters** is included from the neighboring process.
- The first and last processes handle edge cases where no adjacent chunk exists.

### 3. Parallel Computation of Matches
- Each process iterates over its text segment and applies the **Levenshtein distance algorithm** to compare each pattern against substrings within its chunk.
- Matches are counted independently in each process.

### 4. Gathering Results
- After local computation, each process sends its results to the **master process (rank 0)** using `MPI_Reduce`.
- The master process aggregates the match counts from all ranks and produces the final result.

## Performance Benefits
- **Workload distribution:** Each process handles a smaller portion of the text, reducing computational time.
- **Scalability:** The algorithm can efficiently scale across multiple nodes in a distributed system.
- **Space for other optimisations:** The task is reduced by `N` without changing the logic, allowing for further use of **OMP** and **CUDA**.
