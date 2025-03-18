/**
 * APPROXIMATE PATTERN MATCHING
 *
 * INF560
 */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>
#include <cuda.h>

#define APM_DEBUG 0

const int MAX_PATTERN_LENGTH = 200;

char *
read_input_file(char *filename, int *size)
{
    char *buf;
    off_t fsize;
    int fd = 0;
    int n_bytes = 1;

    /* Open the text file */
    fd = open(filename, O_RDONLY);
    if (fd == -1)
    {
        fprintf(stderr, "Unable to open the text file <%s>\n", filename);
        return NULL;
    }

    /* Get the number of characters in the textfile */
    fsize = lseek(fd, 0, SEEK_END);
    if (fsize == -1)
    {
        fprintf(stderr, "Unable to lseek to the end\n");
        return NULL;
    }

#if APM_DEBUG
    printf("File length: %lld\n", fsize);
#endif

    /* Go back to the beginning of the input file */
    if (lseek(fd, 0, SEEK_SET) == -1)
    {
        fprintf(stderr, "Unable to lseek to start\n");
        return NULL;
    }

    /* Allocate data to copy the target text */
    buf = (char *)malloc(fsize * sizeof(char));
    if (buf == NULL)
    {
        fprintf(stderr, "Unable to allocate %lld byte(s) for main array\n",
                fsize);
        return NULL;
    }

    n_bytes = read(fd, buf, fsize);
    if (n_bytes != fsize)
    {
        fprintf(stderr,
                "Unable to copy %lld byte(s) from text file (%d byte(s) copied)\n",
                fsize, n_bytes);
        return NULL;
    }

#if APM_DEBUG
    printf("Number of read bytes: %d\n", n_bytes);
#endif

    *size = n_bytes;

    close(fd);

    return buf;
}

#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))

int levenshtein(char *s1, char *s2, int len, int *column)
{
    unsigned int x, y, lastdiag, olddiag;

    if (len <= 0)
    {
        return 0;
    }

    for (y = 1; y <= len; y++)
    {
        column[y] = y;
    }
    for (x = 1; x <= len; x++)
    {
        column[0] = x;
        lastdiag = x - 1;

        char s2_x_minus_1 = s2[x - 1];
        int column_y_minus_1 = column[0];
        int column_y = column[1];

        for (y = 1; y <= len; y++)
        {
            olddiag = column_y;
            column_y = MIN3(
                column_y + 1,
                column_y_minus_1 + 1,
                lastdiag + (s1[y - 1] == s2_x_minus_1 ? 0 : 1));
            lastdiag = olddiag;

            column[y] = column_y;
            column_y_minus_1 = column_y;
            column_y = column[y + 1];
        }
    }
    return (column[len]);
}


__device__ int levenshtein_kernel(const char *s1, const char *s2, int len) {
    unsigned int x, y, lastdiag, olddiag;
    unsigned int column[MAX_PATTERN_LENGTH + 1];
    for (y = 1; y <= len; y++)
        column[y] = y;
    for (x = 1; x <= len; x++) {
        column[0] = x;
        for (y = 1, lastdiag = x - 1; y <= len; y++) {
            olddiag = column[y];
            column[y] = MIN3(column[y] + 1, column[y - 1] + 1, lastdiag + (s1[y-1] == s2[x - 1] ? 0 : 1));
            lastdiag = olddiag;
        }
    }
    return column[len];
}

__global__ void gpu_pattern_matching(const char *text, int text_length, int chunk_size, const char *patterns, int *match_counts, int size_pattern, int approx_factor, int pattern_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= chunk_size)
    {
        return;
    }
    
    int size = size_pattern;
    if (text_length - idx < size_pattern)
    {
        size = text_length - idx;
    }

    if (size <= 0)
    {
        return;
    }

    int distance = levenshtein_kernel(patterns, &text[idx], size);

    if (distance <= approx_factor) {
        atomicAdd(&match_counts[pattern_idx], 1);
    }
}

void cuda_pattern_matching(const char *text, int text_length, int chunk_size, char **patterns, int num_patterns, int approx_factor, int *match_counts) {
    char *d_text, *d_patterns;
    int *d_match_counts;
    int total_pattern_length = 0;

    // Calculate total pattern length for single allocation
    for (int i = 0; i < num_patterns; i++) {
        total_pattern_length += strlen(patterns[i]) + 1; // +1 for null-terminator
    }

    // CUDA memory allocations (must be outside OpenMP parallel region)
    cudaMalloc((void **)&d_text, text_length * sizeof(char));
    cudaMemcpy(d_text, text, text_length * sizeof(char), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_patterns, total_pattern_length * sizeof(char));
    cudaMalloc((void **)&d_match_counts, num_patterns * sizeof(int));
    cudaMemset(d_match_counts, 0, num_patterns * sizeof(int));

    // Copy all patterns into a single GPU buffer
    int offset = 0;
    int offsets[num_patterns];
    for (int i = 0; i < num_patterns; i++) {
        int pattern_length = strlen(patterns[i]) + 1;
        cudaMemcpy(d_patterns + offset, patterns[i], pattern_length * sizeof(char), cudaMemcpyHostToDevice);
        offsets[i] = offset;
        offset += pattern_length;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_patterns; i++) {
        int pattern_length = strlen(patterns[i]);
        int offset = pattern_length + 1;
        int blockSize = 256;
        int gridSize = (chunk_size + blockSize - 1) / blockSize;

        // Each thread launches a separate kernel for each pattern
        gpu_pattern_matching<<<gridSize, blockSize>>>(d_text, text_length, chunk_size, d_patterns + offsets[i], d_match_counts, pattern_length, approx_factor, i);
    }

    // Synchronize to ensure all GPU work is completed before copying results
    cudaDeviceSynchronize();

    cudaMemcpy(match_counts, d_match_counts, num_patterns * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_text);
    cudaFree(d_patterns);
    cudaFree(d_match_counts);
}


int main(int argc, char **argv)
{
    char **pattern;
    char *filename;
    int approx_factor = 0;
    int nb_patterns = 0;
    int i, j;
    char *buf;
    struct timeval t1, t2;
    double duration;
    int n_bytes;
    int *n_matches;

    /* MPI Initialization */
    MPI_Init(&argc, &argv);

    int rank, size;
    /* Get the rank of the current task and the number
     * of MPI processe
     */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nbGPU;
    cudaGetDeviceCount(&nbGPU) ;
    cudaSetDevice (rank % nbGPU) ;

    /* Check number of arguments */
    if (argc < 4)
    {
        printf("Usage: %s approximation_factor "
               "dna_database pattern1 pattern2 ...\n",
               argv[0]);
        return 1;
    }

    /* Get the distance factor */
    approx_factor = atoi(argv[1]);

    /* Grab the filename containing the target text */
    filename = argv[2];

    /* Get the number of patterns that the user wants to search for */
    nb_patterns = argc - 3;

    /* Fill the pattern array */
    pattern = (char **)malloc(nb_patterns * sizeof(char *));
    if (pattern == NULL)
    {
        fprintf(stderr,
                "Unable to allocate array of pattern of size %d\n",
                nb_patterns);
        return 1;
    }

    /* Grab the patterns */
    int max_pattern_length = 0;
    for (i = 0; i < nb_patterns; i++)
    {
        int l;

        l = strlen(argv[i + 3]);
        if (l <= 0)
        {
            fprintf(stderr, "Error while parsing argument %d\n", i + 3);
            return 1;
        }

        pattern[i] = (char *)malloc((l + 1) * sizeof(char));
        if (pattern[i] == NULL)
        {
            fprintf(stderr, "Unable to allocate string of size %d\n", l);
            return 1;
        }

        if (l > max_pattern_length)
        {
            max_pattern_length = l;
        }

        strncpy(pattern[i], argv[i + 3], (l + 1));
    }

    if (rank == 0)
    {
        printf("Approximate Pattern Mathing: "
               "looking for %d pattern(s) in file %s w/ distance of %d\n",
               nb_patterns, filename, approx_factor);
    }

    buf = read_input_file(filename, &n_bytes);
    if (buf == NULL)
    {
        return 1;
    }

    int chunk_size = (n_bytes + size - 1) / size;
    int extra_overlap = max_pattern_length - 1;
    int local_size = chunk_size + extra_overlap;
    int local_start = rank * chunk_size;
    if (local_start + local_size > n_bytes)
    {
        local_size = n_bytes - local_start;
    }

    char *local_buf = (char *)malloc(local_size * sizeof(char));

    // copy local part
    strncpy(local_buf, buf + local_start, local_size);

    /* Allocate the array of matches */
    n_matches = (int *)malloc(nb_patterns * sizeof(int));
    if (n_matches == NULL)
    {
        fprintf(stderr, "Error: unable to allocate memory for %ldB\n",
                nb_patterns * sizeof(int));
        return 1;
    }

    /*****
     * BEGIN MAIN LOOP
     ******/

    /* Timer start */
    gettimeofday(&t1, NULL);

    cuda_pattern_matching(local_buf, local_size, chunk_size, pattern, nb_patterns, approx_factor, n_matches);

    int *global_matches = NULL;
    if (rank == 0)
    {
        global_matches = (int *)calloc(nb_patterns, sizeof(int));
    }

    MPI_Reduce(n_matches, global_matches, nb_patterns, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    /*****
     * END MAIN LOOP
     ******/

    if (rank == 0)
    {
        printf("APM done in %lf s\n", duration);
        for (i = 0; i < nb_patterns; i++)
        {
            printf("Number of matches for pattern <%s>: %d\n",
                   pattern[i], global_matches[i]);
        }
        free(global_matches);
    }

    free(n_matches);
    free(buf);
    free(local_buf);
    // free(column);
    for (i = 0; i < nb_patterns; i++)
    {
        free(pattern[i]);
    }
    free(pattern);

    MPI_Finalize();

    return 0;
}