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

#define APM_DEBUG 0

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

    int num_threads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    int *column;
    column = (int *)malloc((max_pattern_length + 1) * num_threads * sizeof(int));
    if (column == NULL)
    {
        fprintf(stderr, "Error: unable to allocate memory for column (%ldB)\n",
                (max_pattern_length + 1) * sizeof(int));
        return 1;
        // continue;
    }
    // omp_set_nested(1);

    int pattern_sizes[nb_patterns];
    for (i = 0; i < nb_patterns; i++)
    {
        pattern_sizes[i] = strlen(pattern[i]);
    }
    for (i = 0; i < nb_patterns; i++)
    {
        n_matches[i] = 0;
    }

    
    /* Check each pattern one by one */
    #pragma omp parallel for collapse(2) schedule(static) private(i, j)
    for (i = 0; i < nb_patterns; i++)
    {
        // int size_pattern = strlen(pattern[i]);

        /* Initialize the number of matches to 0 */
        // n_matches[i] = 0;
        // int result = 0;

        // int *column;
        // column = (int *)malloc((size_pattern + 1) * sizeof(int));
        // if (column == NULL)
        // {
        //     fprintf(stderr, "Error: unable to allocate memory for column (%ldB)\n",
        //             (size_pattern + 1) * sizeof(int));
        //     // return 1;
        //     continue;
        // }

        /* Traverse the input data up to the end of the file */
        // #pragma omp parallel for schedule(static) reduction(+:result) private(j)
        for (j = 0; j < chunk_size; j++)
        {
            int size_pattern = pattern_sizes[i];

            int distance = 0;
            int size;
            int *my_column = column + omp_get_thread_num() * (max_pattern_length + 1);
            // int *my_column = malloc((size_pattern + 1) * sizeof(int)); 
            // int *my_column = column;

// #if APM_DEBUG
//         #pragma omp critical
//         {
//             printf("Thread %d out of %d threads\n", omp_get_thread_num(), omp_get_num_threads());
//         }
// #endif

// #if APM_DEBUG
//             if (j % 50000 == 0)
//             {
//                 printf("Procesing byte %d (out of %d) for pattern %d \n", j, local_size, i);
//             }
// #endif

            size = size_pattern;
            if (local_size - j < size_pattern)
            {
                size = local_size - j;
            }

            if (size <= 0)
            {
                continue;
            }

            distance = levenshtein(pattern[i], &local_buf[j], size, my_column);
            // distance = 0;

            if (distance <= approx_factor)
            {
                #pragma omp atomic
                n_matches[i]++;
                // result++;
            }
        }

        // n_matches[i] = result;

#if APM_DEBUG
        printf("Rank %d: Number of matches for pattern <%s>: %d\n", rank, pattern[i], n_matches[i]);
#endif
        // free(column);
    }

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
    free(column);
    for (i = 0; i < nb_patterns; i++)
    {
        free(pattern[i]);
    }
    free(pattern);

    MPI_Finalize();

    return 0;
}