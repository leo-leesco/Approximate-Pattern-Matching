SRC_DIR=src
HEADER_DIR=include
OBJ_DIR=obj

CC=mpicc # gcc
CFLAGS=-O3 -I$(HEADER_DIR)
LDFLAGS=

SRC= apm.c

OBJ= $(OBJ_DIR)/apm.o

all: $(OBJ_DIR) apm apm_parallel apm_cuda

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) -fopenmp $(CFLAGS)  -c -o $@ $^ $(LDFLAGS)

apm:$(OBJ)
	$(CC) -fopenmp $(CFLAGS) $(LDFLAGS) -o $@ $^

apm_parallel: $(OBJ)
	cp apm_cuda test/apm_parallel

SRC_CUDA= apm_cuda.cu
OBJ_CUDA= $(OBJ_DIR)/apm_cuda.o

MPI_INC=$(shell mpicc --showme:compile | sed 's/-pthread//g')  
MPI_LIBS=$(shell mpicc --showme:link | sed 's/-pthread//g')    

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu
	nvcc -Xcompiler "-fopenmp" $(CFLAGS) $(MPI_INC) -c -o $@ $^

apm_cuda: $(OBJ_CUDA)
	mpicxx -fopenmp $(CFLAGS) $(LDFLAGS) $(MPI_LIBS) -lcudart -o $@ $^ -L/usr/local/cuda/lib64 -I/usr/local/cuda/include

clean:
	rm -f apm apm_cuda $(OBJ) $(OBJ_CUDA) ; rmdir $(OBJ_DIR)