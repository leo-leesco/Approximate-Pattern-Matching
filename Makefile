SRC_DIR=src
HEADER_DIR=include
OBJ_DIR=obj

CC=mpicc # gcc
CFLAGS=-O3 -I$(HEADER_DIR) -Wall
LDFLAGS=

SRC= apm.c

OBJ= $(OBJ_DIR)/apm.o

all: $(OBJ_DIR) apm apm_parallel

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) -fopenmp $(CFLAGS)  -c -o $@ $^ $(LDFLAGS)

apm:$(OBJ)
	$(CC) -fopenmp $(CFLAGS) $(LDFLAGS) -o $@ $^

apm_parallel: $(OBJ)
	cp apm test/apm_parallel

clean:
	rm -f apm $(OBJ) ; rmdir $(OBJ_DIR)
