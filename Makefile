
CXXFLAGS = -O2 -std=c++17 -Wall # -O0 -g     -O2 -std=c++17
FILE_NAME = bench
DIRS = bin
RUN_EXE_NAME = bin/$(FILE_NAME)

check-env:
ifndef SBENCH_SYCL_COMPILER_CMD
	$(error Please set the environment variable: SBENCH_SYCL_COMPILER_CMD. It should point to the full sycl compiler path)
endif

# CXX = /home/${USER}/sycl_workspace/llvm/build/bin/clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=/usr/local/cuda
# CXX = dpcpp
# CXX = syclcc
CXX = $(SBENCH_SYCL_COMPILER_CMD)


# Affiche uniquement la liste des devices dispo
all: build run

build: check-env
	$(CXX) $(CXXFLAGS) -o bin/$(FILE_NAME) $(FILE_NAME).cpp

syclcc:
	syclcc $(CXXFLAGS) -o bin/$(FILE_NAME) $(FILE_NAME).cpp

dpcpp:
	dpcpp $(CXXFLAGS) -o bin/$(FILE_NAME) $(FILE_NAME).cpp

run: 
	./$(RUN_EXE_NAME)


clean: 
	rm -f $(RUN_EXE_NAME)


# Creates needed directories
$(shell mkdir -p $(DIRS))
$(shell mkdir -p output_bench)
