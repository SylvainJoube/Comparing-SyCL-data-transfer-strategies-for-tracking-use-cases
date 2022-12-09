
# One should set the variable SBENCH_SYCL_COMPILER_CMD first:
#   /home/${USER}/sycl_workspace/llvm/build/bin/clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=/usr/local/cuda
#   dpcpp
#   syclcc

# Executables
HDRS = $(wildcard *.h)
SRCS = $(wildcard *.cpp)
EXES = $(patsubst %.cpp,bin/%.exe,$(SRCS))

# Options
CXX = $(SBENCH_SYCL_COMPILER_CMD)
CXXFLAGS = -O2 -std=c++17 -Wall # Or -O0 -g instead of -O2

all: build

check-env:
	@echo SBENCH_SYCL_COMPILER_CMD=${SBENCH_SYCL_COMPILER_CMD}
	@if [ -z "${SBENCH_SYCL_COMPILER_CMD}" ] ; then echo "Please set the environment variable: SBENCH_SYCL_COMPILER_CMD. It should point to the full sycl compiler path" ; exit 1 ; fi

build: check-env $(EXES)

clean: 
	-rm -f $(EXES)
	-rm -f bin/*.exe

verbose:
	@echo SRCS: $(SRCS)
	@echo EXES: $(EXES)


#############################################################################
## Implicit rules
#############################################################################

bin/%.exe: %.cpp $(HDRS)
	@rm -f $@
	$(CXX) $(CXXFLAGS) -o $@ $<
