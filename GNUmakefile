
# One should set the variable SBENCH_SYCL_COMPILER_CMD first:
#   /home/${USER}/sycl_workspace/llvm/build/bin/clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=/usr/local/cuda
#   dpcpp
#   syclcc

# Listes
HDRS = $(wildcard src/*.h)
SRCS = $(wildcard src/*.cc)
PRGS = $(wildcard src/*.cpp)
OBJS = $(patsubst src/%.cc,build/%.o,$(SRCS))
EXES = $(patsubst src/%.cpp,bin/%.exe,$(PRGS))

# Options
CXX = $(SBENCH_SYCL_COMPILER_CMD)
CXXFLAGS = -I./src -O2 -std=c++17 -Wall# Or -O0 -g instead of -O2

all: build

check-env:
	@echo SBENCH_SYCL_COMPILER_CMD=${SBENCH_SYCL_COMPILER_CMD}
	@if [ -z "${SBENCH_SYCL_COMPILER_CMD}" ] ; then echo "Please set the environment variable: SBENCH_SYCL_COMPILER_CMD. It should point to the full sycl compiler path" ; exit 1 ; fi

build: check-env $(EXES)

clean: 
	-rm -f $(EXES)
	-rm -f bin/*.exe
	-rm -f build/*.o

verbose:
	@echo HDRS: $(HDRS)
	@echo SRCS: $(SRCS)
	@echo OBJS: $(OBJS)
	@echo PRGS: $(PRGS)
	@echo EXES: $(EXES)


#############################################################################
## Implicit rules
#############################################################################

.PRECIOUS: build/%.o

build/%.o: src/%.cc $(HDRS)
	@rm -f $@
	$(CXX) $(CXXFLAGS) -o $@ -c $<

build/%.o: src/%.cpp $(HDRS)
	@rm -f $@
	$(CXX) $(CXXFLAGS) -o $@ -c $<

bin/%.exe: build/%.o $(OBJS)
	@rm -f $@
	$(CXX) $(CXXFLAGS) -o $@ $^
