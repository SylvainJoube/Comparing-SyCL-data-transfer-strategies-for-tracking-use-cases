
# move to the script directory

SCRIPT_NAME=${BASH_SOURCE[0]}
SCRIPT_DIR=`dirname ${SCRIPT_NAME}`
ORIGINAL_DIR=${PWD}

cd $SCRIPT_DIR
SCRIPT_DIR=`pwd`

# Préparation des répertoires annexes

mkdir -p output

# extend PATH and LD_LIBRARY_PATH

export PATH="${SCRIPT_DIR}/bin:${PATH}"

# aliases

alias lsd=list_devices.exe
alias ubench=micro_bench.exe
alias sparse=sparse_ccl.exe

alias cleant="rm -f ${SCRIPT_DIR}/output/*.t"
alias harvest-sparseccl=2022-09-29_sparseccl_grouped.py
alias harvest-ubench=2022-09-29_ubench_v2_grouped.py

# back to the original directory

export MAIN_DIR=${SCRIPT_DIR}
cd $ORIGINAL_DIR
