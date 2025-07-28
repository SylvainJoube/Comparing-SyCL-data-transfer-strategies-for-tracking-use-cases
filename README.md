# Comparing SyCL data transfer strategies for tracking use cases

1. Cloner le repo sur la machine sur laquelle faire les tests.

## Dépendances


### Téléchargement de EVE, Kiwaku, Raberu et Kumi

```bash
mkdir dependencies && \
git clone https://github.com/SylvainJoube/raberu.git dependencies/raberu && \
git clone https://github.com/SylvainJoube/kumi.git dependencies/kumi && \
git clone https://github.com/SylvainJoube/eve.git dependencies/eve_kwk_compatible && \
cd dependencies/eve_kwk_compatible && \
git checkout kwk_compatible && \
cd ../.. && \
git clone https://github.com/jfalcou/kiwaku.git dependencies/kiwaku_source && \
cd dependencies/kiwaku_source && \
git checkout contexts_v2 && \
cd ../..
```

### Création du script d'environnement

```bash
export ENV_FNAME="setup_env.sh" && \
echo '#!/bin/bash' > ${ENV_FNAME} && \
echo >> ${ENV_FNAME} && \
echo "export SCCL_DEPS_DIR=$(pwd)/dependencies" >> ${ENV_FNAME} && \
echo "export EVE_FLAG=\"-mavx2 -mfma\"" >> ${ENV_FNAME}


chmod +x ${ENV_FNAME}

# Options possibles :
export EVE_FLAG=""
export EVE_FLAG="-msse4.2"
export EVE_FLAG="-mavx2 -mfma"
export EVE_FLAG="-march=skylake-avx512"

# Source du fichier, à chaque nouveau terminal
source ${ENV_FNAME}
```


### Compilation

```bash
export ENV_FNAME="setup_env.sh" && \
source ${ENV_FNAME}

# Source du fichier, à chaque nouveau terminal

# Contexte SYCL par défaut (CPU)
export ICPX_FLAGS=""

# Nvidia (si x86_64 ne fonctionne pas, prendre spir64)
export ICPX_FLAGS="-fsycl-targets=nvptx64-nvidia-cuda,x86_64"

icpx sparse_ccl.cpp constants.cc progress.cc utils.cc \
-o sparseccl \
-ffp-model=precise \
-DNDEBUG \
-fsycl \
${EVE_FLAG} \
${ICPX_FLAGS} \
-O3 \
-std=c++20 \
-Wall \
-Wextra \
-I${SCCL_DEPS_DIR}/kiwaku_source/include \
-I${SCCL_DEPS_DIR}/raberu/include \
-I${SCCL_DEPS_DIR}/kumi/include \
-I${SCCL_DEPS_DIR}/eve_kwk_compatible/include
```

### Compilation sur Legend

```bash
export SCCL_DEPS_DIR="/home/sylvainj/SparseCCL/dependencies" &&\
export EVE_FLAG="-mavx2 -mfma" &&\
export ICPX_FLAGS="-fsycl-targets=nvptx64-nvidia-cuda,x86_64"

cd /home/sylvainj/SparseCCL/Comparing-SyCL-data-transfer-strategies-for-tracking-use-cases/src

```



### Exécution

`./sparseccl "GPU" 10 1`
* arg1: "CPU" ou "GPU" pour l'exécution SYCL
* arg2: nombre de fois qu'il faut charger les données (pour avoir un jeu de données plus gros)
* arg3: nombre de répétitions

```bash
./sparseccl GPU 10 1
```



1. Preparation des répertoires, mises à jour du PATH, définition des alias :

```
source env.sh
```

3. Définir la variable d'environnement `SBENCH_SYCL_COMPILER_CMD` : elle doit indiquer le chemin absolu vers le compilateur. Exemples :

```bash
# DPC++ installé, potentiellement défini via quelque chose du genre : source ~/intel/oneapi/setvars.sh
export SBENCH_SYCL_COMPILER_CMD=dpcpp

# DPC++ compilé
export SBENCH_SYCL_COMPILER_CMD="/...full_path.../llvm/build/bin/clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=/usr/local/cuda"

# HipSYCL
export SBENCH_SYCL_COMPILER_CMD=syclcc
```

4. Dans le dossier principal, faire `make build`.

5. Tourner les programmes SYCL de votre choix :

- `lsd` ou `list_devices.exe` : liste les devices.
- `ubench <score> <data-size-gb> <repeat>` ou `micro_bench.exe <score> <data-size-gb> <repeat>` : exécute sur le device de score `<score>` le banc d'essai ubench, pour une taille de `<data-size-gb>` Gb (1 par défaut), et en répétant les mêmes calculs et échanges de données `<repeat>` fois (12 par défaut).
- `sparse <score> <load-count-value> <repeat>` ou `sparse_ccl.exe <score> <load-count-value> <repeat>` : exécute sur le device de score `<score>` le banc d'essai sparsecll, en duplicant les données `<load-count-value>` fois (1 par défaut), et en répétant les mêmes calculs et échanges de données `<repeat>` fois (12 par défaut).


**Quand le fichier de sortie existe déjà, le programme ne le refait pas**.

6. Les résultats sont produits dans des fichiers `output/*.t` :
  - `sparseccl108_generalFlatten_[nom ordi]_ld[valeur de ld]_RUN1_[nom du device].t`
  - `sparseccl108_generalGraphPtr_uniqueModules_[nom ordi]_ld[valeur de ld]_RUN1_[nom du device].t`
  - `ubench2_2_[nom ordi]_4GiB_RUN1_[nom du device].t`

# A faire

- ne pas laisser trainer de fichier de sortie incomplet si l'execution a echoué avant la fin.


# Pile d'appel micro_bench.cpp

- ubench_v2::init_data_length(gb)
- ubench_v2::run_ubench2_tests(runtime_environment.computer_name, runtime_environment.runs_count)
  - run_ubench2_single_test(computer_name, i) * runtime_environment.runs_count // == 1 ?!? 
    - main_of_bench_v2(OUTPUT_FILE_NAME)
      - progress_init(7)
      - // (USM) explicit copy
      - traccc_main_sequence(myfile, sycl_mode::device_USM, true);
      - traccc_main_sequence(myfile, sycl_mode::shared_USM, true);
      - traccc_main_sequence(myfile, sycl_mode::host_USM,   true);
      - // Implicit copy
      - traccc_main_sequence(myfile, sycl_mode::shared_USM, false);
      - traccc_main_sequence(myfile, sycl_mode::host_USM,   false);
      - traccc_main_sequence(myfile, sycl_mode::accessors,  false);
      - traccc_main_sequence(myfile, sycl_mode::glibc,      false);
        * REPEAT_COUNT_REALLOC
        - traccc_bench(mode, explicit_copy);
          - allocation(bench);
          - fill(bench);
          - copy(bench);
          - kernel(bench);
          - data_type sum = read(bench);
          - dealloc(bench);
        - progress_increment() ;
        - progress_print();



# Notes diverses

To run with syclcc, set those variables :
export HIPSYCL_TARGETS="cuda:sm_35" && \
export HIPSYCL_GPU_ARCH="sm_35" && \
export HIPSYCL_CUDA_PATH="/usr/local/cuda-10.1"

On Sandor :
export HIPSYCL_TARGETS="cuda:sm_75" && \
export HIPSYCL_GPU_ARCH="sm_75" && \
export HIPSYCL_CUDA_PATH="/usr/local/cuda-10.1"
