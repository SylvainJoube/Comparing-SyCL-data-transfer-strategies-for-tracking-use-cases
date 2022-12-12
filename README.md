# Comparing SyCL data transfer strategies for tracking use cases

1. Cloner le repo sur la machine sur laquelle faire les tests.

2. Preparation des répertoires, mises à jour du PATH, définition des alias :

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
