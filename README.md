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

- `devices` ou `devices.exe` : liste les devices.
- `ubench <score>` ou `ubench.exe <score> <load-count-value>` : exécute sur le device de score `<score>` le banc d'essai ubench. Si on veut changer la taille memoire pour ubench_2_2 (code jouet non sparse) : fichier ubench_v2_fcts.h, ligne 48. Tout est rejoué 12 fois (pour l'instant fixé en dur) (REPEAT_COUNT_REALLOC).
- `sparse <score> <load-count-value>` ou `sparse_ccl.exe <device> <load-count-value>` : exécute sur le device de score `<score>` le banc d'essai sparsecll, en duplicant les données `<load-count-value>` fois. USM host prend énormément de temps. Dans l'idéal, sur une grosse machine (comme cassidi ou sandor), un  `<load-count-value>` de `100` est recommandée (mais ça prend facilement une heure à tourner, si ce n'est plus) donc une valeur de `10` pour commencer c'est déjà très bien. Tout est rejoué 12 fois (pour l'instant fixé en dur) (REPEAT_COUNT_REALLOC). Les % à l'affichage : 50% = erreur d'approximation d'un facteur 2. Il manque un retour chariot.

**Quand le fichier de sortie existe déjà, le programme ne le refait pas**.

6. Les résultats sont produits dans des fichiers `output/*.t` :
  - `sparseccl108_generalFlatten_[nom ordi]_ld[valeur de ld]_RUN1_[nom du device].t`
  - `sparseccl108_generalGraphPtr_uniqueModules_[nom ordi]_ld[valeur de ld]_RUN1_[nom du device].t`
  - `ubench2_2_[nom ordi]_4GiB_RUN1_[nom du device].t`

# Notes

To run with syclcc, set those variables :
export HIPSYCL_TARGETS="cuda:sm_35" && \
export HIPSYCL_GPU_ARCH="sm_35" && \
export HIPSYCL_CUDA_PATH="/usr/local/cuda-10.1"

On Sandor :
export HIPSYCL_TARGETS="cuda:sm_75" && \
export HIPSYCL_GPU_ARCH="sm_75" && \
export HIPSYCL_CUDA_PATH="/usr/local/cuda-10.1"
