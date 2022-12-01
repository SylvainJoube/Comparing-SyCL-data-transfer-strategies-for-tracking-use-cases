# Comparing SyCL data transfer strategies for tracking use cases

Doc faite super à l'arrache, juste pour indiquer la marche à suivre, à un moment je ferai une version anglaise.

1. Cloner le repo sur la machine sur laquelle faire les tests.

2. Définir la variable d'environnement `SBENCH_SYCL_COMPILER_CMD` : elle doit indiquer le chemin absolu vers le compilateur. Exemples :

```bash
# DPC++ installé
# Potentiellement défini via quelque chose du genre : source ~/intel/oneapi/setvars.sh
export SBENCH_SYCL_COMPILER_CMD=dpcpp

# DPC++ compilé
export SBENCH_SYCL_COMPILER_CMD="/...full_path.../llvm/build/bin/clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=/usr/local/cuda"

# HipSYCL
export SBENCH_SYCL_COMPILER_CMD=syclcc

# etc.
```

3. Dans le dossier du git, faire `make build` pour créer l'exécutable `bin/bench`.
4. Faire `make run` (ou directement `./bin/bench`).
5. `Please provide the score of the device you wish to use:` => Indiquer le score du périphérique sur lequel exécuter le benchmark. (les scores de chaque device sont listés juste au-dessus, dans le terminal)
6. Si le score entré est valide, il demande confirmation `Device selected: Intel(R) Core(TM) i7-4790 CPU @ 3.60GHz`. Appuyer sur entrée (le `Please press enter to continue...`).
7. `Load count value (traccc): (default = 10)` est le nombre de fois que devra être répété le chargement du fichier `events_bin/lite_all_events.bin`. USM host prend énormément de temps. Dans l'idéal, sur une grosse machine (comme cassidi ou sandor), une valeur de `100` est recommandée (mais ça prend facilement une heure à tourner, si ce n'est plus) donc une valeur de `10` pour commencer c'est déjà très bien. Faire entrée ne fonctionne pas, il faut soit entrer un nombre, soit une chaine de caractères (qui sera vue comme étant invalide, donc sélection de la valeur par défaut).
8. Il y a confirmation : `Selected load count value (traccc): 10` puis `Please press enter to continue...`.
9. Appuyer sur entrée, et le banchmark se lance.
10. Les fichiers à récupérer pour pouvoir les visualiser sont : 
  - `sparseccl108_generalFlatten_[nom ordi]_ld[valeur de ld]_RUN1_[nom du device].t`
  - `sparseccl108_generalGraphPtr_uniqueModules_[nom ordi]_ld[valeur de ld]_RUN1_[nom du device].t`
  - `ubench2_2_[nom ordi]_4GiB_RUN1_[nom du device].t`
