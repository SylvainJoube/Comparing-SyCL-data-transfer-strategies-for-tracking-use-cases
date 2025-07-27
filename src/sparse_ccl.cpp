#include <iostream>
#include <filesystem>
#include <fstream>
#include <chrono>

// file
#include <sys/stat.h>
#include <unistd.h>
#include <string>

// SyCL specific includes
#include <sycl/sycl.hpp>
#include <array>
#include <sys/time.h>
#include <stdlib.h>

#include "utils.h"
#include "progress.h"
#include "constants.h"
#include "traccc_fcts.h"
#include "bench25.hpp"
//#include "bench_mems.hh"

// Installation :


// clear && icpx sparse_ccl.cpp constants.cc progress.cc utils.cc -o sparseccl -fsycl -O3 -std=c++20 -Wall -Wextra -I/home/data_sync/academique/These/kiwaku_2025-06/include -I/home/data_sync/academique/These/dependencies/raberu/include -I/home/data_sync/academique/These/dependencies/kumi/include -I/home/data_sync/academique/These/dependencies/eve_tag2023/include

// Annulé: suppression de utils.cc, pas le temps de gérer les soucis de link, j'ai tout mis dans le même .hpp
// En fait j'ai fait un nouveau fichier bench25.hpp, inclus une seule fois par traccc_fcts.h et c'est tout.

// ./sparseccl "no_device" 10 1
// arg1: nom du device
// arg2: nombre de fois qu'il faut charger les données (pour avoir un jeu de données plus gros)
// arg3: nombre de répétitions

int main(int argc, char *argv[])
{

    init_computers();
    log("");
    log("========~~~~~~~ VERSION " + DISPLAY_VERSION + " ~~~~~~~========");

    assert(argc>1) ;
    assert(argc<5) ;
    // int device = atoi(argv[1]);
    std::string arg_device(argv[1]);

    if ((arg_device == "cpu") || (arg_device == "CPU"))
        bench25::CHOOSEN_BACKEND = bench25::backend_t::CPU;
    else if ((arg_device == "gpu") || (arg_device == "GPU"))
        bench25::CHOOSEN_BACKEND = bench25::backend_t::GPU;
    else
    {
        std::cout << "ERREUR DE DEVICE (premier argument): peut être soit 'CPU', soit 'GPU'.\n";
        std::terminate();
    }


    int load_count = 1 ;
    if (argc>2) { load_count = atoi(argv[2]) ; }

    unsigned int arg_repeat = 12 ;
    if (argc>3) { arg_repeat = atoi(argv[3]) ; }

    // DEL25
    // selector_list_devices_generic dev_list_select2{device};
    // ::sycl::queue temp_queue2(dev_list_select2, exception_handler);


    runtime_environment.repeat_load_count = load_count;
    base_traccc_repeat_load_count = runtime_environment.repeat_load_count;


    // log("device score: " + std::to_string(device));
    log("data replicate: " + std::to_string(runtime_environment.repeat_load_count) + " times");
    log("recompute: " + std::to_string(arg_repeat) + " times");

    log("");
    log("=== Currently running on computer: " + runtime_environment.computer_name + " ===");
    // log("=== device: " + runtime_environment.device_name   + " ===\n");

    std::string computerName = runtime_environment.computer_name;

    // if (argc == 1) no argument, only print devices

    // Common variables declaration

    // FORCE_EXECUTION_ON_NAMED_DEVICE = true; set as const
    //MUST_RUN_ON_DEVICE_NAME = "Intel(R) UHD Graphics 620 [0x5917]";

    REPEAT_COUNT_REALLOC = arg_repeat; // nombre de fois que le test doit être lancé (défini dans le main)

    REPEAT_COUNT_ONLY_PARALLEL = 0;//12;    

    traccc::run_all_traccc_acat_benchs_generic();

    return 0;
}

