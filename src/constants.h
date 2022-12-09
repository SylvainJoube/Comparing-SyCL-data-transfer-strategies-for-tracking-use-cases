#pragma once

#include <iostream>
#include <filesystem>
#include <fstream>
#include <chrono>

// SyCL specific includes
#include <CL/sycl.hpp>
#include <array>
#include <sys/time.h>
#include <stdlib.h>
// #include "intel_noinit_fix.h"

extern const int ACAT_START_TEST_INDEX ;
extern const int ACAT_STOP_TEST_INDEX  ;
extern const int ACAT_RUN_COUNT        ;
extern int ACAT_REPEAT_LOAD_COUNT ;

#define DATA_TYPE unsigned int // TODO : try with unsigned int
using data_type = DATA_TYPE;
//using data_type_sum = unsigned long long;
enum sycl_mode {shared_USM, device_USM, host_USM, accessors, glibc};
//enum dataset_type {implicit_USM, device_USM, host_USM, accessors};

extern unsigned long long PARALLEL_FOR_SIZE;// = 1024 * 32 * 8;// = M ; work items number
extern unsigned long long VECTOR_SIZE_PER_ITERATION;// = 1; // = L ; vector size per workitem (i.e. parallel_for task) = nb itérations internes par work item
extern unsigned long long BASE_VECTOR_SIZE_PER_ITERATION; // updated in list_devices()

extern sycl_mode CURRENT_MODE ;

extern int MEMCOPY_IS_SYCL ;
extern int SIMD_FOR_LOOP ;
constexpr int USE_NAMED_KERNEL = 1; // Sandor does not support anonymous kernels.
constexpr bool KEEP_SAME_DATASETS = true ; 
extern int USE_HOST_SYCL_BUFFER_DMA ; 

extern bool ERR_DEVICE_NOT_FOUND ;

#define DATA_VERSION 7
#define DATA_VERSION_TRACCC 108

// number of diffrent datasets
#define DATASET_NUMBER 1

#define CHECK_SIMD_CPU false

#define INPUT_DATA_LENGTH PARALLEL_FOR_SIZE * VECTOR_SIZE_PER_ITERATION
#define OUTPUT_DATA_LENGTH PARALLEL_FOR_SIZE


#define INPUT_DATA_SIZE INPUT_DATA_LENGTH * sizeof(DATA_TYPE)
#define OUTPUT_DATA_SIZE OUTPUT_DATA_LENGTH * sizeof(DATA_TYPE)

// faire un repeat sur les mêmes données pour essayer d'utiliser le cache
// hypothèse : les données sont évincées du cache avant de pouvoir y avoir accès
// observation : j'ai l'impression d'être un peu en train de me perdre dans les explorations,
// avoir une liste pour prioriser ce que je dois faire et 


// SEE main on bench.cpp
// SEE main on bench.cpp
// SEE main on bench.cpp
// SEE main on bench.cpp
// SEE main on bench.cpp
// SEE main on bench.cpp
// SEE main on bench.cpp



// number of iterations - no realloc to make it go faster
extern int REPEAT_COUNT_REALLOC;// défini dans le main (3) - nombre de fois que le test doit être lancé (défini dans le main)
extern int REPEAT_COUNT_ONLY_PARALLEL; // défini dans le main (0)
// Warmup count : nombre d'itérations non comptabilisées pour ne pas mesurer
// les évènements réalisés en lazy.
extern int REPEAT_COUNT_ONLY_PARALLEL_WARMUP_COUNT ; // 4 défini dans le main (0)

extern const bool FORCE_EXECUTION_ON_NAMED_DEVICE ; // go const ?
extern std::string MUST_RUN_ON_DEVICE_NAME ; //"Intel(R) UHD Graphics 620 [0x5917]"; //std::string("s");

// How many times the sum should be repeated
// (to test caches and data access speed)
extern uint REPEAT_COUNT_SUM ;

extern std::string BENCHMARK_VERSION; // Sandor compatible  v05
extern std::string BENCHMARK_VERSION_TRACCC ;
extern std::string DISPLAY_VERSION ;

// nombre de fois qu'il faut répéter le chargement des données
extern unsigned int base_traccc_repeat_load_count ; // actualisé dans utils.h : selector_list_devices
extern unsigned int traccc_repeat_load_count ;

extern int traccc_SPARSITY_MIN ;
extern int traccc_SPARSITY_MAX ;
extern bool traccc_sparsity_ignore ;

struct s_runtime_environment {
public:
    std::string computer_name;
    std::string device_name;
    int device_score;
    int repeat_load_count = 10; // 100 should be good
    int runs_count = 1;
    
    // micro-benchmark
    int total_elements = 1024L * 1024L * 128L; // 128 milions elements * 4 bytes => 512 MiB
    int workitem_L = 128; // nombre of elements per workitem (default = 128)

    std::string get_size_str() {
        uint64_t size_in_bytes = total_elements * sizeof(data_type);
        if (size_in_bytes < 1024) return std::to_string(size_in_bytes) + std::string("B");
        if (size_in_bytes < 1024*1024) return std::to_string(size_in_bytes/1024) + std::string("KiB");
        if (size_in_bytes < 1024*1024*1024) return std::to_string(size_in_bytes/(1024*1024)) + std::string("MiB");
        return std::to_string(size_in_bytes/(1024*1024*1024)) + std::string("GiB");
    }

    // Defined as global variables
    // const int ACAT_START_TEST_INDEX  = 1;
    // const int ACAT_STOP_TEST_INDEX   = 2;
    // const int ACAT_RUN_COUNT         = 1;
};

extern s_runtime_environment runtime_environment;

extern long long total_elements ;
extern std::string g_size_str ;

// /!\ WARNING : do not forget to change to the desired function in the main
// of bench.cpp !

extern std::string ver_indicator ;







struct s_computer {
    std::string fullName, toFileName, deviceName;
    uint repeat_load_count = 0;
    long long total_elements = 0;
    std::string size_str = "0MiB";
    uint L = 1;
};


constexpr uint g_computer_count = 7 ;
extern s_computer g_computers[g_computer_count];

void init_computers()  ;

std::string get_computer_name(uint computer_id) ;

/// For output file name.
std::string get_computer_name_ofile(uint computer_id) ;

std::string get_computer_device_name(uint computer_id) ;

uint get_computer_repeat_load_count(uint computer_id) ;
