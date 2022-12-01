#include <iostream>
#include <filesystem>
#include <fstream>
#include <chrono>

// file
#include <sys/stat.h>
#include <unistd.h>
#include <string>

// SyCL specific includes
#include <CL/sycl.hpp>
#include <array>
#include <sys/time.h>
#include <stdlib.h>

#include "utils.h"
#include "constants.h"
#include "traccc_fcts.h"
//#include "sycl_helloworld.h"
#include "bench_mems.hpp"
#include "ubench_v2_fcts.h"

int main(int argc, char *argv[])
{

    init_computers();
    log("========~~~~~~~ VERSION " + DISPLAY_VERSION + " ~~~~~~~========");
    log("argc = " + std::to_string(argc));

    select_device_generic(exception_handler);

    log("");
    log("=== Currently running on computer: " + runtime_environment.computer_name + " ===");
    log("=== device: " + runtime_environment.device_name   + " ===\n");

    std::string computerName = runtime_environment.computer_name;

    // if (argc == 1) no argument, only print devices

    // Common variables declaration

    // FORCE_EXECUTION_ON_NAMED_DEVICE = true; set as const
    //MUST_RUN_ON_DEVICE_NAME = "Intel(R) UHD Graphics 620 [0x5917]";

    REPEAT_COUNT_REALLOC = 12; // nombre de fois que le test doit être lancé (défini dans le main)

    REPEAT_COUNT_ONLY_PARALLEL = 0;//12;

    //total_elements = 1024L * 1024L * 256L;   // 256 milions elements * 4 bytes => 1 GiB
    //std::string size_str = "1GiB";
    //total_elements = 1024L * 1024L * 128L; // 128 milions elements * 4 bytes => 512 MiB
    //std::string size_str = "512MiB";
    
    // g_size_str and total_elements are defined in list_devices()

    log("Load count value (traccc): (default = " + std::to_string(runtime_environment.repeat_load_count) + ")");
    std::string in_ld_value;
    std::cin >> in_ld_value;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');

    if (is_number(in_ld_value)) {
        runtime_environment.repeat_load_count = atoi(in_ld_value.data());
        base_traccc_repeat_load_count = runtime_environment.repeat_load_count;
    } else {
        log("WARNING: invalid value: '" + in_ld_value + "'. Will now switch to default value: "
            + std::to_string(runtime_environment.repeat_load_count) + "\n");
    }

    log("Selected load count value (traccc): " + std::to_string(runtime_environment.repeat_load_count));
    press_enter_to_continue();

    // -- version David --
    // Paramétrer l'environnement runtime_environment
    traccc::run_all_traccc_acat_benchs_generic();

    ubench_v2::run_ubench2_tests(runtime_environment.computer_name, runtime_environment.runs_count);

    return 0;



    // Run all tests at once
    // if (argc == 2) {

    //     std::string arg = argv[1];

    //     if (arg.compare("mem") == 0) {
    //         bench_sycl_glibc_mem_speed_main c;
    //         c.main();
    //         return 0;
    //     }

    //     if (arg.compare("mem2") == 0) {
    //         bench_mem_alloc_free b;
    //         b.make_default_values();
    //         b.main_sequence();
    //         return 0;
    //     }

    //     if (arg.compare("mem_test_18GB") == 0) {
    //         bench_mem_alloc_free b;
    //         b.make_default_values();
    //         b.INPUT_INT_COUNT = 1024L * 1024L * 1024L * 18L / 4L; // 18 GiB (4L pour la taille d'un uint)
    //         b.INPUT_OUTPUT_FACTOR = 1024L;
    //         b.refresh_deduced_values();
    //         b.main_sequence();
    //         return 0;
    //     }

    //     if (arg.compare("mem_test_6GB") == 0) {
    //         bench_mem_alloc_free b;
    //         b.make_default_values();
    //         b.INPUT_INT_COUNT = 1024L * 1024L * 1024L * 4L / 4L; // 18 GiB
    //         b.INPUT_OUTPUT_FACTOR = 1024L;
    //         b.refresh_deduced_values();
    //         b.main_sequence();
    //         return 0;
    //     }

    //     if (arg.compare("helloworld") == 0) {
    //         sycl_hello_main();
    //         return 0;
    //     }

    //     // if (arg.compare("traccc") == 0) {
    //     //     std::string runCount = "1";
    //     //     log("=> Run all  TRACCC  tests at once, runCount(" + runCount + ") <=");
    //     //     //traccc::traccc_bench(sycl_mode::glibc);
    //     //     //log("Will now sleep.");
    //     //     //unsigned int microseconds = 10000000;
    //     //     //usleep(microseconds);
    //     //     traccc::run_all_traccc_benchs(computerName + "_AT", std::stoi(runCount));
    //     //     //traccc::traccc_bench(sycl_mode::glibc);
    //     //     //traccc::traccc_bench(sycl_mode::host_USM, traccc::mem_strategy::flatten);
    //     //     return 0;
    //     // }


    //     if (arg.compare("traccc_acat") == 0) {
    //         std::string runCount = "1";
    //         log("=> Run all -ACAT- TRACCC  tests at once, runCount(" + runCount + ") <=");
    //         //traccc::traccc_bench(sycl_mode::glibc);
    //         //log("Will now sleep.");
    //         //unsigned int microseconds = 10000000;
    //         //usleep(microseconds);
    //         traccc::run_all_traccc_acat_benchs(computerName + "_AT", std::stoi(runCount));
    //         //traccc::traccc_bench(sycl_mode::glibc);
    //         //traccc::traccc_bench(sycl_mode::host_USM, traccc::mem_strategy::flatten);
    //         return 0;
    //     }

    //     if (arg.compare("traccc_acat_flat_ld100") == 0) {
    //         std::string runCount = "1";
    //         log("=> Run all -ACAT- TRACCC  tests at once, runCount(" + runCount + ") <=");
    //         //traccc::traccc_bench(sycl_mode::glibc);
    //         //log("Will now sleep.");
    //         //unsigned int microseconds = 10000000;
    //         //usleep(microseconds);
    //         ACAT_REPEAT_LOAD_COUNT = 100;
    //         traccc::run_all_traccc_acat_benchs(computerName + "_AT", std::stoi(runCount));
    //         //traccc::traccc_bench(sycl_mode::glibc);
    //         //traccc::traccc_bench(sycl_mode::host_USM, traccc::mem_strategy::flatten);
    //         return 0;
    //     }
    // }

    // // Run one single test
    // if (argc == 3) {

    //     std::string arg1 = argv[1];
    //     std::string arg2 = argv[2]; // en GiB

    //     if (arg1.compare("mem_test_XGB") == 0) {
    //         bench_mem_alloc_free b;
    //         b.make_default_values();

    //         unsigned long long sz = stoll(arg2) * 1024L * 1024L * 1024L / sizeof(int); // car sizeof(int) = 4

    //         b.INPUT_INT_COUNT = sz; //1024L * 1024L * 1024L * 18L / 4L; // 18 GiB (4L pour la taille d'un uint)
    //         b.INPUT_OUTPUT_FACTOR = 1024L;
    //         b.refresh_deduced_values();
    //         b.main_sequence();
    //         return 0;
    //     }

    //     if (arg1.compare("ubench2") == 0) {
    //         log("sizeof(unsigned long) = " + std::to_string(sizeof(unsigned long)));

    //         if ( ! is_number(arg2) ) { log("ERROR, arg2(" + arg2 + ") as argv[2] is not a number."); return 3; }
    //         ubench_v2::run_ubench2_tests(computerName, stoi(arg2));

    //         return 0;
    //     }

    //     if (std::string(argv[1]).compare("traccc") == 0) {
    //         if (std::string(argv[2]).compare("sparse") == 0) {
    //             traccc::run_single_test_generic_traccc(computerName + "_AT", 5, 1);
    //             traccc::run_single_test_generic_traccc(computerName + "_AT", 6, 1);
    //             return 0;
    //         }
    //         return 0;
    //     }
    // }

    // // Un seul test de traccc
    // if (argc == 4) {

    //     if (std::string(argv[1]).compare("traccc") != 0) {
    //         return 0;
    //     }

    //     std::string testID = argv[2];
    //     std::string runCount = argv[3];

    //     if ( ! is_number(testID) ) {
    //         log("ERROR, testID(" + testID + ") as argv[2] is not a number.");
    //         return 3;
    //     }
    //     if ( ! is_number(runCount) ) {
    //         log("ERROR, runCount(" + runCount + ") as argv[3] is not a number.");
    //         return 3;
    //     }

    //     traccc::run_single_test_generic_traccc(computerName + "_AT", std::stoi(testID), std::stoi(runCount));
    // }

    // if (argc == 7) {
    //      if (std::string(argv[1]).compare("traccc_acat") == 0) {

    //         log("Vension bench finale - 2022-02-09 @ 22h35 ------");
    //         log("Vension bench finale - 2022-02-09 @ 22h35 ------");
    //         log("Vension bench finale - 2022-02-09 @ 22h35 ------");
    //         log("Vension bench finale - 2022-02-09 @ 22h35 ------");
    //         log("Vension bench finale - 2022-02-09 @ 22h35 ------");

    //         std::string start_test_index  = argv[2];
    //         std::string stop_test_index   = argv[3];
    //         std::string run_count         = argv[4];
    //         std::string ld_repeat         = argv[5];
    //         std::string ubench_run_count  = argv[6];
    //         if ( ! is_number(start_test_index) ) { log("ERROR, start_test_index(" + start_test_index + ") as argv[2] is not a number."); return 3; }
    //         if ( ! is_number(stop_test_index) )  { log("ERROR, stop_test_index(" + stop_test_index + ") as argv[3] is not a number."); return 3; }
    //         if ( ! is_number(run_count) )        { log("ERROR, run_count(" + run_count + ") as argv[4] is not a number."); return 3; }
    //         if ( ! is_number(ld_repeat) )        { log("ERROR, ld_repeat(" + ld_repeat + ") as argv[5] is not a number."); return 3; }
    //         if ( ! is_number(ubench_run_count) ) { log("ERROR, ubench_run_count(" + ubench_run_count + ") as argv[6] is not a number."); return 3; }
            
    //         log("=> Run all -ACAT- TRACCC  tests at once <=");
    //         log("start_test_index = " + start_test_index);
    //         log("stop_test_index = " + stop_test_index);
    //         log("run_count = " + run_count);
    //         log("ld_repeat = " + ld_repeat);
    //         log("ubench_run_count = " + ubench_run_count);


    //         // uint previous_ld = g_computers[3].repeat_load_count;
    //         // s_computer* c = &g_computers[3];
    //         for (s_computer & c : g_computers) {
    //             uint previous_ld = c.repeat_load_count;
    //             c.repeat_load_count = std::stoi(ld_repeat);
    //             log("Setting " + c.fullName + " repeat_load_count to " + std::to_string(c.repeat_load_count) + ". Previous value = " + std::to_string(previous_ld));
    //         }

    //         base_traccc_repeat_load_count = std::stoi(ld_repeat);
            
    //         log("=====================================");

    //         //traccc::traccc_bench(sycl_mode::glibc);
    //         //log("Will now sleep.");
    //         //unsigned int microseconds = 10000000;
    //         //usleep(microseconds);
    //         traccc::run_all_traccc_acat_benchs(computerName + "_AT",
    //                                            std::stoi(start_test_index),
    //                                            std::stoi(stop_test_index),
    //                                            std::stoi(run_count));
    //         //traccc::traccc_bench(sycl_mode::glibc);
    //         //traccc::traccc_bench(sycl_mode::host_USM, traccc::mem_strategy::flatten);

    //         return 0;
    //     }
    // }
    
    return 0;
}

// traccc testID runCount 
// traccc 2 1

/*
To run with syclcc, set those variables :
export HIPSYCL_TARGETS="cuda:sm_35" && \
export HIPSYCL_GPU_ARCH="sm_35" && \
export HIPSYCL_CUDA_PATH="/usr/local/cuda-10.1"

On Sandor :
export HIPSYCL_TARGETS="cuda:sm_75" && \
export HIPSYCL_GPU_ARCH="sm_75" && \
export HIPSYCL_CUDA_PATH="/usr/local/cuda-10.1"
*/
