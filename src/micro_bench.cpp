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
#include "progress.h"
#include "constants.h"
#include "bench_mems.hh"
#include "ubench_v2_fcts.h"

int main(int argc, char *argv[])
{

    init_computers();

    log("");
    log("========~~~~~~~ VERSION " + DISPLAY_VERSION + " ~~~~~~~========");

    assert(argc>1) ;
    assert(argc<5) ;
    int arg_score = atoi(argv[1]);
    unsigned long arg_gb = 1UL ;
    if (argc>2)
     { arg_gb = atoi(argv[2]) ; }
    unsigned int arg_repeat = 12 ;
    if (argc>3)
     { arg_repeat = atoi(argv[3]) ; }

    selector_list_devices_generic dev_list_select2{arg_score};
    cl::sycl::queue temp_queue2(dev_list_select2, exception_handler);
    log("device score: " + std::to_string(arg_score));
    log("data size: " + std::to_string(arg_gb) + "Gb");
    log("recompute: " + std::to_string(arg_repeat) + " times");

    log("");
    log("=== Currently running on computer: " + runtime_environment.computer_name + " ===");
    log("=== device: " + runtime_environment.device_name   + " ===\n");

    std::string computerName = runtime_environment.computer_name;

    // if (argc == 1) no argument, only print devices

    // Common variables declaration

    // FORCE_EXECUTION_ON_NAMED_DEVICE = true; set as const
    //MUST_RUN_ON_DEVICE_NAME = "Intel(R) UHD Graphics 620 [0x5917]";

    REPEAT_COUNT_REALLOC = arg_repeat ;

    REPEAT_COUNT_ONLY_PARALLEL = 0 ;

    //total_elements = 1024L * 1024L * 256L;   // 256 milions elements * 4 bytes => 1 GiB
    //std::string size_str = "1GiB";
    //total_elements = 1024L * 1024L * 128L; // 128 milions elements * 4 bytes => 512 MiB
    //std::string size_str = "512MiB";
    
    ubench_v2::init_data_length(arg_gb) ;
    ubench_v2::run_ubench2_tests(runtime_environment.computer_name, runtime_environment.runs_count);

    return 0;
}

