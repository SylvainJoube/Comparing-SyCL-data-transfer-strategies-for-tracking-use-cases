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
#include "bench_mems.hpp"
#include "ubench_v2_fcts.h"

int main(int argc, char *argv[])
{

    init_computers();
    log("");
    log("========~~~~~~~ VERSION " + DISPLAY_VERSION + " ~~~~~~~========");
    assert(argc==3) ;
    int device = atoi(argv[1]);
    selector_list_devices_generic dev_list_select2{device};
    cl::sycl::queue temp_queue2(dev_list_select2, exception_handler);
    runtime_environment.repeat_load_count = atoi(argv[2]);
    base_traccc_repeat_load_count = runtime_environment.repeat_load_count;
    log("device: " + std::to_string(device));
    log("repeat_load_count: " + std::to_string(runtime_environment.repeat_load_count));

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

    traccc::run_all_traccc_acat_benchs_generic();

    return 0;
}

