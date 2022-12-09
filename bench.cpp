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
    log("========~~~~~~~ VERSION " + DISPLAY_VERSION + " ~~~~~~~========");
    log("argc = " + std::to_string(argc));
    //assert(argc==3) ;
    runtime_environment.repeat_load_count = atoi(argv[2]);


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

    log("Load count value (traccc): (suggested = " + std::to_string(runtime_environment.repeat_load_count) + ")");
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
}

