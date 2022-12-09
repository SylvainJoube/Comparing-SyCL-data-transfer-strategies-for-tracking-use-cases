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
#include "constants.h"

#include <unistd.h>
#include <limits.h>

/*
Here are some structs and useful functions that are not meant to change
very often.
*/

std::string sys_get_hostname() {
    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
    return std::string(hostname);
}
std::string sys_get_username() {
    char username[LOGIN_NAME_MAX];
    getlogin_r(username, LOGIN_NAME_MAX);
    return std::string(username);
}


int mode_to_int(sycl_mode m) {
    switch (m) {
    case shared_USM : return 0;
    case device_USM : return 1;
    case host_USM : return 2;
    case accessors : return 3;
    case glibc : return 20;
    }
    return -1;
}

std::string mode_to_string(sycl_mode m) {
    switch (m) {
    case shared_USM : return "shared_USM";
    case device_USM : return "device_USM";
    case host_USM : return "host_USM";
    case accessors : return "accessors";
    case glibc : return "glibc";
    }
    return "unknown";
}



bool is_number(const std::string& s)
{
    std::string::const_iterator it = s.begin();
    while (it != s.end() && std::isdigit(*it)) ++it;
    return !s.empty() && it == s.end();
}

void press_enter_to_continue() {
    
    std::cout << "Please press enter to continue...\n";
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
}



// SyCL asynchronous exception handler
// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](cl::sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

class stime_utils {
private :
    std::chrono::_V2::steady_clock::time_point _start, _stop;

public :
    
    void start() {
        _start = std::chrono::steady_clock::now();
    }

    // Gets the us count since last start or reset.
    uint64_t reset() {
        std::chrono::duration<int64_t, std::nano> dur = std::chrono::steady_clock::now() - _start;
        _start = std::chrono::steady_clock::now();
        int64_t ns = dur.count();
        int64_t us = ns / 1000;
        return us;
    }

};

std::string padTo(std::string str, const size_t num, const char paddingChar = ' ') {
    std::string res = str;
    if(num > res.size())
        res.insert(0, num - str.size(), paddingChar);
    return res;
}

/*uint64_t get_ms() {
    auto tm = std::chrono::steady_clock::now();
    std::chrono::duration<double> s = tm - tm;


    struct timeval tp;
    gettimeofday(&tp, NULL);
    uint64_t ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
    return ms;
}*/

void log(std::string str) {
    std::cout << str << std::endl;
}
void logs(std::string str) {
    std::cout << str << std::flush;
}

// level :
// 0 : important
// 1 : info
// 2 : flood

const int MAX_SHOWN_LEVEL = 1;

void log(std::string str, int level) {
    if (level <= MAX_SHOWN_LEVEL) {
        std::cout << str << std::endl;
    }
}

// Memory intensive operation (read only)
/*int compute_sum(int* array, int size) {
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += array[i];
    }
    return sum;
}*/


//static sycl_mode mode = sycl_mode::device_USM;
// static bool wait_queue = true;

struct host_dataset {
    //bool need_copy; // USM device needs a copy, but not host or shared.
    data_type *data_input;
    data_type *data_output;
    data_type final_result_verif = 0;
    unsigned int seed;
    // Memory allocated on the device : USM only
    data_type *device_input = nullptr;
    data_type *device_output = nullptr;
    // Buffers on the device for accessors-buffers
    // Those are pointers to be created during the allocation phase
    cl::sycl::buffer<data_type, 1> *buffer_input = nullptr;
    cl::sycl::buffer<data_type, 1> *buffer_output = nullptr;
};
unsigned int global_t_data_generation_and_ram_allocation = 0;

struct gpu_timer {
    uint64_t t_data_generation_and_ram_allocation = 0;
    uint64_t t_queue_creation = 0;
    uint64_t t_allocation = 0;

    // used if USE_HOST_SYCL_BUFFER_DMA = true
    uint64_t t_sycl_host_alloc = 0;
    uint64_t t_sycl_host_copy = 0;

    // if USE_HOST_SYCL_BUFFER_DMA, this is malloc_host -> shared/device/host
    // otherwise this is (classic buffer allocated with new) -> shared/device/host
    uint64_t t_copy_to_device = 0;

    uint64_t t_sycl_host_free = 0;

    uint64_t t_parallel_for = 0;
    uint64_t t_read_from_device = 0;
    uint64_t t_free_gpu = 0;
};

class timerv2 {
private:
    static const size_t max_steps_ = 10;
public:
    timerv2(std::string name) {
        for (size_t i = 0; i < max_steps_; ++i) {
            step_time[i] = 0;
        }
        this->name = name;
    }
    size_t step_time[max_steps_];
    std::string name;

    void print() {
        logs(padTo(name, 12) + " : ");
        for (size_t i = 1; i < 6; ++i) {
            logs(padTo(std::to_string(step_time[i]), 12) + " | ");
        }
        log("");
    }

    // Etape 1 : allocation de la mémoire SYCL / stdlib.
    // Etape 2 : copie (explicite) de la mémoire host vers la mémoire de l'étape 1.
    // Etape 3 : sommes partielles device / CPU
    // Etape 4 : copie (explicite) vers la mémoire stdlib host
    // Etape 5 : libération de la mémoire de l'étape 1
    void print_header() {
        logs(padTo("noms", 12) + " : ");
        logs(padTo("Allocation", 12) + " | ");
        logs(padTo("Copie ->", 12) + " | ");
        logs(padTo("Calcul", 12) + " | ");
        logs(padTo("Copie <-", 12) + " | ");
        logs(padTo("Free mem", 12) + " | ");
        log("");
    }
};

const bool SHOW_TIME_STATS = false;

void print_timer_iter_alloc(gpu_timer& time) {
    if ( ! SHOW_TIME_STATS ) return;
    uint64_t t_gpu;
    t_gpu = time.t_allocation + time.t_copy_to_device + time.t_read_from_device
            + time.t_parallel_for + time.t_free_gpu;
    std::cout 
            << "t_gpu - - - - - - - - - -  = " << t_gpu << std::endl
            //<< "t_queue_creation           = " << t_queue_creation << std::endl
            << "t_allocation - - - - - - - = " << time.t_allocation << std::endl
            << "t_copy_to_device           = " << time.t_copy_to_device << std::endl
            << "t_parallel_for - - - - - - = " << time.t_parallel_for << std::endl
            << "t_read_from_device         = " << time.t_read_from_device << std::endl
            << "t_free_gpu - - - - - - - - = " << time.t_free_gpu << std::endl
            ;

    log("");
}

void print_timer_iter(gpu_timer& time) {
    if ( ! SHOW_TIME_STATS ) return;
    //uint64_t t_gpu;
    //t_gpu = time.t_read_from_device + time.t_parallel_for;
    std::cout 
            << "t_parallel_for - - - - - - = " << time.t_parallel_for << std::endl
            << "t_read_from_device         = " << time.t_read_from_device << std::endl
            ;

    log("");
}

void print_timer_alloc(gpu_timer& time) {
    if ( ! SHOW_TIME_STATS ) return;
    uint64_t t_alloc_and_free;
    t_alloc_and_free = time.t_allocation + time.t_copy_to_device + time.t_free_gpu;
    std::cout 
            << "t_alloc_and_free - - - - - = " << t_alloc_and_free << std::endl
            << "t_allocation - - - - - - - = " << time.t_allocation << std::endl
            << "t_copy_to_device           = " << time.t_copy_to_device << std::endl
            << "t_free_gpu - - - - - - - - = " << time.t_free_gpu << std::endl
            ;

    log("");
}

host_dataset* generate_datasets(uint a_DATASET_NUMBER, uint a_INPUT_DATA_LENGTH,
                                uint a_OUTPUT_DATA_LENGTH, bool a_CHECK_SIMD_CPU,
                                uint a_PARALLEL_FOR_SIZE, uint a_VECTOR_SIZE_PER_ITERATION) {

    log("Generating data...", 1);
    stime_utils chrono;
    chrono.start();

    host_dataset *hdata = new host_dataset[a_DATASET_NUMBER];

    for (int i = 0; i < a_DATASET_NUMBER; ++i) {
        host_dataset *hd = &hdata[i];

        hd->data_input = new data_type[a_INPUT_DATA_LENGTH];
        hd->data_output = new data_type[a_OUTPUT_DATA_LENGTH];
        hd->seed = 452 + i * 68742;

        srand(hd->seed);

        // Fills the array with random data
        for (int i = 0; i < a_INPUT_DATA_LENGTH; ++i) {
            data_type v = rand();
            hd->data_input[i] = v;
            hd->final_result_verif += v;
        }

        // Perform SMID-like operations to verify the sum algorithm
        if (a_CHECK_SIMD_CPU) {
            data_type sum_simd_check_cpu = 0;
            // SIMD-like check
            for (int ip = 0; ip < a_PARALLEL_FOR_SIZE; ++ip) {
                for (int it = 0; it < a_VECTOR_SIZE_PER_ITERATION; ++it) {
                    int iindex = ip + it * a_PARALLEL_FOR_SIZE;
                    sum_simd_check_cpu += hd->data_input[iindex];
                }
            }

            // SMID-like OKAY VALLID - total sum = -1553315753
            if (sum_simd_check_cpu == hd->final_result_verif) {
                std::cout << "SMID-like OKAY VALLID - total sum = " << sum_simd_check_cpu << std::endl;
                std::cout << "SMID-like OKAY VALLID - total sum = " << sum_simd_check_cpu << std::endl;
                std::cout << "SMID-like OKAY VALLID - total sum = " << sum_simd_check_cpu << std::endl;
            } else {
                std::cout << "SMID-like ERROR should be " << hd->final_result_verif << " but is " << sum_simd_check_cpu << " - ERROR ERROR" << std::endl;
                std::cout << "SMID-like ERROR should be " << hd->final_result_verif << " but is " << sum_simd_check_cpu << " - ERROR ERROR" << std::endl;
                std::cout << "SMID-like ERROR should be " << hd->final_result_verif << " but is " << sum_simd_check_cpu << " - ERROR ERROR" << std::endl;
            }
        }
    }

    global_t_data_generation_and_ram_allocation = chrono.reset(); //get_ms() - t_start;

    return hdata;
}


void delete_datasets(host_dataset* hdata, uint a_DATASET_NUMBER) {
    if (hdata == nullptr) return;

    for (int i = 0; i < a_DATASET_NUMBER; ++i) {
        host_dataset *hd = &hdata[i];
        delete[] hd->data_input;
        delete[] hd->data_output;
    }
    delete[] hdata;
}

int total_main_seq_runs = 1;
//int current_main_seq_runs = 0;
int current_iteration_count = 0;

void init_progress() {
    total_main_seq_runs = 1;
    current_iteration_count = 0;
}

void print_total_progress() {
    const int total_iteration_count_per_seq = DATASET_NUMBER * (REPEAT_COUNT_REALLOC + REPEAT_COUNT_ONLY_PARALLEL
    + ( (REPEAT_COUNT_ONLY_PARALLEL == 0) ? 0 : REPEAT_COUNT_ONLY_PARALLEL_WARMUP_COUNT) );

    /*logs( "total_iteration_count_per_seq(" + std::to_string(total_iteration_count_per_seq) + ")"
    + " " +  );*/
    // dataset number * (number of iterations total, with warmup time if applicable)
    int total_iteration_count = total_iteration_count_per_seq * total_main_seq_runs;
    int progress = 100 * double(current_iteration_count) / double(total_iteration_count);
    logs( std::to_string(progress) + "% ");
}

// Only show once
static bool only_show_once_right_device_found_has_been_found = false;

/* Classes can inherit from the device_selector class to allow users
 * to dictate the criteria for choosing a device from those that might be
 * present on a system. This example looks for a device with SPIR support
 * and prefers GPUs over CPUs. */
// Selects the device named MUST_RUN_ON_DEVICE_NAME.
class custom_device_selector : public cl::sycl::device_selector {
private:
    cl::sycl::default_selector def_selector;
public:
    custom_device_selector() : cl::sycl::device_selector() {}

    /* The selection is performed via the () operator in the base
    * selector class.This method will be called once per device in each
    * platform. Note that all platforms are evaluated whenever there is
    * a device selection. */
    int operator()(const cl::sycl::device& device) const override {
        
        if ( ! FORCE_EXECUTION_ON_NAMED_DEVICE ) {
            // Use the default recommended device
            return def_selector(device);
        } else {
            // Use the specified device
            // Multiple devices may have the same name but not the same score
            // so I return the device score if it has the right name
            // and -1 (i.e. never choose this one) otherwise.
            std::string devName =  device.get_info<cl::sycl::info::device::name>();
            if (devName.compare(MUST_RUN_ON_DEVICE_NAME) == 0) {
                int devScore = def_selector(device);
                if ( ! only_show_once_right_device_found_has_been_found) {
                    log("Right device found, score(" + std::to_string(devScore) + ") - " + devName);
                    only_show_once_right_device_found_has_been_found = true;
                }
                return devScore;
            }
            return -1;
        }
    }
};

// Generic device selector that asks the user to choose on which device to
// run the benchmark.
class selector_list_devices_generic : public cl::sycl::device_selector {
private:
    cl::sycl::default_selector def_selector;
    // print_devices=true  => 1) print devices and asks the user to pick a score
    // print_devices=false => 2) silently selects the desired device for running the benchmark
    bool print_devices    = true;
    int  choosen_score;
public:
    selector_list_devices_generic() : cl::sycl::device_selector() {}
    selector_list_devices_generic(int _choosen_score) : cl::sycl::device_selector(), choosen_score(_choosen_score) {
        log("choosen score : " + std::to_string(choosen_score));
        print_devices = false;
    }

    /* The selection is performed via the () operator in the base
    * selector class.This method will be called once per device in each
    * platform. Note that all platforms are evaluated whenever there is
    * a device selection. */
    int operator()(const cl::sycl::device& device) const override {
        
        // List device names and return the default score for the device
        std::string devName =  device.get_info<cl::sycl::info::device::name>();
        if (print_devices) logs(devName);

        std::string devType = "";
        switch (device.get_info<cl::sycl::info::device::device_type>()) {
        case cl::sycl::info::device_type::cpu :  devType = "cpu"; break;
        case cl::sycl::info::device_type::gpu :  devType = "gpu"; break;
        case cl::sycl::info::device_type::host : devType = "host"; break;
        default : devType = "unknown type"; break;
        }
        if (print_devices) logs(" (" + devType + ")");

        int defaultScore = def_selector(device);

        // log("compare score(" + std::to_string(defaultScore) + ") - choosen(" + std::to_string(choosen_score) + ")");

        if ( (! print_devices) && (defaultScore == choosen_score) ) {

            // base_traccc_repeat_load_count = c->repeat_load_count;

            // total_elements = c->total_elements;
            // g_size_str = c->size_str;
            // BASE_VECTOR_SIZE_PER_ITERATION = c->L;
            log("Device selected: " + devName);
            runtime_environment.computer_name = sys_get_hostname();
            runtime_environment.device_name   = devName;
            runtime_environment.device_score  = choosen_score;
            // FORCE_EXECUTION_ON_NAMED_DEVICE   = true; set as const
            MUST_RUN_ON_DEVICE_NAME           = devName;
            ERR_DEVICE_NOT_FOUND              = false;
        }

        if (print_devices) log(" - score " + std::to_string(defaultScore));

        // Return the default device score
        return defaultScore;
    }
};

void select_device_generic(std::function<void(cl::sycl::exception_list)> func) {
    log("Computer name: " + sys_get_hostname());
    log("== List of available devices ==");
    selector_list_devices_generic dev_list_select{};
    cl::sycl::queue temp_queue(dev_list_select, func);
    
    log("\nPlease provide the score of the device you wish to use:");
    std::string input;
    std::cin >> input;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    if (! is_number(input) ) {
        log("ERROR: '" + input + "' is not a number.");
        log("        Program stops here.");
        log("        Please provide a score matching a device.");
        exit(0);
    }

    int value = stoi(input);
    log("Provided: '" + std::to_string(value) + "'");

    selector_list_devices_generic dev_list_select2{value};
    cl::sycl::queue temp_queue2(dev_list_select2, func);

    if ( ERR_DEVICE_NOT_FOUND ) {
        log("ERROR : unable to find the desired device.");
        log("        Program stops here.");
        log("        Please provide a score matching a device.");
        exit(0);
    }

    press_enter_to_continue();

}

/*
Taken from : https://github.com/codeplaysoftware/computecpp-sdk/blob/master/samples/custom-device-selector.cpp#L46
pointed by the answer https://stackoverflow.com/questions/59061444/how-do-you-make-sycl-default-selector-select-an-intel-gpu-rather-than-an-nvidi
*/

long GetFileSize(std::string filename)
{
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

inline bool file_exists_test0 (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

