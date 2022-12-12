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

std::string sys_get_hostname() ;

std::string sys_get_username() ;

int mode_to_int(sycl_mode m) ;

std::string mode_to_string(sycl_mode m) ;

bool is_number(const std::string& s) ;

void press_enter_to_continue() ;

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

std::string padTo(std::string str, const size_t num, const char paddingChar = ' ') ;

/*uint64_t get_ms() {
    auto tm = std::chrono::steady_clock::now();
    std::chrono::duration<double> s = tm - tm;


    struct timeval tp;
    gettimeofday(&tp, NULL);
    uint64_t ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
    return ms;
}*/

void log(std::string str) ;
void logs(std::string str) ;

// level :
// - 0 : important
// - 1 : info
// - 2 : flood

constexpr int MAX_SHOWN_LEVEL = 2 ;

void log(std::string str, int level) ;





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

extern unsigned int global_t_data_generation_and_ram_allocation ;

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

constexpr bool SHOW_TIME_STATS = false;

void print_timer_iter_alloc(gpu_timer& time) ;

void print_timer_iter(gpu_timer& time) ;

void print_timer_alloc(gpu_timer& time) ;

host_dataset* generate_datasets(uint a_DATASET_NUMBER, uint a_INPUT_DATA_LENGTH,
                                uint a_OUTPUT_DATA_LENGTH, bool a_CHECK_SIMD_CPU,
                                uint a_PARALLEL_FOR_SIZE, uint a_VECTOR_SIZE_PER_ITERATION) ;


void delete_datasets(host_dataset* hdata, uint a_DATASET_NUMBER) ;

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

/*
Taken from : https://github.com/codeplaysoftware/computecpp-sdk/blob/master/samples/custom-device-selector.cpp#L46
pointed by the answer https://stackoverflow.com/questions/59061444/how-do-you-make-sycl-default-selector-select-an-intel-gpu-rather-than-an-nvidi
*/

long GetFileSize(std::string filename) ;

inline bool file_exists_test0 (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

