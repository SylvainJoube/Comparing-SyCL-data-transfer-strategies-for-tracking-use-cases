#include "utils.h"

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

std::string padTo(std::string str, const size_t num, const char paddingChar ) {
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

unsigned int global_t_data_generation_and_ram_allocation = 0;

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

