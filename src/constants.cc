
#include "constants.h"

const int ACAT_START_TEST_INDEX  = 1;
const int ACAT_STOP_TEST_INDEX   = 2;
const int ACAT_RUN_COUNT         = 1;
int ACAT_REPEAT_LOAD_COUNT = 10;

unsigned long long PARALLEL_FOR_SIZE;// = 1024 * 32 * 8;// = M ; work items number
unsigned long long VECTOR_SIZE_PER_ITERATION;// = 1; // = L ; vector size per workitem (i.e. parallel_for task) = nb itérations internes par work item
unsigned long long BASE_VECTOR_SIZE_PER_ITERATION; // updated in list_devices()

sycl_mode CURRENT_MODE = sycl_mode::device_USM;

int MEMCOPY_IS_SYCL = 1;
int SIMD_FOR_LOOP = 1;
int USE_HOST_SYCL_BUFFER_DMA = 0; 

bool ERR_DEVICE_NOT_FOUND = true;

// number of iterations - no realloc to make it go faster
int REPEAT_COUNT_REALLOC;// défini dans le main (3) - nombre de fois que le test doit être lancé (défini dans le main)
int REPEAT_COUNT_ONLY_PARALLEL; // défini dans le main (0)
// Warmup count : nombre d'itérations non comptabilisées pour ne pas mesurer
// les évènements réalisés en lazy.
int REPEAT_COUNT_ONLY_PARALLEL_WARMUP_COUNT = 0; // 4 défini dans le main (0)

const bool FORCE_EXECUTION_ON_NAMED_DEVICE = true; // go const ?
std::string MUST_RUN_ON_DEVICE_NAME = "<unknown device>"; //"Intel(R) UHD Graphics 620 [0x5917]"; //std::string("s");

// How many times the sum should be repeated
// (to test caches and data access speed)
uint REPEAT_COUNT_SUM = 1;

std::string BENCHMARK_VERSION = "ubench" + std::to_string(DATA_VERSION); // Sandor compatible  v05
std::string BENCHMARK_VERSION_TRACCC = "sparseccl" + std::to_string(DATA_VERSION_TRACCC);
std::string DISPLAY_VERSION = BENCHMARK_VERSION_TRACCC + " - TRACCC-016";

// Not used anymore
//std::string TRACCC_OUT_FNAME = "tracccMemLocStrat7_sansGraphPtr";

// nombre de fois qu'il faut répéter le chargement des données
unsigned int base_traccc_repeat_load_count = 1; // actualisé dans utils.h : selector_list_devices
unsigned int traccc_repeat_load_count = 1;

int traccc_SPARSITY_MIN = 0;
int traccc_SPARSITY_MAX = 100000;
bool traccc_sparsity_ignore = true;

s_runtime_environment runtime_environment;

//const long long total_elements = 1024L * 1024L * 256L * 8L; // 8 GiB
//const long long total_elements = 1024L * 1024L * 256L * 6L; // 6 GiB
long long total_elements = 1024L * 1024L * 256L; // 1 GiB
std::string g_size_str = "0MiB";
//const long long total_elements = 1024L * 1024L * 128L; // 512 MiB
// 256 => 1 GiB 
// 128 => 512 MiB ; 
// 32  => 128 MiB ; 
// 256 * 4 bytes = 1   GiB.
// 32  * 4 bytes = 128 MiB.

// /!\ WARNING : do not forget to change to the desired function in the main
// of bench.cpp !

std::string ver_indicator = std::string("13d");

s_computer g_computers[g_computer_count];

void init_computers() {
    s_computer * c;
    uint ci = 0;

    // 1
    c = &g_computers[ci++];
    c->fullName   = "Thinkpad";
    c->toFileName = "thinkpad";
    c->deviceName = "Intel(R) UHD Graphics 620 [0x5917]";
    c->repeat_load_count = 1;
    c->total_elements = 1024L * 1024L * 128L; // 128 milions elements * 4 bytes => 512 MiB
    c->size_str = "512MiB";
    c->L = 128;

    // 2
    c = &g_computers[ci++];
    c->fullName   = "MSI_Intel";
    c->toFileName = "msiIntel";
    c->deviceName = "???";
    c->repeat_load_count = 1;
    c->total_elements = 1024L * 1024L * 128L; // 128 milions elements * 4 bytes => 512 MiB
    c->size_str = "512MiB";
    c->L = 128;


    // 3
    c = &g_computers[ci++];
    c->fullName   = "MSI_Nvidia";
    c->toFileName = "msiNvidia";
    c->deviceName = "NVIDIA GeForce GTX 960M";
    c->repeat_load_count = 1;
    c->total_elements = 1024L * 1024L * 128L; // 128 milions elements * 4 bytes => 512 MiB
    c->size_str = "512MiB";
    c->L = 128;


    // 4
    c = &g_computers[ci++];
    c->fullName   = "Sandor";
    c->toFileName = "sandor";
    c->deviceName = "Quadro RTX 5000";
    c->repeat_load_count = 100; // TEMP ACAT
    c->total_elements = 1024L * 1024L * 256L * 6L; // 256 milions elements * 4 bytes => 1 GiB ; *6 => 6 GiB
    c->size_str = "6GiB";
    c->L = 128;


    // 5
    c = &g_computers[ci++];
    c->fullName   = "Blop_Intel";
    c->toFileName = "blopIntel";
    c->deviceName = "????";
    c->repeat_load_count = 1;
    c->total_elements = 1024L * 1024L * 128L; // 128 milions elements * 4 bytes => 512 MiB
    c->size_str = "512MiB";
    c->L = 128;

    // 6
    c = &g_computers[ci++];
    c->fullName   = "Blop_Nvidia";
    c->toFileName = "blopNvidia";
    c->deviceName = "GeForce GTX 780";//"NVIDIA GeForce GTX 780";
    c->repeat_load_count = 10; // benchs ACAT
    c->total_elements = 1024L * 1024L * 128L; // 128 milions elements * 4 bytes => 512 MiB
    c->size_str = "512MiB";
    c->L = 128;

    // 7
    c = &g_computers[ci++];
    c->fullName   = "ls-cassidi";
    c->toFileName = "cassidi";
    c->deviceName = "NVIDIA RTX A6000";
    c->repeat_load_count = 10; // benchs ACAT
    c->total_elements = 1024L * 1024L * 256L * 6L; // 256 milions elements * 4 bytes => 1 GiB ; *6 => 6 GiB
    c->size_str = "6GiB";
    c->L = 128;


}

std::string get_computer_name(uint computer_id) {
    if ( (computer_id > g_computer_count) || (computer_id == 0) )
        return "unknown_computer_id" + std::to_string(computer_id);
    
    return g_computers[computer_id - 1].fullName;
    
    /*switch (computer_id) {
    case 1 : return "Thinkpad";
    case 2 : return "MSI_Intel";
    case 3 : return "MSI_Nvidia";
    case 4 : return "Sandor";
    case 5 : return "Blop_Intel";
    case 6 : return "Blop_Nvidia";
    default : return "unknown_computer";
    }*/
}

/// For output file name.
std::string get_computer_name_ofile(uint computer_id) {
    if ( (computer_id > g_computer_count) || (computer_id == 0) )
        return "unknownComputerId" + std::to_string(computer_id);
    
    return g_computers[computer_id - 1].toFileName;

    /*switch (computer_id) {
    case 1 : return "thinkpad";
    case 2 : return "msiIntel";
    case 3 : return "msiNvidia";
    case 4 : return "sandor";
    case 5 : return "blopIntel";
    case 6 : return "blopNvidia";
    default : return "unknownComputer";
    }*/
}

std::string get_computer_device_name(uint computer_id) {
    if ( (computer_id > g_computer_count) || (computer_id == 0) )
        return "unknown_device_name_computer_id" + std::to_string(computer_id);
    
    return g_computers[computer_id - 1].deviceName;
}

uint get_computer_repeat_load_count(uint computer_id) {
    if ( (computer_id > g_computer_count) || (computer_id == 0) )
        return 0;
    
    return g_computers[computer_id - 1].repeat_load_count;
}
