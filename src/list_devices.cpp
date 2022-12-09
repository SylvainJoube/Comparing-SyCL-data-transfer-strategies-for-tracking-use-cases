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
//#include "constants.h"
//#include "traccc_fcts.h"
//#include "bench_mems.hh"
//#include "ubench_v2_fcts.h"

int main(int argc, char *argv[])
 {
  init_computers() ;
  std::cout<<std::endl ;
  selector_list_devices_generic dev_list_select {} ;
  cl::sycl::queue q { dev_list_select,exception_handler } ;
  std::cout<<std::endl ;
  return 0 ;
 }

