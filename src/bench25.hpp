#pragma once

#include <iostream>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <ctime>

// SyCL specific includes
#include <sycl/sycl.hpp>
#include <array>
#include <sys/time.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>

namespace bench25
{
  std::string get_full_date()
  {
    std::time_t timestamp = time(nullptr);
    struct tm datetime = *localtime(&timestamp);
    char output[50];
    strftime(output, 50, "20%y-%m-%d_%Hh%Mm%Ss", &datetime);
    return std::string{output};
  }

  std::string get_computer_name()
  {
    // Does not handle error codes returned by gethostname.
    // As described in: https://man7.org/linux/man-pages/man2/gethostname.2.html
    
    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
    std::string name(hostname);
    return name;
  }
  std::string get_computer()  { return get_computer_name(); }
  std::string get_host_name() { return get_computer_name(); }


  std::string get_user_name()
  {
    // Does not handle error codes returned by getlogin_r.
    // As described in: https://man7.org/linux/man-pages/man2/gethostname.2.html
    
    char username[LOGIN_NAME_MAX];
    getlogin_r(username, LOGIN_NAME_MAX);
    std::string name(username);
    return name;
  }
  std::string get_login() { return get_user_name(); }
  std::string get_user()  { return get_user_name(); }

  std::string fprefix()
  {
    return get_host_name() + "_" + get_full_date() + "_";
  }

  template<typename T>
  T random_float(T min, T max)
  {
    return (static_cast<T>(rand()) / static_cast<T>(RAND_MAX)) * (max - min) + min; 
  }

  enum backend_t { GPU, CPU, UNKNOWN };

  // Indiqué dans les options lors du lancement
  backend_t CHOOSEN_BACKEND = UNKNOWN;

  // Mis en variable globale pour aller plus vite
  bool use_file = false;
  std::ofstream f_log;  // tous les logs, human-readable
  // std::ofstream f_plot; // seulement les infos à plot -> j'essaie de plot selon ce que j'ai déjà fait pour ACAT

  void print_choosen_backend()
  {
    std::cout << "====== print_choosen_backend ";
    if (CHOOSEN_BACKEND == backend_t::CPU) { std::cout << "CPU"; }
    if (CHOOSEN_BACKEND == backend_t::GPU) { std::cout << "GPU"; }
    if (CHOOSEN_BACKEND == backend_t::UNKNOWN) { std::cout << "UNKNOWN"; }
  }
  
}