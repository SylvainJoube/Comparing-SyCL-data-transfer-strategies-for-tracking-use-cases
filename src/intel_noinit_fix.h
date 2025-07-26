#pragma once

// SyCL specific includes
#include <sycl/sycl.hpp>

inline namespace cl {
    namespace sycl {
        constexpr property::noinit no_init = ::sycl::noinit;
    };
};