#ifndef OPERON_VERSION_HPP
#define OPERON_VERSION_HPP

#include "buildinfo.hpp"

#include <string>
#include <fmt/format.h>

namespace Operon {

    std::string Version() {
        fmt::memory_buffer buf;
        fmt::format_to(buf, "Operon {} {} {} {}\n", OPERON_REVISION, OPERON_BUILD, OPERON_PLATFORM, OPERON_ARCH);
        return std::string(buf.begin(), buf.end());
    };
}

#endif
