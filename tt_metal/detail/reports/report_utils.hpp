#include "common/utils.hpp"

namespace tt::tt_metal {

namespace detail {

inline const std::string &metal_reports_dir() {
    static const std::string reports_path = tt::utils::get_reports_dir() + "tt_metal/";
    return reports_path;
}

}   // namespace detail

}   // namespace tt::tt_metal
