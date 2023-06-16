#pragma once
#include <iostream>

#include "common/logger.hpp"

using namespace std;

namespace tt::test_utils::df {

//! Custom type is supported as long as the custom type supports the following custom functions
//! static SIZEOF - indicates byte size of custom type
//! to_float() - get float value from custom type
//! to_packed() - get packed (into an integral type that is of the bitwidth specified by SIZEOF)
//! Constructor(float in) - constructor with a float as the initializer

class float32 {
   private:
    uint32_t uint32_data;

   public:
    static const size_t SIZEOF = 4;

    float32() : uint32_data(0) {}

    // create from float: no rounding, just truncate
    float32(float float_num) {
        uint32_t uint32_data;
        tt::log_assert(sizeof float_num == sizeof uint32_data, "Can only support 32bit fp");
        uint32_data = *reinterpret_cast<uint32_t*>(&float_num);
        // just move upper 16 to lower 16 (truncate)
        uint32_data = (uint32_data >> 16);
        // store lower 16 as 16-bit uint
        uint32_data = (uint32_t)uint32_data;
    }

    // store lower 16 as 16-bit uint
    float32(uint32_t uint32_data) { uint32_data = (uint32_t)uint32_data; }

    float to_float() const {
        // move lower 16 to upper 16 (of 32)
        uint32_t uint32_data = (uint32_t)uint32_data << 16;
        // return 32 bits as float
        return *reinterpret_cast<float*>(&uint32_data);
    }
    uint32_t to_packed() const { return uint32_data; }
    bool operator==(float32 rhs) { return uint32_data == rhs.uint32_data; }
    bool operator!=(float32 rhs) { return uint32_data != rhs.uint32_data; }
};

inline ostream& operator<<(ostream& os, const float32& val) {
    os << val.to_packed();
    return os;
}
}  // namespace tt::test_utils::df
