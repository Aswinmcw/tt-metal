#pragma once

/*
* Device-side debug print API for device kernels.
* Works on either one of NC/BR/TR threads.
* On device the use is as follows:
*
* DPRINT << SETW(2) << 0 << 0.1f << "string" << ENDL();
*
* This DebugPrinter object can be created multiple times.
*
* On the host it's required to start the print server first, otherwise the behavior will be incorrect.
* This is because the host print server writes a special value that is used in DebugPrinter() constructor
* to initialize the read/write pointers to 0 only once.
* It is also needed to empty the print buffer, otherwise device code will stall waiting on the host to flush it.
*
* Use llrt/tt_debug_print_server.h APIs to start the host-side print server.
*
*/

#include <cstdint>
#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC)
// TODO(AP): this ifdef doesn't seem to make sense given we include risc_common.h
// The issue is some files included inside risc_common.h only apply to NC/BRISCS
// But moving this ifdef inside of the header breaks other code
// So there are some not fully decoupled dependencies in this header.
#include "risc_common.h"
#endif
#include "hostdevcommon/debug_print_common.h"

#define DPRINT DebugPrinter()
#define ATTR_ALIGN4 __attribute__((aligned(4)))
#define ATTR_PACK __attribute__((packed))

struct F16 { uint16_t val; F16(uint16_t val) : val(val) {} } ATTR_PACK;
struct F32 { float val; F32(float val) : val(val) {} } ATTR_PACK;
struct U32 { uint32_t val; U32(uint32_t val) : val(val) {} } ATTR_PACK;

struct ENDL { char tmp; } ATTR_PACK; // Analog of cout << std::endl - not making it zero size to avoid special cases
struct SETW { char w; SETW(char wa) : w(wa) {} } ATTR_PACK; // Analog of cout << std::setw()
struct SETP { char p; SETP(char pa) : p(pa) {} } ATTR_PACK; // Analog of cout << std::setprecision()
struct FIXP { char tmp; } ATTR_PACK; // Analog of cout << std::fixed
struct HEX { char tmp; } ATTR_PACK; // Analog of cout << std::hex

//
// Samples count values of a tile itile at cb with value offset and stride.
// sampling happens relative to the current CB read or write pointer.
// This means that for printing a tile read from the front of the CB,
// the DPRINT << TILESAMPLES(...) call has to occur after cb_wait_front and before cb_pop_front
// For this case, bool packer=true needs to be passed as an argument
// For the case of printing a tile from the back of the CB, packer=false needs to passed and
// the DPRINT << TILESAMPLES(...) call has to occur after cb_reserve_back and before cb_push_back.
//
template<int MAXSAMPLES>
struct TileSamples : TileSamplesHostDev<MAXSAMPLES> {
    inline int min_(int a, int b) { return a < b ? a : b; } // to avoid inclusion of <algorithm>
    inline TileSamples(bool packer, int cb, int itile, int count, int offs, int stride) {
        struct TILE { uint16_t vals[1024] __attribute__((packed)); } __attribute__((aligned(2)));
        this->count_ = count;
        volatile TILE* t;
        if (packer) { // note that front is not valid in packer and back is not valid/initialized in packer
            this->ptr_ = cb_write_interface[cb].fifo_wr_ptr<<4;
            t = (reinterpret_cast<volatile TILE*>(this->ptr_) + itile); // front of q is reader/consumer
        } else {
            this->ptr_ = cb_read_interface[cb].fifo_rd_ptr<<4;
            t = (reinterpret_cast<volatile TILE*>(this->ptr_) + itile); // back of q is writer/producer
        }
        for (int j = 0; j < min_(MAXSAMPLES, count); j++)
            this->samples_[j] = t->vals[offs + stride*j];
    }
};
using TILESAMPLES8 = TileSamples<8>; // max 8 values
using TILESAMPLES32 = TileSamples<32>; // max 32 values

// These primitives are intended for ordering debug prints
// A possible use here is to synchronize debug print order between cores/harts
// It could be implemented, for instance as code = linearize({x,y})*5 + hart_id
// With another core/hart waiting on that index
struct RAISE { uint32_t code; RAISE(uint32_t val) : code(val) {} } ATTR_PACK; // raise a condition with specified code
struct WAIT { uint32_t code; WAIT(uint32_t val) : code(val) {} } ATTR_PACK; // wait for a condition with specified code

// didn't want to include string.h
inline uint32_t DebugPrintStrLen(const char* val) {
    const char* end = val;
    while (*end) { end++; };
    return uint32_t(end-val)+1;
}

inline uint32_t DebugPrintStrCopy(char* dst, const char* src) {
    uint32_t len = DebugPrintStrLen(src);
    for (uint32_t j = 0; j < len; j++)
        dst[j] = src[j];
    return len;
}

// Extend with new type id here, each new type needs specializations for 1 (or 3) of these functions below:
// This template instantiation maps from type to type id to send over our comm channel
template<typename T> uint8_t DebugPrintTypeToId();
template<typename T> uint32_t DebugPrintTypeToSize(T val) { return sizeof(T); };
template<typename T> const uint8_t* DebugPrintTypeAddr(T* val) { return reinterpret_cast<const uint8_t*>(val); }

template<> uint8_t DebugPrintTypeToId<const char*>() { return DEBUG_PRINT_TYPEID_CSTR; }
template<> uint8_t DebugPrintTypeToId<char*>() { return DEBUG_PRINT_TYPEID_CSTR; }
template<> uint8_t DebugPrintTypeToId<ENDL>() { return DEBUG_PRINT_TYPEID_ENDL; }
template<> uint8_t DebugPrintTypeToId<SETW>() { return DEBUG_PRINT_TYPEID_SETW; }
template<> uint8_t DebugPrintTypeToId<uint32_t>() { return DEBUG_PRINT_TYPEID_UINT32; }
template<> uint8_t DebugPrintTypeToId<float>() { return DEBUG_PRINT_TYPEID_FLOAT32; }
template<> uint8_t DebugPrintTypeToId<char>() { return DEBUG_PRINT_TYPEID_CHAR; }
template<> uint8_t DebugPrintTypeToId<RAISE>() { return DEBUG_PRINT_TYPEID_RAISE; }
template<> uint8_t DebugPrintTypeToId<WAIT>() { return DEBUG_PRINT_TYPEID_WAIT; }
template<> uint8_t DebugPrintTypeToId<F16>() { return DEBUG_PRINT_TYPEID_FLOAT16; }
template<> uint8_t DebugPrintTypeToId<SETP>() { return DEBUG_PRINT_TYPEID_SETP; }
template<> uint8_t DebugPrintTypeToId<FIXP>() { return DEBUG_PRINT_TYPEID_FIXP; }
template<> uint8_t DebugPrintTypeToId<HEX>() { return DEBUG_PRINT_TYPEID_HEX; }
template<> uint8_t DebugPrintTypeToId<F32>() { return DEBUG_PRINT_TYPEID_FLOAT32; }
template<> uint8_t DebugPrintTypeToId<U32>() { return DEBUG_PRINT_TYPEID_UINT32; }
template<> uint8_t DebugPrintTypeToId<int>() { return DEBUG_PRINT_TYPEID_INT32; }
template<> uint8_t DebugPrintTypeToId<TILESAMPLES8>() { return DEBUG_PRINT_TYPEID_TILESAMPLES8; }
template<> uint8_t DebugPrintTypeToId<TILESAMPLES32>() { return DEBUG_PRINT_TYPEID_TILESAMPLES32; }
static_assert(sizeof(int) == 4);

// Specializations for const char* (string literals), typically you will not need these for other types
template<> uint32_t DebugPrintTypeToSize<const char*>(const char* val) { return DebugPrintStrLen(val); } // also copy the terminating zero
template<> const uint8_t* DebugPrintTypeAddr<const char*>(const char** val) { return reinterpret_cast<const uint8_t*>(*val); }
template<> uint32_t DebugPrintTypeToSize<char*>(char* val) { return DebugPrintStrLen(val); } // also copy the terminating zero
template<> const uint8_t* DebugPrintTypeAddr<char*>(char** val) { return reinterpret_cast<const uint8_t*>(*val); }

struct DebugPrinter {
    volatile uint32_t* wpos() {
        auto printbuf = get_debug_print_buffer();
        return &reinterpret_cast<DebugPrintMemLayout*>(printbuf)->aux.wpos;
    }
    volatile uint32_t* rpos() {
        auto printbuf = get_debug_print_buffer();
        return &reinterpret_cast<DebugPrintMemLayout*>(printbuf)->aux.rpos;
    }
    uint8_t* buf() { return get_debug_print_buffer(); }
    uint8_t* data() { return reinterpret_cast<DebugPrintMemLayout*>(buf())->data; }
    uint8_t* bufend() { return buf() + PRINT_BUFFER_SIZE; }

    DebugPrinter() {
        if (*wpos() == DEBUG_PRINT_SERVER_STARTING_MAGIC) {
            // Host debug print server writes this value
            // we don't want to reset wpos/rpos to 0 unless this is the first time
            // DebugPrinter() is created (even across multiple kernel calls)
            *wpos() = 0;
            *rpos() = 0;
        }
    }
};

template<typename T>
DebugPrinter operator <<(DebugPrinter dp, T val) {

    if (*dp.wpos() == DEBUG_PRINT_SERVER_DISABLED_MAGIC) {
        // skip all prints if this hart+core was not specifically enabled on the host
        return dp;
    }

    uint32_t payload_sz = DebugPrintTypeToSize<T>(val); // includes terminating 0 for char*
    uint8_t typecode = DebugPrintTypeToId<T>();

    constexpr int code_sz = 1; // size of type code
    constexpr int sz_sz = 1; // size of serialized size
    uint32_t wpos = *dp.wpos(); // copy wpos into local storage
    auto sum_sz = payload_sz + code_sz + sz_sz;
    if (dp.data() + wpos + sum_sz >= dp.bufend()) {
        // buffer is full - wait for the host reader to flush+update rpos
        while (*dp.rpos() < *dp.wpos())
            ; // wait for host to catch up to wpos with it's rpos
        *dp.wpos() = 0;
        // TODO(AP): are these writes guaranteed to be ordered?
        *dp.rpos() = 0;
        wpos = 0;
        if (payload_sz >= sizeof(DebugPrintMemLayout::data)-2) {
            // Handle a special case - this value cannot be printed because it cannot fit in the buffer.
            // -2 is for code_sz and sz_sz.
            // Note that the outer if is definitely also true if we got to this inner if.
            // In this case we replace the input value with debug error message.
            // We cannot recursively call operator << from here because it hasn't been defined yet
            // so there's a bit of code duplication here for this special case
            // Another possibility is to wait for the device to flush and print the string piecemeal.
            // As a negative side effect,
            // unfortunately this special case increases the code size generated for each instance of <<.
            uint8_t* printbuf = dp.data();
            payload_sz = DebugPrintStrCopy(
                reinterpret_cast<char*>(printbuf+code_sz+sz_sz),
                debug_print_overflow_error_message);
            printbuf[0] = DEBUG_PRINT_TYPEID_CSTR;
            printbuf[code_sz] = payload_sz;
            wpos = payload_sz + sz_sz + code_sz;
            *dp.wpos() = wpos;
            return dp;
        }
    }

    uint8_t* printbuf = dp.data();
    // no need for a circular buffer since perf is not critical
    printbuf[wpos] = typecode;
    wpos += code_sz;
    printbuf[wpos] = payload_sz;
    wpos += sz_sz;
    const uint8_t* valaddr = DebugPrintTypeAddr<T>(&val);
    for (uint32_t j = 0; j < payload_sz; j++)
        printbuf[wpos+j] = valaddr[j];
    wpos += payload_sz;

    // our message needs to be atomic w.r.t code, size and payload
    // so we only update wpos in the end
    *dp.wpos() = wpos;

    return dp;
}

// explicit instantiations of operator<<
template DebugPrinter operator<< <const char*>(DebugPrinter dp, const char* val);
template DebugPrinter operator<< <ENDL>(DebugPrinter, ENDL val);
template DebugPrinter operator<< <SETW>(DebugPrinter, SETW val);
template DebugPrinter operator<< <uint32_t>(DebugPrinter, uint32_t val);
template DebugPrinter operator<< <float>(DebugPrinter, float val);
template DebugPrinter operator<< <char>(DebugPrinter, char val);
template DebugPrinter operator<< <RAISE>(DebugPrinter, RAISE val);
template DebugPrinter operator<< <WAIT>(DebugPrinter, WAIT val);
template DebugPrinter operator<< <FIXP>(DebugPrinter, FIXP val);
template DebugPrinter operator<< <HEX>(DebugPrinter, HEX val);
template DebugPrinter operator<< <SETP>(DebugPrinter, SETP val);
template DebugPrinter operator<< <F16>(DebugPrinter, F16 val);
template DebugPrinter operator<< <F32>(DebugPrinter, F32 val);
template DebugPrinter operator<< <U32>(DebugPrinter, U32 val);
template DebugPrinter operator<< <TILESAMPLES8>(DebugPrinter, TILESAMPLES8 val);
template DebugPrinter operator<< <TILESAMPLES32>(DebugPrinter, TILESAMPLES32 val);
