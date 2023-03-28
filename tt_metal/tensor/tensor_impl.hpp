#pragma once

#include "tt_metal/tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace tt_metal {

namespace tensor_impl {

// -----------------------------------------------------------------------------------------------------------------------------------------------
// ===============================================================================================================================================
//                                                              Low Level APIs
// ===============================================================================================================================================
// -----------------------------------------------------------------------------------------------------------------------------------------------

// ======================================================================================
//                        Data type converters, packers, and unpackers
// ======================================================================================
template <typename T1, typename T2>
inline std::vector<T1> cast_vec(std::vector<T2> &data_to_convert) {
    std::vector<T1> converted_data;
    for (auto datum : data_to_convert) {
        converted_data.push_back(static_cast<T1>(datum));
    }
    return converted_data;
}

template <>
inline std::vector<float> cast_vec(std::vector<bfloat16> &data_to_convert) {
    std::vector<float> converted_data;
    for (auto datum : data_to_convert) {
        converted_data.push_back(datum.to_float());
    }
    return converted_data;
}

template <>
inline std::vector<uint32_t> cast_vec(std::vector<bfloat16> &data_to_convert) {
    std::vector<uint32_t> converted_data;
    for (auto datum : data_to_convert) {
        converted_data.push_back((uint32_t)datum.to_uint16());
    }
    return converted_data;
}

template <typename T>
constexpr inline std::vector<uint32_t> pack_vec_into_uint32_vec(std::vector<T> &data_to_pack) {
    std::vector<uint32_t> packed_data;
    TT_ASSERT(false && "Don't know how to pack data into uint32 vector generically!");
    return packed_data;
}

template <>
inline std::vector<uint32_t> pack_vec_into_uint32_vec(std::vector<uint32_t> &data_to_pack) {
    return data_to_pack;
}

template <>
inline std::vector<uint32_t> pack_vec_into_uint32_vec(std::vector<bfloat16> &data_to_pack) {
    return pack_bfloat16_vec_into_uint32_vec(data_to_pack);
}

template <>
inline std::vector<uint32_t> pack_vec_into_uint32_vec(std::vector<float> &data_to_pack) {
    std::vector<uint32_t> uint32_data;
    assert(data_to_pack.size() % 2 == 0);
    for (auto i = 0; i < data_to_pack.size(); i += 2) {
        auto float_val1 = data_to_pack[i];
        auto float_val2 = data_to_pack[i + 1];
        auto bfloat_val1 = bfloat16(float_val1);
        auto bfloat_val2 = bfloat16(float_val2);
        auto uint32_val = pack_two_bfloat16_into_uint32({bfloat_val1, bfloat_val2});
        uint32_data.push_back(uint32_val);
    }
    return uint32_data;
}

template <typename T>
constexpr inline std::vector<T> unpack_uint32_vec(std::vector<uint32_t> &data_to_unpack) {
    std::vector<uint32_t> unpacked_data;
    TT_ASSERT(false && "Don't know how to unpack uint32 data generically!");
    return unpacked_data;
}

template <>
inline std::vector<uint32_t> unpack_uint32_vec(std::vector<uint32_t> &data_to_unpack) {
    return data_to_unpack;
}

template <>
inline std::vector<bfloat16> unpack_uint32_vec(std::vector<uint32_t> &data_to_unpack) {
    return unpack_uint32_vec_into_bfloat16_vec(data_to_unpack);
}

template <>
inline std::vector<float> unpack_uint32_vec(std::vector<uint32_t> &data_to_unpack) {
    std::vector<float> float_data;
    for (auto i = 0; i < data_to_unpack.size(); i++) {
        auto unpacked = unpack_two_bfloat16_from_uint32(data_to_unpack[i]);
        auto float_val1 = unpacked.first.to_float();
        auto float_val2 = unpacked.second.to_float();
        float_data.push_back(float_val1);
        float_data.push_back(float_val2);
    }
    return float_data;
}

template <typename T>
constexpr inline uint32_t element_size_bytes() {
    return sizeof(T);
}

template <typename T>
constexpr inline uint32_t packed_buffer_size_bytes(uint32_t volume_unpacked_data) {
    auto num_type_in_u32 = sizeof(uint32_t) / sizeof(T);
    return (volume_unpacked_data/num_type_in_u32) * sizeof(uint32_t);
}

// Specialization for float because it gets converted to bfloat16 before being packed
template <>
constexpr inline uint32_t packed_buffer_size_bytes<float>(uint32_t volume_unpacked_data) {
    auto num_type_in_u32 = sizeof(uint32_t) / sizeof(bfloat16);
    return (volume_unpacked_data/num_type_in_u32) * sizeof(uint32_t);
}


// ======================================================================================
//                                  Layout converters
// ======================================================================================
template <typename T>
inline std::vector<T> convert_layout_row_major_to_tile(const std::array<uint32_t, 4> &shape, const std::vector<T>& data_to_convert) {
    std::vector<uint32_t> shape_vec = {shape[0], shape[1], shape[2], shape[3]};
    return convert_layout(data_to_convert, shape_vec, TensorLayout::LIN_ROW_MAJOR, TensorLayout::TILED32_4FACES);
}

template <typename T>
inline std::vector<T> convert_layout_tile_to_row_major(const std::array<uint32_t, 4> &shape, const std::vector<T>& data_to_convert) {
    std::vector<uint32_t> shape_vec = {shape[0], shape[1], shape[2], shape[3]};
    return convert_layout(data_to_convert, shape_vec, TensorLayout::TILED32_4FACES, TensorLayout::LIN_ROW_MAJOR);
}

template <typename T>
inline std::vector<T> convert_layout_row_major_to_channels_last(const std::array<uint32_t, 4> &shape, const std::vector<T>& data_to_convert) {
    std::vector<uint32_t> shape_vec = {shape[0], shape[1], shape[2], shape[3]};
    return convert_layout(data_to_convert, shape_vec, TensorLayout::LIN_ROW_MAJOR, TensorLayout::CHANNELS_LAST);
}

template <typename T>
inline std::vector<T> convert_layout_channels_last_to_row_major(const std::array<uint32_t, 4> &shape, const std::vector<T>& data_to_convert) {
    std::vector<uint32_t> shape_vec = {shape[0], shape[1], shape[2], shape[3]};
    return convert_layout(data_to_convert, shape_vec, TensorLayout::CHANNELS_LAST, TensorLayout::LIN_ROW_MAJOR);
}

// ======================================================================================
//                                         Print
// ======================================================================================
std::ostream& operator<<(std::ostream& os, const DataType& dtype);

template <typename T>
inline void print_datum(T datum) {
    std::cout << datum;
}

template <>
inline void print_datum(bfloat16 datum) {
    std::cout << datum.to_float();
}

template <typename T>
void print_data(const std::vector<T> &data, DataType dtype) {
    std::cout << "[ ";
    for (int i = 0; i < data.size(); i++) {
        print_datum(data[i]);
        if (i < data.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << " dtype=" <<  dtype << " ]" << std::endl;
}

template <typename T>
void print_row_major_data(const std::vector<T> &data, std::array<uint32_t, 4> shape, DataType dtype) {
    std::cout << "[ ";
    for(auto w = 0; w < shape[0]; w++) {
        if(w == 0)
            std::cout << "[";
        else
            std::cout << "  [";
        for(auto z = 0; z < shape[1]; z++) {
            if (z == 0)
                std::cout << "[";
            else
                std::cout << "   [";
            for(auto y = 0; y < shape[2]; y++) {
                if (y == 0)
                    std::cout << "[";
                else
                    std::cout << "    [";
                for(auto x = 0; x < shape[3]; x++) {
                    // data in row major order
                    auto index = x + y*shape[3] + z*shape[2]*shape[3] + w*shape[1]*shape[2]*shape[3];
                    print_datum(data[index]);
                    if (x < shape[3] - 1) {
                        std::cout << ", ";
                    }
                }
                if(y < shape[2] - 1)
                    std::cout << "]," << std::endl;
                else
                    std::cout << "]";
            }
            if(z < shape[1] - 1)
                std::cout << "]," << std::endl << std::endl;
            else
                std::cout << "]";
        }
        if(w < shape[0] - 1)
            std::cout << "]," << std::endl << std::endl << std::endl;
        else
            std::cout << "]";
    }
    std::cout << " dtype=" <<  dtype << " ]" << std::endl;
}


// ======================================================================================
//                                      Validators
// ======================================================================================
void validate_on_device_dtype_and_layout(Device *device, DataType dtype, Layout layout);

// ======================================================================================
//                           Data reader, writer, and initializers
// ======================================================================================
template <class T>
inline std::vector<T> initialize_row_major_tensor_data(const std::array<uint32_t, 4> &shape, Initialize init_type, int rand_max_val = 100, int seed = 0) {
    std::vector<T> values;

    auto rand_float = std::bind(std::uniform_real_distribution<float>(0, rand_max_val), std::mt19937(seed));

    auto get_val = [&init_type, &shape, &rand_float](int x, int y, int z, int w) {
        T val;
        switch (init_type) {
            case Initialize::ZEROS:
                val = static_cast<T>(0);
            break;
            case Initialize::ONES:
                val = static_cast<T>(1);
            break;
            case Initialize::INCREMENT: {
                float float_val = x + shape[3] * y + shape[3] * shape[2] * z + shape[3] * shape[2] * shape[1] * w;
                val = static_cast<T>(float_val);
            }
            break;
            case Initialize::RANDOM: {
                float float_val = rand_float();
                val = static_cast<T>(float_val);
            }
            break;
            default:
                TT_ASSERT(false && "Unsupported initializer type");
            break;
        }
        return val;
    };

    for(auto w = 0; w < shape[0]; w++) {
        for(auto z = 0; z < shape[1]; z++) {
            for(auto y = 0; y < shape[2]; y++) {
                for(auto x = 0; x < shape[3]; x++) {
                    T val = get_val(x, y, z, w);
                    values.push_back(val);
                }
            }
        }
    }
    return values;
}

std::tuple<int, int, int> get_interleaved_read_write_unit_metadata(DataType dtype, Layout layout, uint32_t total_size_bytes, const std::array<uint32_t, 4>& shape);

void allocate_dram_buffer_on_device(Tensor &tensor, uint32_t buffer_size_bytes);

void read_contiguous_data_from_device(const Tensor &tensor, uint32_t size_in_bytes, std::vector<uint32_t> &host_buffer);

void write_contiguous_data_to_device(const Tensor &tensor, std::vector<uint32_t> &data);

// -----------------------------------------------------------------------------------------------------------------------------------------------
// ===============================================================================================================================================
//                                                              High Level APIs
// ===============================================================================================================================================
// -----------------------------------------------------------------------------------------------------------------------------------------------

// ======================================================================================
//                           Data reader, writer, and initializers
// ======================================================================================
void allocate_interleaved_buffer_on_device(Tensor &tensor, uint32_t buffer_size_bytes);

void allocate_buffer_on_device(Tensor &tensor, uint32_t buffer_size_bytes);

template <typename T>
inline std::vector<T> initialize_data(const std::array<uint32_t, 4> &shape, Initialize init_type, Layout layout) {
    TT_ASSERT(layout == Layout::TILE or layout == Layout::ROW_MAJOR or layout == Layout::CHANNELS_LAST);
    std::vector<T> data = initialize_row_major_tensor_data<T>(shape, init_type);
    if (layout == Layout::TILE) {
        data = convert_layout_row_major_to_tile(shape, data);
    }
    else if (layout == Layout::CHANNELS_LAST) {
        data = convert_layout_row_major_to_channels_last(shape, data);
    }
    return data;
}

void read_interleaved_data_from_device(const Tensor &tensor, uint32_t size_in_bytes, std::vector<uint32_t> &host_buffer);

template <typename T>
std::vector<T> read_data_from_device(const Tensor &tensor, uint32_t size_in_bytes) {
    std::vector<uint32_t> device_data;
    if (tensor.interleaved()) {
        read_interleaved_data_from_device(tensor, size_in_bytes, device_data);
    } else {
        read_contiguous_data_from_device(tensor, size_in_bytes, device_data);
    }
    auto unpacked_data = unpack_uint32_vec<T>(device_data);
    return unpacked_data;
}

void write_interleaved_data_to_device(const Tensor &tensor, std::vector<uint32_t> &data);

template <typename T>
inline void write_data_to_device(const Tensor &tensor, std::vector<T> &data) {
    std::vector<uint32_t> uint32_data = pack_vec_into_uint32_vec<T>(data);
    if (tensor.interleaved()) {
        write_interleaved_data_to_device(tensor, uint32_data);
    } else {
        write_contiguous_data_to_device(tensor, uint32_data);
    }
}

template <typename T>
inline void initialize_data_on_device(Tensor &tensor, std::vector<T> &data) {
    TT_ASSERT(tensor.device() != nullptr);
    uint32_t packed_size_in_bytes = packed_buffer_size_bytes<T>(data.size());
    allocate_buffer_on_device(tensor, packed_size_in_bytes);
    write_data_to_device<T>(tensor, data);
}

template <typename T>
inline void write_data(Tensor &tensor, std::vector<T> &data) {
    if (tensor.on_host()) {
        auto data_ptr = new std::vector<T>(std::move(data));
        tensor.data_ = static_cast<void *>(data_ptr);
    } else {
        initialize_data_on_device<T>(tensor, data);
    }
}

template <typename T1, typename T2>
inline void convert_and_write_data(Tensor &tensor, std::vector<T2> &data) {
    if (std::is_same<T1, T2>::value) {
        write_data(tensor, data);
    } else {
        std::vector<T1> converted_data = cast_vec<T1>(data);
        write_data(tensor, converted_data);
    }
}

template <typename T>
inline void initialize_data_helper(Tensor &tensor, Initialize init_type) {
    auto data = initialize_data<T>(tensor.shape(), init_type, tensor.layout());
    write_data(tensor, data);
}

template <typename T1, typename T2>
inline void convert_layout_or_type_and_write_data(Tensor &tensor, std::vector<T2> &data) {
    auto layout = tensor.layout();
    auto shape = tensor.shape();
    if (layout == Layout::TILE) {
        std::cout << "Converting from row major to tile " << std::endl;
        data = convert_layout_row_major_to_tile(shape, data);
    }
    else if (layout == Layout::CHANNELS_LAST) {
        std::cout << "Converting from tile to channels last " << std::endl;
        data = convert_layout_row_major_to_channels_last(shape, data);
    }
    if (std::is_same<T1, T2>::value) {
        write_data(tensor, data);
    } else {
        std::vector<T1> converted_data = cast_vec<T1>(data);
        write_data(tensor, converted_data);
    }
}

// ======================================================================================
//                                         .to()
// ======================================================================================
template <typename T>
inline Tensor to_host(const Tensor &tensor) {
    TT_ASSERT(tensor.buffer() != nullptr, "Need DRAM buffers on device to exist to copy data to host!");
    TT_ASSERT(tensor.device() != nullptr && "Need device to be set copy data from device to host!");
    uint32_t size_in_bytes = tensor.buffer()->size();
    auto data_vec = read_data_from_device<T>(tensor, size_in_bytes);
    return Tensor(data_vec, tensor.shape(), tensor.dtype(), tensor.layout());
}

template <typename T>
inline Tensor to_device(const Tensor &tensor, Device *target_device, const MemoryConfig &mem_config) {
    TT_ASSERT(target_device != nullptr && "Need target device in order to move tensor to device!");
    TT_ASSERT(tensor.data_ptr() != nullptr && "Need data to exist in order to move it to device");
    auto data_vec = *reinterpret_cast<std::vector<T>*>(tensor.data_ptr());
    return Tensor(data_vec, tensor.shape(), tensor.dtype(), tensor.layout(), target_device, mem_config);
}

template <typename T>
inline Tensor to_layout(const Tensor &tensor, Layout target_layout) {
    TT_ASSERT(tensor.layout() != target_layout && "Cannot convert to target layout same as it is the same the current layout.");
    auto data = *reinterpret_cast<std::vector<T>*>(tensor.data_ptr());
    switch (tensor.layout()) {
        case Layout::ROW_MAJOR:
            if (target_layout == Layout::TILE) {
                data = convert_layout_row_major_to_tile(tensor.shape(), data);
            }
            else if (target_layout == Layout::CHANNELS_LAST) {
                data = convert_layout_row_major_to_channels_last(tensor.shape(), data);
            }
            else {
                TT_ASSERT(false && "Unsupported layout conversion");
            }
        break;
        case Layout::TILE:
            if (target_layout == Layout::ROW_MAJOR) {
                data = convert_layout_tile_to_row_major(tensor.shape(), data);
            }
            else {
                TT_ASSERT(false && "Unsupported layout conversion");
            }
        break;
        case Layout::CHANNELS_LAST:
            if (target_layout == Layout::ROW_MAJOR) {
                data = convert_layout_channels_last_to_row_major(tensor.shape(), data);
            }
            else {
                TT_ASSERT(false && "Unsupported layout conversion");
            }
        break;
        default:
            TT_ASSERT(false && "Unsupported layout conversion");
    }
    return Tensor(data, tensor.shape(), tensor.dtype(), target_layout);
}

// ======================================================================================
//                                         Print
// ======================================================================================
template <typename T>
inline void print(const Tensor &tensor, Layout print_layout, bool pretty_print) {
    auto data_ptr_to_print = tensor.data_ptr();
    if (not tensor.on_host()) {
        auto temp_tensor = to_host<T>(tensor);
        data_ptr_to_print = temp_tensor.data_ptr();
    }
    auto data_vec = *reinterpret_cast<std::vector<T>*>(data_ptr_to_print);

    switch (tensor.layout()) {
        case Layout::ROW_MAJOR:
            if (print_layout == Layout::ROW_MAJOR) {
                pretty_print ? print_row_major_data(data_vec, tensor.shape(), tensor.dtype()) : print_data(data_vec, tensor.dtype());
            } else if (print_layout == Layout::TILE) {
                TT_ASSERT(pretty_print == false && "Can only pretty print in Row Major layout!");
                auto converted_data = convert_layout_row_major_to_tile(tensor.shape(), data_vec);
                print_data(converted_data, tensor.dtype());
            }
            else {
                TT_ASSERT(false && "Unsupported print layout");
            }
        break;
        case Layout::TILE:
            if (print_layout == Layout::ROW_MAJOR) {
                auto converted_data = convert_layout_tile_to_row_major(tensor.shape(), data_vec);
                pretty_print ? print_row_major_data(converted_data, tensor.shape(), tensor.dtype()) : print_data(converted_data, tensor.dtype());
            } else if (print_layout == Layout::TILE) {
                TT_ASSERT(pretty_print == false && "Can only pretty print in Row Major layout!");
                print_data(data_vec, tensor.dtype());
            } else {
                TT_ASSERT(false && "Unsupported print layout");
            }
        break;
        case Layout::CHANNELS_LAST:
            if (print_layout == Layout::ROW_MAJOR) {
                auto converted_data = convert_layout_channels_last_to_row_major(tensor.shape(), data_vec);
                pretty_print ? print_row_major_data(converted_data, tensor.shape(), tensor.dtype()) : print_data(converted_data, tensor.dtype());
            }
            else if (print_layout == Layout::CHANNELS_LAST) {
                auto cl_shape = tensor.shape();
                cl_shape[3] = tensor.shape()[1];
                cl_shape[2] = tensor.shape()[3];
                cl_shape[1] = tensor.shape()[2];
                pretty_print ? print_row_major_data(data_vec, cl_shape, tensor.dtype()) : print_data(data_vec, tensor.dtype());
            }
            else {
                TT_ASSERT(false && "Unsupported print layout");
            }
        break;
        default:
            TT_ASSERT(false && "Unsupported print layout");
    }
}

}  // namespace tensor_impl

}  // namespace tt_metal

}  // namespace tt
