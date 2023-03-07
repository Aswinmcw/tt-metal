#pragma once

#include "ll_buda/tensor/tensor.hpp"
#include "ll_buda/host_api.hpp"

namespace tt {

namespace ll_buda {

namespace tensor_impl {

// Copied from root/tensor.hpp
template <class T>
inline std::vector<T> initialize_row_major_tensor_data(const std::array<uint32_t, 4> &shape, Initialize init_type, int rand_max_val = 100, int seed = 0) {
    std::vector<T> values;
    auto rand_float = std::bind(std::uniform_real_distribution<float>(0, rand_max_val), std::mt19937(seed));
    auto get_val = [&init_type, shape, &rand_float](int x, int y, int z, int w) {
        float val;
        switch (init_type) {
            case Initialize::ZEROS:
                val = 0;
                break;
            case Initialize::ONES:
                val = 1;
                break;
            case Initialize::INCREMENT:
                val = x + shape[3] * y + shape[3] * shape[2] * z + shape[3] * shape[2] * shape[1] * w;
                break;
            case Initialize::RANDOM:
                val = rand_float();
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
                    float val = get_val(x, y, z, w);
                    values.push_back(static_cast<T>(val));
                }
            }
        }
    }

    return values;
}

// ======================================================================================
//                          dtype to uint32_t packer and unpackers
// ======================================================================================
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

// ======================================================================================
//                           Data reader, writer, and initializer
// ======================================================================================
std::tuple<int, int, int> get_interleaved_read_write_unit_metadata(DataType dtype, Layout layout, uint32_t total_size_bytes, const std::array<uint32_t, 4>& shape);

void allocate_interleaved_buffer_on_device(Tensor &tensor, uint32_t buffer_size_bytes);

template <typename T>
inline std::vector<T> initialize_data(const std::array<uint32_t, 4> &shape, Initialize init_type, Layout layout) {
    TT_ASSERT(layout == Layout::TILE or layout == Layout::ROW_MAJOR, "Only ROW_MAJOR or TILE layout is supported!");
    std::vector<T> data = initialize_row_major_tensor_data<T>(shape, init_type);
    if (layout == Layout::TILE) {
        data = convert_layout_row_major_to_tile(shape, data);
    }
    return data;
}

template <typename T>
std::vector<T> read_data_from_device(const Tensor &tensor, uint32_t size_in_bytes) {
    std::vector<uint32_t> device_data;
    auto [num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry] = get_interleaved_read_write_unit_metadata(tensor.dtype(), tensor.layout(), size_in_bytes, tensor.shape());
    ReadFromDeviceDRAMChannelsInterleaved(tensor.device(), device_data, tensor.buffer()->address(), num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry);
    auto unpacked_data = unpack_uint32_vec<T>(device_data);
    return unpacked_data;
}

template <typename T>
inline void write_data_to_device(const Tensor &tensor, std::vector<T> data) {
    uint32_t packed_size_in_bytes = packed_buffer_size_bytes<T>(data.size());
    auto [num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry] = get_interleaved_read_write_unit_metadata(tensor.dtype(), tensor.layout(), packed_size_in_bytes, tensor.shape());
    std::vector<uint32_t> uint32_data = pack_vec_into_uint32_vec<T>(data);
    WriteToDeviceDRAMChannelsInterleaved(
        tensor.device(), uint32_data, tensor.buffer()->address(), num_bank_units, num_entries_per_bank_unit, num_bytes_per_entry);
}

template <typename T>
inline void initialize_data_on_device(Tensor &tensor, std::vector<T> data) {
    TT_ASSERT(tensor.device() != nullptr);
    uint32_t packed_size_in_bytes = packed_buffer_size_bytes<T>(data.size());
    allocate_interleaved_buffer_on_device(tensor, packed_size_in_bytes);
    write_data_to_device<T>(tensor, data);
}

template <typename T>
inline void initialize_data_helper(Tensor &tensor, Initialize init_type) {
    auto data = initialize_data<T>(tensor.shape(), init_type, tensor.layout());
    if (tensor.on_host()) {
        auto data_ptr = new std::vector<T>(std::move(data));
        tensor.data_ = static_cast<void *>(data_ptr);
    } else {
        initialize_data_on_device<T>(tensor, data);
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
inline Tensor to_device(const Tensor &tensor, Device *target_device) {
    TT_ASSERT(target_device != nullptr && "Need target device in order to move tensor to device!");
    TT_ASSERT(tensor.data_ptr() != nullptr && "Need data to exist in order to move it to device");
    auto data_vec = *reinterpret_cast<std::vector<T>*>(tensor.data_ptr());
    return Tensor(data_vec, tensor.shape(), tensor.dtype(), tensor.layout(), target_device);
}

// ======================================================================================
//                                         Print
// ======================================================================================
inline std::ostream& operator<<(std::ostream& os, const DataType& dtype) {
    switch (dtype) {
        case DataType::BFLOAT16: os << "bfloat16"; break;
        case DataType::FLOAT32: os << "float32"; break;
        case DataType::UINT32: os << "uint32"; break;
        default: throw std::invalid_argument("Unknown data type");
    }
    return os;
}

template <typename T>
void print_data(const std::vector<T> &data, DataType dtype) {
    std::cout << "[ ";
    for (int i = 0; i < data.size(); i++) {
        std::cout << data[i] << ", ";
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
                    std::cout << data[index];
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
                auto converted_data = convert_layout_row_major_to_tile(tensor.shape(), data_vec);
                pretty_print ? print_row_major_data(converted_data, tensor.shape(), tensor.dtype()) : print_data(converted_data, tensor.dtype());
            } else {
                TT_ASSERT(false && "Unsupported print layout");
            }
        break;
        case Layout::TILE:
            if (print_layout == Layout::ROW_MAJOR) {
                auto converted_data = convert_layout_tile_to_row_major(tensor.shape(), data_vec);
                pretty_print ? print_row_major_data(converted_data, tensor.shape(), tensor.dtype()) : print_data(converted_data, tensor.dtype());
            } else if (print_layout == Layout::TILE) {
                pretty_print ? print_row_major_data(data_vec, tensor.shape(), tensor.dtype()) : print_data(data_vec, tensor.dtype());
            } else {
                TT_ASSERT(false && "Unsupported print layout");
            }
        break;
        case Layout::CHANNELS_LAST:
            TT_ASSERT(false && "Unsupported print layout");
        break;
        default:
            TT_ASSERT(false && "Unsupported print layout");
    }
}

}  // namespace tensor_impl

}  // namespace ll_buda

}  // namespace tt
