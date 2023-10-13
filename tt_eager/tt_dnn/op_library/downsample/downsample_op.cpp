// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>


#include "tt_dnn/op_library/downsample/downsample_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {


void Downsample::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_ASSERT(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to downsample need to be on device!");
    TT_ASSERT(input_tensor_a.buffer() != nullptr , "Operands to downsample need to be allocated in buffers on device!");
    TT_ASSERT(input_tensor_a.dtype() == DataType::BFLOAT16, "Only bloat16 dataformat supported");
    TT_ASSERT(input_tensor_a.layout() == Layout::TILE, "Can only downsample tile major data");

    TT_ASSERT(input_tensor_a.volume() % TILE_HW == 0);
    TT_ASSERT(input_tensor_a.memory_config().is_sharded() && this->output_mem_config.is_sharded());
}

std::vector<Shape> Downsample::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_ASSERT(input_tensor_a.shape()[0] == 1 && input_tensor_a.shape()[1] == 1);
    uint32_t input_height = input_tensor_a.shape()[2];
    auto [input_height_size_z, input_height_size_y, input_height_size_x, height_y_stride, height_x_stride] = this->downsample_params;
    TT_ASSERT(input_height == input_height_size_z * input_height_size_y * input_height_size_x);
    uint32_t output_height = (input_height_size_z * ceil(input_height_size_y / height_y_stride) * ceil(input_height_size_x / height_x_stride));
    uint32_t output_width = input_tensor_a.shape()[3];
    return {Shape({1, 1, output_height, output_width})};
}

std::vector<Tensor> Downsample::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto input_shard_spec = input_tensor.shard_spec().value();
    auto input_shard_height = input_shard_spec.shard_shape[0];
    TT_ASSERT(input_shard_height % (this->downsample_params[3] * this->downsample_params[4]) == 0);
    uint32_t output_shard_height = input_shard_height / (this->downsample_params[3] * this->downsample_params[4]);
    auto output_shard_width = input_shard_spec.shard_shape[1];
    auto output_shard_grid = input_shard_spec.shard_grid;
    return {create_sharded_device_tensor(this->compute_output_shapes(input_tensors).at(0), input_tensor.dtype(), Layout::TILE, input_tensor.device(), this->output_mem_config, ShardSpec{output_shard_grid, std::array<uint32_t, 2>{{output_shard_height, output_shard_width}}})};
}

operation::ProgramWithCallbacks Downsample::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return {downsample_single_core(input_tensor_a, downsample_params, output_tensor)};
}

tt::stl::reflection::Attributes Downsample::attributes() const {
    return {
        {"output_mem_config", this->output_mem_config},
        {"downsample_params", this->downsample_params}
    };
}

Tensor downsample(const Tensor &input_tensor_a, std::array<uint32_t, 5> downsample_params, const MemoryConfig& mem_config) {
    return operation::run_without_autoformat(Downsample{mem_config, downsample_params}, {input_tensor_a}).at(0);
}

struct DownsampleReadPatternParams {
    uint32_t top_partial_middle_aligned_row_width;
    uint32_t skip_top_partial_middle_aligned_row;
    uint32_t top_partial_right_aligned_row_width;
    uint32_t skip_top_partial_right_aligned_row;
    uint32_t num_rows_top_partial_image;
    uint32_t num_skip_rows_top_partial_image;
    uint32_t num_full_images;
    uint32_t num_rows_bottom_partial_image;
    uint32_t num_skip_rows_bottom_partial_image;
    uint32_t bottom_partial_left_aligned_row_width;
    uint32_t skip_bottom_partial_left_aligned_row;
};

struct ImgTrackingVars {
    uint32_t img_h = 0;
    uint32_t img_w = 0;
    uint32_t next_img_h = 0; // img_h after stride
    uint32_t next_img_w = 0;
    uint32_t input_flat_h = 0; // index within sharded input
    uint32_t output_flat_h = 0; // index within sharded output
};

DownsampleReadPatternParams generate_downsample_read_pattern(ImgTrackingVars & v, uint32_t img_height, uint32_t img_width, uint32_t img_stride_h, uint32_t img_stride_w, uint32_t input_shard_height, uint32_t output_shard_height) {
    cout << "img_h=" << v.img_h << ", img_w=" << v.img_w << ", next_img_h=" << v.next_img_h << ", next_img_w=" << v.img_w << endl;
    // Sanity checks at the start for local data
    TT_ASSERT(v.next_img_h >= v.img_h);
    TT_ASSERT(v.next_img_w == v.img_w); // assumption that the start is picked and not skipped by stride
    TT_ASSERT(v.img_h < img_height);
    TT_ASSERT(v.next_img_w < img_width);

    bool current_region_is_halo_from_prev_core = false;
    if (v.input_flat_h != 0) {
        current_region_is_halo_from_prev_core = true;
        cout << "GENERATING READ PATTERN FOR HALO REGION FROM PREVIOUS CORE" << endl;
        TT_ASSERT(v.output_flat_h == 0);
    } else {
        cout << "GENERATING READ FOR LOCAL REGION" << endl;
    }

    // constant input and output shard per core
    uint32_t input_end_flat_h = input_shard_height - 1;
    uint32_t output_end_flat_h = output_shard_height - 1;
    TT_ASSERT(v.input_flat_h < input_end_flat_h);
    TT_ASSERT(v.output_flat_h < output_end_flat_h);

    uint32_t output_img_height = std::ceil ( (double) img_height / (double) img_stride_h);
    uint32_t output_img_width = std::ceil ( (double) img_width / (double) img_stride_w);
    bool found_halo_for_next_core = false;

    uint32_t top_partial_middle_aligned_row_width = 0;
    uint32_t skip_top_partial_middle_aligned_row = 1;
    uint32_t top_partial_right_aligned_row_width = 0;
    uint32_t skip_top_partial_right_aligned_row = 1;
    uint32_t num_rows_top_partial_image = 0;
    uint32_t num_skip_rows_top_partial_image = 0;
    uint32_t num_full_images = 0;
    uint32_t num_rows_bottom_partial_image = 0;
    uint32_t num_skip_rows_bottom_partial_image = 0;
    uint32_t bottom_partial_left_aligned_row_width = 0;
    uint32_t skip_bottom_partial_left_aligned_row = 1;
    cout << "input_flat_h=" << v.input_flat_h << endl;
    if (v.img_w != 0) {
        // Check if its right aligned or middle aligned (special corner case for halo)
        if (v.input_flat_h + img_width - v.img_w <= input_end_flat_h+1) {
            // top partial right aligned
            top_partial_right_aligned_row_width = img_width - v.img_w;
            skip_top_partial_right_aligned_row = (v.next_img_h == v.img_h) ? 0 : 1;
            v.input_flat_h += top_partial_right_aligned_row_width;
            if (!skip_top_partial_right_aligned_row) {
                v.output_flat_h += std::ceil((double) top_partial_right_aligned_row_width / (double) img_stride_w);
                TT_ASSERT(v.output_flat_h < output_shard_height);
            }
            v.img_w = 0;
            v.next_img_w = 0;
            if (v.img_h == img_height - 1) {
                v.img_h = 0;
                v.next_img_h = 0;
            } else {
                v.img_h += 1;
                if (v.next_img_h < v.img_h) {
                    v.next_img_h += img_stride_h;
                }
            }
        } else {
            // special corner case for halo region
            // middle aligned
            TT_ASSERT(input_end_flat_h - v.input_flat_h + 1 < img_width);
            TT_ASSERT(current_region_is_halo_from_prev_core);
            // top partial middle aligned
            top_partial_middle_aligned_row_width = input_end_flat_h - v.input_flat_h + 1;
            skip_top_partial_middle_aligned_row = (v.next_img_h == v.img_h) ? 0 : 1;
            v.input_flat_h += top_partial_middle_aligned_row_width;
            if (!skip_top_partial_middle_aligned_row) {
                v.output_flat_h += std::ceil((double) top_partial_middle_aligned_row_width / (double) img_stride_w);
                TT_ASSERT(v.output_flat_h < output_shard_height);
            }
            while (v.img_w < top_partial_middle_aligned_row_width) {
                v.img_w += 1;
                if (v.next_img_w < v.img_w) {
                    v.next_img_w += img_stride_w;
                }
            }
            TT_ASSERT(v.img_w < img_width-1);
            TT_ASSERT(v.next_img_w >= v.img_w);
        }
    }

    TT_ASSERT(v.output_flat_h <= output_end_flat_h);
    TT_ASSERT(v.next_img_h >= v.img_h);
    if (v.img_w != 0) {
        // special case for halo
        TT_ASSERT(v.input_flat_h == input_end_flat_h+1);
    }
    TT_ASSERT(v.img_h < img_height && v.img_w < img_width);

    uint32_t num_rows_remaining_of_current_image = (v.img_h == 0) ? 0 : img_height - v.img_h;
    if (num_rows_remaining_of_current_image > 0) {
        uint32_t num_rows_to_skip = v.next_img_h - v.img_h;
        uint32_t output_h_from_remaining_rows_of_current_image = std::ceil( (double) (num_rows_remaining_of_current_image - num_rows_to_skip) / (double) img_stride_h ) * output_img_width;
        bool output_for_partial_top_image = v.output_flat_h + output_h_from_remaining_rows_of_current_image <= output_end_flat_h+1;
        bool input_for_partial_top_image = v.input_flat_h + (num_rows_remaining_of_current_image * img_width) <= input_end_flat_h+1;
        if (output_for_partial_top_image && input_for_partial_top_image) {
            // Top partial image section
            num_rows_top_partial_image = img_height - v.img_h;
            num_skip_rows_top_partial_image = v.next_img_h - v.img_h;
            // Sanity check
            TT_ASSERT((v.img_h + num_rows_top_partial_image == img_height));
            v.img_h = 0;
            v.next_img_h = 0;
            v.input_flat_h += (num_rows_top_partial_image * img_width);
            v.output_flat_h += output_h_from_remaining_rows_of_current_image;
            TT_ASSERT(v.input_flat_h <= input_end_flat_h+1);
        }
    TT_ASSERT(v.output_flat_h <= output_end_flat_h+1);
    }
    TT_ASSERT(v.img_h < img_height && v.img_w < img_width);

    if (v.img_h == 0 && v.img_w == 0) {
        // Check for full images
        while(1) {
            bool output_for_current_full_image = v.output_flat_h + (output_img_height * output_img_width) <= output_end_flat_h+1;
            bool input_for_current_full_image = v.input_flat_h + (img_height * img_width) <= input_end_flat_h+1;
            if (!output_for_current_full_image || !input_for_current_full_image) {
                break;
            }
            v.input_flat_h += (img_height * img_width);
            v.img_h = 0;
            v.img_w = 0;
            v.next_img_h = 0;
            v.next_img_w = 0;
            num_full_images += 1;
            v.output_flat_h +=  (output_img_height * output_img_width);
        }
        TT_ASSERT(v.img_h == 0 && v.img_w == 0 && v.next_img_h == 0 && v.next_img_w == 0);
    }

    // Sanity check
    TT_ASSERT(v.input_flat_h <= input_end_flat_h+1);
    TT_ASSERT(v.output_flat_h <= output_end_flat_h+1);

    bool found_first_unskipped_row_in_bottom_partial_imgage = false;
    // check for bottom partial image rows
    while (1) {
        bool output_for_bottom_partial_image_row = (v.next_img_h == v.img_h) ? (v.output_flat_h + output_img_width <= output_end_flat_h+1) : true; // true for skipped row
        bool input_for_bottom_partial_image_row = v.input_flat_h + img_width <= input_end_flat_h+1;
        if (!output_for_bottom_partial_image_row || !input_for_bottom_partial_image_row) {
            break;
        }
        if (!found_first_unskipped_row_in_bottom_partial_imgage) {
            if (v.next_img_h == v.img_h) {
                found_first_unskipped_row_in_bottom_partial_imgage = true;
            } else {
                TT_ASSERT(v.next_img_h > v.img_h);
                num_skip_rows_bottom_partial_image += 1;
            }
        }
        v.input_flat_h += img_width;
        if (v.next_img_h == v.img_h) {
            v.output_flat_h += output_img_width;
        }
        v.img_w = 0;
        v.next_img_w = 0;
        TT_ASSERT(v.img_h < img_height - 1); // this is supposed to be a bottom partial image
        v.img_h += 1;
        if (v.next_img_h < v.img_h) {
            v.next_img_h += img_stride_h;
            TT_ASSERT(v.next_img_h < img_height); // odd heights and odd size sharding with stride > 1 not supported
        }
        num_rows_bottom_partial_image += 1;
    }

    // Sanity check
    TT_ASSERT(v.input_flat_h <= input_end_flat_h+1);
    TT_ASSERT(v.output_flat_h <= output_end_flat_h+1);
    TT_ASSERT(v.img_h < img_height && v.img_w < img_width);

    // check if there is a bottom partial left aligned row
    if (v.input_flat_h < input_end_flat_h && v.output_flat_h < output_end_flat_h) {
        TT_ASSERT(v.img_w == 0 && v.next_img_w == 0);
        // bottom partial left aligned row width can be split between 2 cores
        uint32_t input_remaining = input_end_flat_h - v.input_flat_h + 1;
        uint32_t output_remaining = output_end_flat_h - v.output_flat_h + 1;
        TT_ASSERT(output_remaining < output_img_width || input_remaining < img_width);  // there must be a partial width either on input side or output side
        bottom_partial_left_aligned_row_width = input_remaining;
        if (output_remaining < output_img_width) {
            bottom_partial_left_aligned_row_width = std::min(input_remaining, output_remaining * img_stride_w);
        }
        // sanity
        TT_ASSERT(bottom_partial_left_aligned_row_width < img_width);
        TT_ASSERT(v.next_img_h >= v.img_h);
        skip_bottom_partial_left_aligned_row = (v.next_img_h == v.img_h) ? 0 : 1;
        while(v.img_w < bottom_partial_left_aligned_row_width) {
            v.img_w += 1;
            if (v.next_img_w < v.img_w) {
                v.next_img_w += img_stride_w;
                TT_ASSERT(v.next_img_w < img_width); // odd widths and odd size sharding with stride > 1 not supported
            }
        }
        TT_ASSERT(v.img_w == bottom_partial_left_aligned_row_width && v.next_img_w >= v.img_w);
        v.input_flat_h += bottom_partial_left_aligned_row_width;
        if (!skip_bottom_partial_left_aligned_row) {
            v.output_flat_h += std::ceil( (double) bottom_partial_left_aligned_row_width / (double) img_stride_w);
        }
    }
    TT_ASSERT(v.img_h < img_height && v.img_w < img_width);

    cout << "   top_partial_middle_aligned_row_width=" << top_partial_middle_aligned_row_width << endl;
    cout << "   skip_top_partial_middle_aligned_row=" << skip_top_partial_middle_aligned_row << endl;
    cout << "   top_partial_right_aligned_row_width=" << top_partial_right_aligned_row_width << endl;
    cout << "   skip_top_partial_right_aligned_row=" << skip_top_partial_right_aligned_row << endl;
    cout << "   num_rows_top_partial_image=" << num_rows_top_partial_image << endl;
    cout << "   num_skip_rows_top_partial_image=" << num_skip_rows_top_partial_image << endl;
    cout << "   num_full_images=" << num_full_images << endl;
    cout << "   num_rows_bottom_partial_image=" << num_rows_bottom_partial_image << endl;
    cout << "   num_skip_rows_bottom_partial_image=" << num_skip_rows_bottom_partial_image << endl;
    cout << "   bottom_partial_left_aligned_row_width=" << bottom_partial_left_aligned_row_width << endl;
    cout << "   skip_bottom_partial_left_aligned_row=" << skip_bottom_partial_left_aligned_row << endl;
    //cout << "   output_flat_h=" << v.output_flat_h << endl;
    cout << "   v.output_flat_h=" << v.output_flat_h << endl;

    // Sanity check
    TT_ASSERT(v.input_flat_h <= input_end_flat_h+1);
    TT_ASSERT(v.output_flat_h <= output_end_flat_h+1);

    if (v.output_flat_h < output_end_flat_h+1) {
        TT_ASSERT(current_region_is_halo_from_prev_core);
        TT_ASSERT(v.input_flat_h == input_end_flat_h+1);
    }

    if (v.input_flat_h < input_end_flat_h+1) {
        TT_ASSERT(!current_region_is_halo_from_prev_core);
    }

    if (v.input_flat_h == input_end_flat_h + 1) {
        v.input_flat_h = 0;
    }
    if (v.output_flat_h == output_end_flat_h + 1) {
        v.output_flat_h = 0;
    }
    return DownsampleReadPatternParams{.top_partial_middle_aligned_row_width=top_partial_middle_aligned_row_width,
                                    .skip_top_partial_middle_aligned_row=skip_top_partial_middle_aligned_row,
                                    .top_partial_right_aligned_row_width=top_partial_right_aligned_row_width,
                                    .skip_top_partial_right_aligned_row=skip_top_partial_right_aligned_row,
                                    .num_rows_top_partial_image=num_rows_top_partial_image,
                                    .num_skip_rows_top_partial_image=num_skip_rows_top_partial_image,
                                    .num_full_images=num_full_images,
                                    .num_rows_bottom_partial_image=num_rows_bottom_partial_image,
                                    .num_skip_rows_bottom_partial_image=num_skip_rows_bottom_partial_image,
                                    .bottom_partial_left_aligned_row_width=bottom_partial_left_aligned_row_width,
                                    .skip_bottom_partial_left_aligned_row=skip_bottom_partial_left_aligned_row};
}

operation::ProgramWithCallbacks downsample_single_core(const Tensor &a, std::array<uint32_t, 5> downsample_params, Tensor& output) {

    tt_metal::Program program = tt_metal::Program();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);
    auto [input_height_size_z, input_height_size_y, input_height_size_x, height_y_stride, height_x_stride] = downsample_params;
    tt_metal::Buffer *src0_buffer = a.buffer();

    TT_ASSERT(a.shape()[0] == 1 && a.shape()[1] == 1);
    TT_ASSERT(output.shape()[0] == 1 && output.shape()[1] == 1);

    tt_metal::Device *device = a.device();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");
    // Sanity check of output size
    TT_ASSERT(output.volume() % TILE_HW == 0);
    TT_ASSERT(ceil(a.volume() / (height_y_stride * height_x_stride)) == output.volume());


    uint32_t ncores_x_full_grid = device->compute_with_storage_grid_size().x;
    uint32_t ncores_y_full_grid = device->compute_with_storage_grid_size().y;
    auto all_cores = a.shard_spec().value().shard_grid;
    TT_ASSERT(all_cores == output.shard_spec().value().shard_grid);
    uint32_t num_cores = 0;
    for (const auto& core_range : all_cores.ranges()) {
        num_cores += core_range.size();
    }
    uint32_t ncores = num_cores;
    auto core_range = all_cores;

    uint32_t input_height = a.shape()[2]; // input height == flattened face of input image, multiple images are stacked in H dim
    uint32_t input_width = a.shape()[3]; // input width == input image # of channels
    uint32_t output_height = output.shape()[2]; // output height == flattened face of output image, multiple images are stacked in H dim
    uint32_t output_width = output.shape()[3];
    TT_ASSERT(input_width == output_width);

    uint32_t input_shard_height = a.shard_spec().value().shard_shape[0];
    TT_ASSERT(input_shard_height * num_cores == input_height);
    uint32_t input_shard_width = a.shard_spec().value().shard_shape[1];
    TT_ASSERT(input_shard_width == input_width); // tensor is sharded across height dim only

    uint32_t output_shard_height = output.shard_spec().value().shard_shape[0];
    uint32_t output_shard_width = output.shard_spec().value().shard_shape[1];
    TT_ASSERT(output_shard_width == output_width);

    uint32_t input_width_bytes = input_width * a.element_size();

    TT_ASSERT(input_width % TILE_WIDTH == 0);
    uint32_t num_input_tiles_in_row = input_width / TILE_WIDTH;
    TT_ASSERT(input_shard_height % TILE_HEIGHT == 0);
    uint32_t num_rows_of_input_tiles = input_shard_height / TILE_HEIGHT;

    TT_ASSERT(output_width % TILE_WIDTH == 0);
    uint32_t num_output_tiles_in_row = output_width / TILE_WIDTH;
    TT_ASSERT(output_shard_height % TILE_HEIGHT == 0);
    uint32_t num_rows_of_output_tiles = output_shard_height / TILE_HEIGHT;

    uint32_t input_cb_index = CB::c_in0;
    uint32_t num_input_tiles = num_input_tiles_in_row * num_rows_of_input_tiles;
    tt_metal::CircularBufferConfig input_cb_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{input_cb_index, cb_data_format}})
		.set_page_size(input_cb_index, single_tile_size);
    input_cb_config = input_cb_config.set_globally_allocated_address(a.buffer()->address());
    auto input_cb = tt_metal::CreateCircularBuffer(program, core_range, input_cb_config);
    cout << "input cb created with - " << num_input_tiles << " tiles" << std::endl;

    // CB to store halo data
    // hardcode to store 1 row of tiles
    uint32_t halo_input_cb_index = CB::c_intermed0;
    uint32_t num_halo_cb_input_tiles = num_input_tiles_in_row * 4;
    tt_metal::CircularBufferConfig halo_input_cb_config = tt_metal::CircularBufferConfig(num_halo_cb_input_tiles * single_tile_size, {{halo_input_cb_index, cb_data_format}})
		.set_page_size(halo_input_cb_index, single_tile_size);
    auto halo_input_cb = tt_metal::CreateCircularBuffer(program, core_range, halo_input_cb_config);

    // CB to store reader pattern array
    // read pattern array size == output_height
    uint32_t reader_pattern_array_size = output_shard_height;
    cout << "output_shard_height=" << output_shard_height << endl;
    uint32_t reader_pattern_array_cb_index = CB::c_intermed1;
    tt_metal::CircularBufferConfig reader_pattern_array_cb_config = tt_metal::CircularBufferConfig(reader_pattern_array_size * 4, {{reader_pattern_array_cb_index, DataFormat::Float16_b}})
		.set_page_size(reader_pattern_array_cb_index, 4);
    auto reader_pattern_array_cb = tt_metal::CreateCircularBuffer(program, core_range, reader_pattern_array_cb_config);
    cout << "reader pattern cb created with - " << reader_pattern_array_size * 4 << " bytes" << std::endl;

    // untilized CB has size - [32, full width]
    uint32_t untilize_cb_index = CB::c_intermed2;
    uint32_t num_tiles_untilize_cb = num_input_tiles_in_row;
    tt_metal::CircularBufferConfig untilize_cb_config = tt_metal::CircularBufferConfig(num_tiles_untilize_cb * single_tile_size, {{untilize_cb_index, cb_data_format}})
		.set_page_size(untilize_cb_index, single_tile_size);
    auto untilize_cb = tt_metal::CreateCircularBuffer(program, core_range, untilize_cb_config);

    uint32_t num_output_tiles_all_cores =  output.volume() / TILE_HW;
    assert(num_output_tiles_all_cores % num_cores == 0);
    assert((num_output_tiles_all_cores / num_cores) == num_output_tiles_in_row * num_rows_of_output_tiles);
    uint32_t num_output_tiles = (num_output_tiles_all_cores / num_cores);
    uint32_t untilize_downsampled_cb_index = CB::c_intermed3;
    uint32_t num_tiles_untilize_downsampled_cb = num_output_tiles; // untilize downsampled cb size == output size per core
    tt_metal::CircularBufferConfig untilize_downsampled_cb_config = tt_metal::CircularBufferConfig(num_tiles_untilize_downsampled_cb * single_tile_size, {{untilize_downsampled_cb_index, cb_data_format}})
		.set_page_size(untilize_downsampled_cb_index, single_tile_size);
    auto untilize_downsampled_cb = tt_metal::CreateCircularBuffer(program, core_range, untilize_downsampled_cb_config);

    uint32_t final_tilize_output_cb_index = CB::c_out0;
    uint32_t num_tiles_final_tilize_output_cb = num_output_tiles; // final output cb size == output size per core
    tt_metal::CircularBufferConfig final_tilize_output_cb_config = tt_metal::CircularBufferConfig(num_tiles_final_tilize_output_cb * single_tile_size, {{final_tilize_output_cb_index, cb_data_format}})
		.set_page_size(final_tilize_output_cb_index, single_tile_size);
    final_tilize_output_cb_config = final_tilize_output_cb_config.set_globally_allocated_address(output.buffer()->address());
    auto final_tilize_output_cb = tt_metal::CreateCircularBuffer(program, core_range, final_tilize_output_cb_config);

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) untilize_cb_index,
        (std::uint32_t) untilize_downsampled_cb_index,
        (std::uint32_t) final_tilize_output_cb_index,
        (std::uint32_t) reader_pattern_array_cb_index,
        (std::uint32_t) a.element_size(),
        (std::uint32_t) input_width_bytes,
        (std::uint32_t) halo_input_cb_index,
    };

    // Writer to downsample - drops rows from untilized cb
    tt_metal::KernelID downsample_writer_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_eager/tt_dnn/op_library/downsample/kernels/downsample_writer_kernel.cpp",
        core_range,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    vector<uint32_t> compute_args = {
        input_cb_index,
        halo_input_cb_index,
        untilize_cb_index,
        untilize_downsampled_cb_index,
        final_tilize_output_cb_index,
        num_input_tiles_in_row,
        num_rows_of_output_tiles,
        num_output_tiles_in_row,
    };

    auto downsample_compute_kernel_id = tt_metal::CreateComputeKernel(
        program,
        "tt_eager/tt_dnn/op_library/downsample/kernels/downsample_compute_kernel.cpp",
        core_range,
        tt_metal::ComputeConfig{.compile_args = compute_args}
    );

    // track img h, img w, across cores
    ImgTrackingVars v;
    uint32_t img_height = input_height_size_y;
    uint32_t img_width = input_height_size_x;

    uint32_t img_stride_h = height_y_stride;
    uint32_t img_stride_w = height_x_stride;


    bool halo_from_prev_core = false;
    uint32_t halo_start_img_h = 0; // this should not be skipped row
    uint32_t halo_start_img_w = 0; // this should not be skipped row
    uint32_t halo_start_input_flat_h = 0;
    uint32_t halo_end_input_flat_h = 0;
    CoreCoord prev_core = {0,0};

    // !!ASSUMPTION!! in determining core coordinate is that all 12 cores in x dim are used
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i % ncores_x_full_grid, i / ncores_x_full_grid};
        cout << "i=" << i << endl;

        // Halo region
        uint32_t output_flat_h_with_halo_from_prev_core = 0;
        uint32_t input_flat_h_len_halo = halo_end_input_flat_h - halo_start_input_flat_h;
        if (halo_from_prev_core) {
            cout << "   halo_start_input_flat_h=" << halo_start_input_flat_h << endl;
            cout << "   halo_end_input_flat_h=" << halo_end_input_flat_h << endl;
        }
        bool halo_read_enabled = false;
        DownsampleReadPatternParams halo_read_pattern_params;
        uint32_t halo_noc_x = 0;
        uint32_t halo_noc_y = 0;
        uint32_t halo_start_addr = 0;
        uint32_t halo_addr_offset = 0;
        uint32_t halo_num_tiles = 0;
        uint32_t halo_size_bytes = 0;
        uint32_t halo_input_num_rows_of_tiles = 0;
        uint32_t halo_read_pattern_offset = 0;
        uint32_t local_read_pattern_offset = 0;
        if (v.input_flat_h != 0) {
            // halo region of previous core
            TT_ASSERT(i != 0);
            halo_read_enabled = true;
            TT_ASSERT(v.input_flat_h < input_shard_height);
            // get halo start tile address from height idx
            uint32_t halo_start_tile_id_h = v.input_flat_h / TILE_HEIGHT;
            TT_ASSERT(input_shard_height - v.input_flat_h <= TILE_HEIGHT * 4); // halo input cb is hardcoded to store only 4 rows of tiles for now. TODO: allocate bigger CB or read in blocks
            // get halo size
            halo_size_bytes = (input_shard_height - (halo_start_tile_id_h * TILE_HEIGHT)) * input_width * a.element_size();
            TT_ASSERT(halo_size_bytes % single_tile_size == 0);
            halo_num_tiles = halo_size_bytes / single_tile_size;
            TT_ASSERT(halo_num_tiles <= num_halo_cb_input_tiles);
            TT_ASSERT(halo_num_tiles % num_input_tiles_in_row == 0);
            halo_input_num_rows_of_tiles = halo_num_tiles / num_input_tiles_in_row;
            halo_addr_offset = num_input_tiles_in_row * halo_start_tile_id_h * single_tile_size;
            halo_start_addr = GetCircularBufferConfig(program, input_cb).globally_allocated_address().value();
            TT_ASSERT((halo_start_addr + halo_addr_offset) % 32 == 0); // read address should be 32 byte aligned
            auto halo_noc_coords = device->worker_core_from_logical_core(prev_core);
            halo_noc_x = halo_noc_coords.x;
            halo_noc_y = halo_noc_coords.y;
            TT_ASSERT(v.input_flat_h >= halo_start_tile_id_h * TILE_HEIGHT);
            halo_read_pattern_offset = v.input_flat_h - (halo_start_tile_id_h * TILE_HEIGHT);
            local_read_pattern_offset = halo_input_num_rows_of_tiles * TILE_HEIGHT;
            halo_read_pattern_params = generate_downsample_read_pattern(v, img_height, img_width, img_stride_h, img_stride_w, input_shard_height, output_shard_height);
        }
        // local core
        TT_ASSERT(v.input_flat_h == 0);
        TT_ASSERT(v.output_flat_h < output_shard_height);
        DownsampleReadPatternParams local_read_pattern_params = generate_downsample_read_pattern(v, img_height, img_width, img_stride_h, img_stride_w, input_shard_height, output_shard_height);
        uint32_t local_input_num_rows_of_tiles = num_rows_of_input_tiles;
        if (v.input_flat_h != 0) {
            local_input_num_rows_of_tiles = std::ceil( (double) v.input_flat_h / (double) TILE_HEIGHT);
        }
        TT_ASSERT(local_input_num_rows_of_tiles <= num_rows_of_input_tiles);
        TT_ASSERT(v.output_flat_h == 0);

        // Compile runtime args
        vector<uint32_t> compile_rt_kernel_args = {
            local_input_num_rows_of_tiles,
            halo_read_enabled,
            halo_input_num_rows_of_tiles,
        };

        tt_metal::SetRuntimeArgs(
            program,
            downsample_compute_kernel_id,
            core,
            compile_rt_kernel_args
        );

        // Writer runtime args
        vector<uint32_t> writer_kernel_args = {
            (uint32_t) input_height_size_y,
            (uint32_t) input_height_size_x,
            (uint32_t) height_y_stride,
            (uint32_t) height_x_stride,

            // halo args
            halo_read_enabled,
            halo_noc_x,
            halo_noc_y,
            halo_num_tiles,
            halo_start_addr,
            halo_addr_offset,
            halo_size_bytes,

            // halo read pattern args
            halo_read_pattern_offset,
            halo_read_pattern_params.top_partial_middle_aligned_row_width,
            halo_read_pattern_params.skip_top_partial_middle_aligned_row,
            halo_read_pattern_params.top_partial_right_aligned_row_width,
            halo_read_pattern_params.skip_top_partial_right_aligned_row,
            halo_read_pattern_params.num_rows_top_partial_image,
            halo_read_pattern_params.num_skip_rows_top_partial_image,
            halo_read_pattern_params.num_full_images,
            halo_read_pattern_params.num_rows_bottom_partial_image,
            halo_read_pattern_params.num_skip_rows_bottom_partial_image,
            halo_read_pattern_params.bottom_partial_left_aligned_row_width,
            halo_read_pattern_params.skip_bottom_partial_left_aligned_row,

            // local read pattern args
            local_read_pattern_offset,
            local_read_pattern_params.top_partial_middle_aligned_row_width,
            local_read_pattern_params.skip_top_partial_middle_aligned_row,
            local_read_pattern_params.top_partial_right_aligned_row_width,
            local_read_pattern_params.skip_top_partial_right_aligned_row,
            local_read_pattern_params.num_rows_top_partial_image,
            local_read_pattern_params.num_skip_rows_top_partial_image,
            local_read_pattern_params.num_full_images,
            local_read_pattern_params.num_rows_bottom_partial_image,
            local_read_pattern_params.num_skip_rows_bottom_partial_image,
            local_read_pattern_params.bottom_partial_left_aligned_row_width,
            local_read_pattern_params.skip_bottom_partial_left_aligned_row,

            halo_input_num_rows_of_tiles + local_input_num_rows_of_tiles,
            num_input_tiles_in_row,
            num_output_tiles,

            (uint32_t) false
        };

        tt_metal::SetRuntimeArgs(
            program,
            downsample_writer_kernel_id,
            core,
            writer_kernel_args
        );
        prev_core = core;
    }

    auto override_runtime_args_callback = [
        input_cb=input_cb,
        final_tilize_output_cb=final_tilize_output_cb,
        downsample_writer_kernel_id=downsample_writer_kernel_id,
        num_cores=num_cores,
        ncores_x_full_grid=ncores_x_full_grid
    ](
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        auto& input_cb_config = GetCircularBufferConfig(program, input_cb);
        input_cb_config.set_globally_allocated_address(src_buffer->address());
        auto& final_tilize_output_cb_config = GetCircularBufferConfig(program, final_tilize_output_cb);
        final_tilize_output_cb_config.set_globally_allocated_address(dst_buffer->address());
        for (uint32_t i = 0; i < num_cores; i++) {
            CoreCoord core = {i % ncores_x_full_grid, i / ncores_x_full_grid};
            if (i != 0) {
                auto runtime_args = GetRuntimeArgs(program, downsample_writer_kernel_id, core);
                runtime_args[8] = src_buffer->address();
                SetRuntimeArgs(program, downsample_writer_kernel_id, core, runtime_args);
            }
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
