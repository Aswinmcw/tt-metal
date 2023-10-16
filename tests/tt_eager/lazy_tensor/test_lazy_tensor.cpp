// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#include "common/constants.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_dnn/lazy_tensor.hpp"

using tt::tt_metal::Device;
using tt::tt_metal::Shape;
using tt::tt_metal::Tensor;
using tt::tt_metal::DataType;
using tt::tt_metal::Layout;

int main() {

    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    int device_id = 0;
    auto device = tt::tt_metal::CreateDevice(device_id);
    tt::tt_metal::AutoFormat::SetDefaultDevice(device);

    auto shape = Shape{1, 1, TILE_HEIGHT, TILE_WIDTH};

    std::cout << "Initialize queues" << std::endl;
    auto queue_0 = tt::lazy::Queue();
    auto queue_1 = tt::lazy::Queue();

    std::cout << "Queue up operations" << std::endl;
    auto lazy_input_tensor_a = queue_0.push(tt::lazy::ones(shape));
    auto lazy_input_tensor_b = queue_0.push(tt::lazy::ones(shape));
    auto lazy_output_tensor =  queue_0.push(tt::lazy::matmul(lazy_input_tensor_a, lazy_input_tensor_b));

    queue_1.push(tt::lazy::ones(shape));

    std::cout << "Wait until queue 0 finishes" << std::endl;
    queue_0.finish();

    std::cout << "Wait until queue 1 finishes" << std::endl;
    queue_1.finish();

    Tensor output_tensor = queue_0.get_tensor(lazy_output_tensor);
    output_tensor.print();

    TT_ASSERT(tt::tt_metal::CloseDevice(device));

    return 0;
}
