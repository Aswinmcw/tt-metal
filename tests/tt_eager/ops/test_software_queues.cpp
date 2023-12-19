// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#include "common/constants.hpp"
#include "tensor/owned_buffer.hpp"
#include "tensor/owned_buffer_functions.hpp"
#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_dnn/op_library/pad/pad_op.hpp"
#include "tt_dnn/op_library/program_cache.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_numpy/functions.hpp"

using tt::tt_metal::DataType;
using tt::tt_metal::Device;

using tt::tt_metal::Layout;
using tt::tt_metal::OwnedStorage;
using tt::tt_metal::Shape;
using tt::tt_metal::Tensor;

auto Sqrt = tt::tt_metal::EltwiseUnary{{tt::tt_metal::UnaryWithParam{tt::tt_metal::UnaryOpType::SQRT, std::nullopt}}};
void sqrt(Queue& queue, Tensor& input, Tensor& output) { EnqueueOperation(queue, Sqrt, {input, output}); }
void sqrt(Tensor& input, Tensor& output) { EnqueueOperation(GetDefaultQueue(), Sqrt, {input, output}); }

int main() {
    Tensor host_input_tensor = ...;

    Queue data_queue = GetDefaultQueue();
    Queue math_queue = CreateNewQueue();
    Queue third_queue = CreateNewQueue();  // throw error because only 2 queues are supported
    size_t num_queues = GetNumQueues();

    Event data_transfer_event;
    Event math_event;

    std::shared_ptr<Buffer> device_input_buffer = create_device_buffer(device, size);
    EnqueueAllocateDeviceBuffer(data_queue, device_input_buffer);
    Tensor device_input_tensor = Tensor{DeviceStorage{device_input_buffer, ...}, ...};
    EnqueueHostToDeviceTransfer(data_queue, host_input_tensor, device_input_tensor);

    std::shared<Buffer> device_output_buffer{device};
    EnqueueAllocateDeviceBuffer(data_queue, device_output_buffer);
    Tensor device_output_tensor = Tensor{DeviceStorage{device_output_buffer, ...}, ...};

    RecordEvent(data_queue, data_transfer_event);
    WaitForEvent(math_queue, data_transfer_event);

    EnqueueOperation(math_queue, Sqrt, {device_input_tensor}, {device_output_tensor});
    // OR to run on default_queue
    sqrt(device_input_tensor, device_output_tensor);

    RecordEvent(math_queue, math_event);

    owned_buffer::Buffer host_output_buffer;
    EnqueueAllocateHostBuffer(data_queue, host_output_buffer);
    Tensor host_output_tensor = Tensor{OwnedStorage{host_buffer, ...}, ...};

    WaitForEvent(data_queue, math_event);
    EnqueueDeviceToHostTransfer(data_queue, device_output_tensor, host_output_tensor);
}

EnqueueAllocateDeviceBuffer(Queue&, std::shared_ptr<Buffer>&);
EnqueueAllocateHostBuffer(Queue&, owned_buffer::Buffer&);

EnqueueHostToDeviceTransfer(Queue&, Tensor& src, Tensor& dst);
EnqueueDeviceToHostTransfer(Queue&, Tensor& src, Tensor& dst);

RecordEvent(Queue&, Event&);
WaitForEvent(Queue&, Event&);

EnqueueOperation(
    Queue&, DeviceOperation&, const std::vector<Tensor>& input_tensors, const std::vector<Tensor>& output_tensors);
EnqueueOperation(
    Queue&,
    DeviceOperation&,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<Tensor>>& optional_input_tensors,
    const std::vector<Tensor>& output_tensors);
