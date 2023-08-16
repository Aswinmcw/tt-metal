import time
import tt_lib
import torch
import numpy as np
from loguru import logger
# from tt_lib.utils import _nearest_32
from os import environ
import math
import struct

### Math operations ###
def _nearest_32(x):
    return math.ceil(x / 32) * 32

def enable_persistent_kernel_cache():
    """
    Enables persistent compiled kernel caching - disables recompiling the kernels for the duration of running process if built_kernels/.../hash directory with kernel binaries is present.
    """
    tt_lib.device.EnablePersistentKernelCache()


def disable_persistent_kernel_cache():
    """
    Disables persistent compiled kernel caching. This is the default state.
    """
    tt_lib.device.DisablePersistentKernelCache()


def enable_compilation_reports():
    """
    Enables generating reports of compilation statistics in .reports/tt_metal dir
    """
    return tt_lib.device.EnableCompilationReports()


def disable_compilation_reports():
    """
    Disables generating reports of compilation statistics
    """
    return tt_lib.device.DisableCompilationReports()


def enable_memory_reports():
    """
    Enables generating reports of memory allocation statistics in .reports/tt_metal dir
    """
    return tt_lib.device.EnableMemoryReports()


def disable_memory_reports():
    """
    Disables generating reports of memory allocation statistics
    """
    return tt_lib.device.DisableMemoryReports()


### Tensor conversion ###
def torch2tt_tensor(
    py_tensor: torch.Tensor,
    tt_device,
    tt_layout=tt_lib.tensor.Layout.TILE,
    tt_memory_config=tt_lib.tensor.MemoryConfig(True),
    tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
):
    size = list(py_tensor.size())

    while len(size) < 4:
        size.insert(0, 1)

    tt_tensor = tt_lib.tensor.Tensor(py_tensor.reshape(size), tt_dtype)
    tt_tensor = tt_tensor.to(tt_layout)

    if tt_device is not None:
        tt_tensor = tt_tensor.to(tt_device, tt_memory_config)
    else:
        tt_tensor = tt_tensor.cpu()

    return tt_tensor


def tt2torch_tensor(tt_tensor, tt_host=None):
    tt_output = tt_tensor.cpu()
    if tt_output.layout() != tt_lib.tensor.Layout.ROW_MAJOR:
        tt_output = tt_output.to(tt_lib.tensor.Layout.ROW_MAJOR)
    return tt_output.to_torch()

def tt_to_torch_tensor(tt_tensor):
    tt_output = tt_tensor.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR)
    return tt_output.to_torch()

def torch_to_tt_tensor_rm(py_tensor, device, shape=None, put_on_device=True):
    if shape is None:
        shape = list(py_tensor.size())
        while len(shape) < 4:
            shape.insert(0, 1)

    tt_tensor = (
         tt_lib.tensor.Tensor(py_tensor.reshape(shape), tt_lib.tensor.DataType.BFLOAT16)
    )
    if put_on_device:
        tt_tensor = tt_tensor.to(device)
    return tt_tensor

def torch_to_tt_tensor(py_tensor, device):
    shape = list(py_tensor.size())
    while len(shape) < 4:
        shape.insert(0, 1)

    tt_tensor = (
        tt_lib.tensor.Tensor(py_tensor.reshape(shape), tt_lib.tensor.DataType.BFLOAT16)
        .to(tt_lib.tensor.Layout.TILE)     # change memory layout of TT Tensor to TILE (as operation that will use it expects TILE layout)
        .to(device)                         # move TT Tensor from host to TT accelerator device (device is of type tt_lib.device.Device)
    )

    return tt_tensor

### Padding / Unpadding ###
def pad_by_zero(x: torch.Tensor, device):
    initial_shape = x.shape
    pad_shape = list(x.shape)
    while len(pad_shape) < 4:
        pad_shape.insert(0, 1)
    if pad_shape[3] % 32 != 0 or pad_shape[2] % 32 != 0:
        tt_tensor = tt_lib.tensor.Tensor(
            x.reshape(pad_shape), tt_lib.tensor.DataType.BFLOAT16
        )
        x = tt_tensor.pad(
            (
                pad_shape[0],
                pad_shape[1],
                _nearest_32(pad_shape[2]),
                _nearest_32(pad_shape[3]),
            ),
            (0, 0, 0, 0),
            0,
        )
        x = x.to(tt_lib.tensor.Layout.TILE).to(device)

    else:
        x = torch2tt_tensor(x, device)
    return x, initial_shape


def unpad_from_zero(x, desired_shape):
    if x.shape()[-1] == desired_shape[-1] and x.shape()[-2] == desired_shape[-2]:
        x = tt2torch_tensor(x)
    else:
        x = x.cpu()
        if x.layout() != tt_lib.tensor.Layout.ROW_MAJOR:
            x = x.to(tt_lib.tensor.Layout.ROW_MAJOR)
        x = x.unpad(
            (0, 0, 0, 0),
            (
                desired_shape[0] - 1,
                desired_shape[1] - 1,
                desired_shape[2] - 1,
                desired_shape[3] - 1,
            ),
        )

        x = x.to_torch()
    return x

def pad_activation(x):
    """
    This function pads an activation with 0s as a pre-preprocessing step to tilization.

    In the 2d case, it pads a vector to the right with 0s, and in the 2+d case,
    it pads the bottom and right corners of the last two dimensions.

    :param x: Input PyTorch Tensor
    :type x: class:`torch.Tensor`

    WARNING: This function should eventually be retired in favour of padding on device
    """
    nearest_32 = _nearest_32

    assert isinstance(x, torch.Tensor), "Input to this function must be an instance of torch.Tensor"
    assert len(x.shape) >= 1 and len(x.shape) <= 4, "Only tensors with dimension 1-4 supported"
    if len(x.shape) == 1: # (num_features,)
        padded_tensor = torch.zeros(1, 1, 32, nearest_32(x.shape[0]))
        padded_tensor[:, 0, 0, :x.shape[0]] = x
    elif len(x.shape) == 2: # (batch, num features)
        padded_tensor = torch.zeros(x.shape[0], 1, 32, nearest_32(x.shape[1]))
        padded_tensor[:, 0, 0, :x.shape[1]] = x
    elif len(x.shape) == 3: # (batch, num features y, num features x)
        padded_tensor = torch.zeros(x.shape[0], 1, nearest_32(x.shape[-2]), nearest_32(x.shape[-1]))
        padded_tensor[..., 0, :x.shape[-2], :x.shape[-1]] = x
    else: # (batch, num channels, num features y, num features x)
        padded_tensor = torch.zeros(*x.shape[:-2], nearest_32(x.shape[-2]), nearest_32(x.shape[-1]))
        padded_tensor[..., :x.shape[-2], :x.shape[-1]] = x
    return padded_tensor

def pad_weight(x):
    """
    This function pads a weight/bias with 0s as a pre-preprocessing step to tilization.

    tt_tensor = tt_lib.tensor.Tensor(
        py_tensor.reshape(shape), tt_lib.tensor.DataType.BFLOAT16
    In the 2d case, it pads a vector to the right with 0s, and in the 2+d case,
    it pads the bottom and right corners of the last two dimensions.

    :param x: Input PyTorch Tensor
    :type x: class:`torch.Tensor`

    WARNING: This function should eventually be retired in favour of padding on device
    """
    nearest_32 = _nearest_32

    assert isinstance(x, torch.Tensor), "Input to this function must be an instance of torch.Tensor"
    assert len(x.shape) >= 1 and len(x.shape) <= 4, "Only tensors with dimension 1-4 supported"

    if len(x.shape) == 1: # (num_features,)
        padded_tensor = torch.zeros(1, 1, 32, nearest_32(x.shape[0]))
        padded_tensor[:, 0, 0, :x.shape[0]] = x
    elif len(x.shape) == 2: # (r_features, c_features)
        padded_tensor = torch.zeros(1, 1, nearest_32(x.shape[0]), nearest_32(x.shape[1]))
        padded_tensor[:, 0, :x.shape[0], :x.shape[1]] = x
    else:
        padded_tensor = torch.zeros(*x.shape[:-2], nearest_32(x.shape[-2]), nearest_32(x.shape[-1]))
        padded_tensor[..., :x.shape[-2], :x.shape[-1]] = x

    return padded_tensor


def convert_weights_2d_matrix(weights, w_shape):
    """
    :param weights: Input PyTorch Tensor
    :type weights: class:`torch.Tensor`
    """
    ret_shape = [1,1,w_shape[0],w_shape[1]*w_shape[2]*w_shape[3]]
    if isinstance(weights, torch.Tensor):
        ret = torch.zeros(np.prod(ret_shape))
    else:
        ret = np.zeros(np.prod(ret_shape))
    idx = 0
    for k in range(w_shape[0]):
        for r in range(w_shape[2]):
            for s in range(w_shape[3]):
                for c in range(w_shape[1]):
                    ret[idx] = weights[k][c][r][s]
                    idx+=1
    assert idx == np.prod(ret_shape)
    return ret.reshape(ret_shape).transpose(2,3)

def convert_act_2d_matrix(activation, kernel_y, kernel_x, stride_y, stride_x, pad_y, pad_x):
    """
    :param activation: Input PyTorch Tensor
    :type activation: class:`torch.Tensor`
    """
    N = activation.shape[0]
    C = activation.shape[1]
    H = activation.shape[2]
    W = activation.shape[3]

    OH = (int) ((H - kernel_y + 2*pad_y) // stride_y) + 1
    OW = ((W - kernel_x + 2*pad_x) // stride_x) + 1
    nrows = OH*OW
    ncols = C*kernel_x*kernel_y
    ret_shape = [1,N,nrows,ncols]
    if isinstance(activation, torch.Tensor):
        ret = torch.zeros(np.prod(ret_shape))
    else:
        ret = np.zeros(np.prod(ret_shape))
    idx = 0
    for n in range(N):
        for h in range(-1*pad_y, H+pad_y-kernel_y+1, stride_y):
            for w in range(-1*pad_x, W+pad_x-kernel_x+1, stride_x):
                for r in range(kernel_y):
                    for s in range(kernel_x):
                        for c in range(C):
                            h_offs = h+r
                            w_offs = w+s
                            pad = h_offs < 0 or h_offs >= H or w_offs < 0 or w_offs >= W
                            ret[idx] = 0 if pad else activation[n][c][h_offs][w_offs]
                            idx+=1
    assert idx == np.prod(ret_shape)
    return ret.reshape(ret_shape)


### Tilizing / Untilizing ###
def tilize(x):
    """
    This function tilizes a tensor. The last two tensor dims must be divisible by 32, after which this function
    produces row major tiles and creates faces. The output of this function is a flattened list that
    we can send to the device.

    :param x: Input PyTorch Tensor
    :type x: class:`torch.Tensor`

    WARNING: This function should eventually be retired in favour of fully tilizing on device.
    """
    nearest_32 = _nearest_32

    assert isinstance(x, (torch.Tensor, np.ndarray)), "Input to this function must be an instance of torch.Tensor or np.array"
    assert len(x.shape) == 4, "Only 4D tensors suppported"
    assert (x.shape[-2] % 32) == 0 and (x.shape[-1] % 32) == 0, "The last two dimensions of the tensor must be divisible by 32"

    if isinstance(x, torch.Tensor):
        ret = torch.zeros(np.prod(x.shape))
    else:
        ret = np.zeros(np.prod(x.shape))

    idx = 0
    for B in range(x.shape[0]):
        for C in range(x.shape[1]):
            for H in range(0, x.shape[2], 32):
                for W in range(0, x.shape[3], 32):
                    unfaced_tile = x[B, C, H:H + 32, W:W + 32]

                    face0 = unfaced_tile[:16, :16]
                    face1 = unfaced_tile[:16, 16:]
                    face2 = unfaced_tile[16:, :16]
                    face3 = unfaced_tile[16:, 16:]

                    for face in (face0, face1, face2, face3):
                        ret[idx:idx + 256] = face.reshape(-1)
                        idx += 256

    return ret.reshape(x.shape)

def tilize_to_list(x):
    """
    Tilize a PyTorch and then return the values as a flat list. The last two
    tensor dims must be divisible by 32, after which this function produces row
    major tiles and creates faces.

    :param x: Input PyTorch Tensor
    :type x: class:`torch.Tensor`

    WARNING: This function should eventually be retired in favour of fully tilizing on device.
    """

    return tilize(x).reshape(-1).tolist()

def untilize(x):
    """
    This function untilizes a tensor to row major format.

    :param x: Input PyTorch Tensor
    :type x: class:`torch.Tensor`

    WARNING: This function should eventually be retired in favour of fully tilizing on device.
    """
    nearest_32 = _nearest_32

    assert isinstance(x, (torch.Tensor, np.ndarray)), "Input to this function must be an instance of torch.Tensor"
    assert len(x.shape) == 4, "Only 4D tensors suppported"
    assert (x.shape[-2] % 32) == 0 and (x.shape[-1] % 32) == 0, "The last two dimensions of the tensor must be divisible by 32"

    if isinstance(x, torch.Tensor):
        ret = torch.zeros(x.shape)
    else:
        ret = np.zeros(x.shape)

    for B in range(x.shape[0]):
        for C in range(x.shape[1]):
            x_hw = x[B,C,:].reshape(-1)
            hw = 0
            for h in range(0, x.shape[2], 32):
                for w in range(0, x.shape[3], 32):
                    f_tile = x_hw[hw:hw+256].reshape(16, 16)
                    ret[B, C, h:h+16, w:w+16] = f_tile

                    f_tile = x_hw[hw+256:hw+512].reshape(16, 16)
                    ret[B, C, h:h+16, w+16:w+32] = f_tile

                    f_tile = x_hw[hw+512:hw+768].reshape(16, 16)
                    ret[B, C, h+16:h+32, w:w+16] = f_tile

                    f_tile = x_hw[hw+768:hw+1024].reshape(16, 16)
                    ret[B, C, h+16:h+32, w+16:w+32] = f_tile
                    hw += 1024 # traverse tiles in RM-order

    return ret

### Measuring accuracy and other metrics ###
def is_close(a, b, rtol=1e-2, atol=1e-2, max_mag=2.0, max_mag_fraction=0.02):
    """
    A variant of np.isclose with logging.
    """
    absdiff = (a - b).abs()
    reldiff1 = (a.abs() / b.abs()) - 1.0
    reldiff2 = (a.abs() + 1.0) / (b.abs() + 1.0) - 1.0  # in case b.abs() is 0
    reldiff_or = torch.logical_or(reldiff1.abs() < rtol, reldiff2.abs() < rtol)
    max_mag_ok = absdiff < max_mag * max_mag_fraction

    or_abs_rel = torch.logical_or(absdiff < atol, reldiff_or)
    or_abs_rel = torch.logical_or(or_abs_rel, max_mag_ok)
    debug_index = or_abs_rel.to(torch.int32).argmin().item()

    if not or_abs_rel.reshape(-1)[debug_index]:
        logger.info("isclose mismatch at index=", debug_index)
        logger.info(a.reshape(-1)[debug_index])
        logger.info(b.reshape(-1)[debug_index])
        logger.info("reldiff1=", reldiff1.reshape(-1)[debug_index])
        logger.info("reldiff2=", reldiff2.reshape(-1)[debug_index])
        logger.info("absdiff=", absdiff.reshape(-1)[debug_index])

        HT = a.shape[-2] // 32
        WT = a.shape[-1] // 32
        hwt = debug_index//1024
        wt = hwt % WT
        ht = hwt // WT
        h = (debug_index % 1024) // 32
        w = (debug_index % 1024) % 32

        logger.info("****    at ", debug_index, " --- ", "HTWT=", ht, wt, "HW=", h, w)

    return torch.all(or_abs_rel)


def comp_allclose(golden, calculated, rtol=1e-05, atol=1e-08):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    atol_delta = torch.max(torch.abs(golden - calculated)).item()
    rtol_delta = torch.max(
        torch.abs(golden - calculated) / torch.abs(calculated)
    ).item()
    return (
        torch.allclose(golden, calculated, rtol, atol, True),
        f"Max ATOL Delta: {atol_delta}, Max RTOL Delta: {rtol_delta}",
    )


def comp_pcc(golden, calculated, pcc=0.99):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    tt_tensor = (
        tt_lib.tensor.Tensor(py_tensor.reshape(shape), tt_lib.tensor.DataType.BFLOAT16)
        .to(
            tt_lib.tensor.Layout.TILE
        )  # change memory layout of TT Tensor to TILE (as operation that will use it expects TILE layout)
        .to(
            device
        )  # move TT Tensor from host to TT accelerator device (device is of type tt_lib.device.Device)
    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        logger.warning("Both tensors are 'nan'")
        return True, f"PCC: {1.0}"

    if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
        logger.error("One tensor is all nan, the other is not.")
        return False, f"PCC: {0.0}"

    # Test if either is completely zero
    if torch.any(golden.bool()) != torch.any(calculated.bool()):
        logger.error("One tensor is all zero")
        return False, f"PCC: {0.0}"

    # For now, mask all infs and nans so that we check the rest... TODO
    golden = golden.clone()
    golden[
        torch.logical_or(
            torch.isnan(golden),
            torch.logical_or(torch.isinf(golden), torch.isneginf(golden)),
        )
    ] = 0
    calculated = calculated.clone()
    calculated[
        torch.logical_or(
            torch.isnan(calculated),
            torch.logical_or(torch.isinf(calculated), torch.isneginf(calculated)),
        )
    ] = 0

    if torch.equal(golden, calculated):
        return True, f"PCC: {1.0}"

    if golden.dtype == torch.bfloat16:
        golden = golden.type(torch.float32)
        calculated = calculated.type(torch.float32)
    cal_pcc = np.min(
        np.ma.corrcoef(
            np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
            np.ma.masked_invalid(torch.squeeze(calculated).detach().numpy()).flatten(),
        )
    )

    if isinstance(cal_pcc, np.ma.core.MaskedConstant):
        return True, f"PCC: {1.0}"

    return cal_pcc >= pcc, f"PCC: {cal_pcc}"


def comp_allclose_and_pcc(golden, calculated, rtol=1e-05, atol=1e-08, pcc=0.99):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    passing = True
    output = ""
    passing_allclose, output_allclose = comp_allclose(golden, calculated, rtol, atol)
    passing &= passing_allclose
    output += output_allclose
    if torch.numel(golden) != 1:
        passing_pcc, output_pcc = comp_pcc(golden, calculated, pcc)
        passing &= passing_pcc
        output += f", {output_pcc}"

    return passing, output


def get_oom_of_float(float_lst):
    """
    Given a list of floats, returns a list of the order or magnitudes
    of the floats. Useful when you want to make sure that even if your
    tt outputs don't match pytorch all that well, they are at least
    on the same order of magnitude
    """
    ooms = []
    for el in float_lst:
        str_el = str(el)
        if "e" in str_el:
            oom = int(str_el.split("e")[1])
        elif str_el[:2] == "0.":
            str_el = str_el.split(".")[1]

            oom = -1
            for e in str_el:
                if e != "0":
                    break
                oom -= 1
        else:
            oom = len(str_el.split(".")[0])

        ooms.append(oom)

    return ooms

def print_diff_argmax(a, b, annotation = ""):
    """
    Prints out the value of both tensors at a point where the absolute difference is the largest.
    """
    absdiff = (a-b).abs()
    argmax = absdiff.argmax().item()
    diff = absdiff.reshape(-1)[argmax]
    rela = a.abs()/(torch.max(a.abs(), b.abs()))
    relb = b.abs()/(torch.max(a.abs(), b.abs()))
    HT = a.shape[-2] // 32
    WT = a.shape[-1] // 32
    hwt = argmax//1024
    wt = hwt % WT
    ht = hwt // WT
    h = (argmax % 1024) // 32
    w = (argmax % 1024) % 32
    print("Abs diff=", diff, " at ", argmax, " --- ", annotation, "HTWT=", ht, wt, "HW=", h, w)
    print("  (a=", a.reshape(-1)[argmax].item(), ")")
    print("  (b=", b.reshape(-1)[argmax].item(), ")")
    print("  Rel a=", rela.reshape(-1)[argmax], " at ", argmax)
    print("  Rel b=", relb.reshape(-1)[argmax], " at ", argmax)
    return diff.item()

def print_diff_tt_pyt(a, b, annotation=""):
    # first convert a pytorch tensor argument b to tt
    padded_b = pad_weight(b)
    pyt_a = tt2torch(a)  # untilizes also
    return print_diff_argmax(pyt_a, padded_b, annotation)


def ttP(x, count=4, offset=0, stride=1):
    if type(x) == torch.Tensor:
        t1 = x.reshape(-1)
    else:
        tt_out = x.cpu()
        torch_out = untilize(tt_out.to_torch())
        t1 = torch_out.reshape(-1)
    print("Tensor vals: (", end="")
    for j in range(offset, offset + count * stride, stride):
        print(t1[j].item(), " ", end="")
    print(")")


def prep_report(model_name: str, batch_size: int, inference_and_compile_time: float, inference_time: float, expected_compile_time: float, expected_inference_time: float, comments: str, inference_time_cpu: float=None):
    today = time.strftime("%Y_%m_%d")

    def write_dict_to_file(csv_path, dict_res):
        columns = ", ".join([str(d) for d in dict_res.keys()])
        # values = ", ".join([("{:.2f}".format(d) if isinstance(d, float) else str(d)) for d in dict_res.values()])
        values = ", ".join([d for d in dict_res.values()])

        with open(csv_path, "w") as csvfile:
            csvfile.write(columns)
            csvfile.write("\n")
            csvfile.write(values)


    compile_time = inference_and_compile_time - inference_time
    gs_throughput = "{:.4f}".format(batch_size * (1/inference_time))
    cpu_throughput = batch_size * (1/inference_time_cpu) if inference_time_cpu else "unknown"
    cpu_throughput = "{:.4f}".format(cpu_throughput) if not isinstance(cpu_throughput, str) else cpu_throughput
    dict_res = {
        "Model": model_name,
        "Setting": comments,
        "Batch": str(batch_size),
        "First Run (sec)": "{:.2f}".format(inference_and_compile_time),
        "Second Run (sec)":  "{:.2f}".format(inference_time),
        "Compile Time (sec)": "{:.2f}".format(compile_time),
        "Expected Compile Time (sec)": "{:.2f}".format(expected_compile_time),
        "Inference Time GS (sec)": "{:.4f}".format(inference_time),
        "Expected Inference Time GS (sec)": "{:.4f}".format(expected_inference_time),
        "Throughput GS (batch*inf/sec)": gs_throughput,
        "Inference Time CPU (sec)": "{:.4f}".format(inference_time_cpu),
        "Throughput CPU (batch*inf/sec)": cpu_throughput,
    }

    csv_file = f"perf_{model_name}_{today}.csv"
    write_dict_to_file(csv_file, dict_res)


### Conv related operations ###
def read_conv_act_into_mm_act_block(conv_act, act_address_map_index, address_map, address_map_this_block_size, act_block_h, act_block_w):
    mm_act_block_shape = [1,1,act_block_h*32, act_block_w*32]
    mm_act_block_size = act_block_h*act_block_w*1024
    mm_act_block = torch.zeros(mm_act_block_size, dtype=torch.bfloat16).float()
    for i in range(0, address_map_this_block_size, 4):
        src_address = address_map[act_address_map_index]
        dst_address = address_map[act_address_map_index+1]
        read_size = address_map[act_address_map_index+2]
        pad = address_map[act_address_map_index+3]
        for s in range(read_size):
            assert(dst_address+s < mm_act_block_size)
            if pad:
                mm_act_block[dst_address+s] = 0
            else:
                assert(src_address+s < len(conv_act))
                mm_act_block[dst_address+s] = conv_act[src_address+s]
        act_address_map_index += 4
    return (mm_act_block.reshape(mm_act_block_shape), act_address_map_index)

def read_conv_weight_into_mm_weight_block(conv_weight, weight_address_map_index, weight_address_map, weight_address_map_this_block_size, weight_block_h, weight_block_w):
    mm_weight_block_shape = [1,1,weight_block_h*32, weight_block_w*32]
    mm_weight_block_size = weight_block_h*weight_block_w*1024
    mm_weight_block = torch.zeros(mm_weight_block_size, dtype=torch.bfloat16).float()
    for i in range(0, weight_address_map_this_block_size, 4):
        src_address = weight_address_map[weight_address_map_index]
        dst_address = weight_address_map[weight_address_map_index+1]
        read_size = weight_address_map[weight_address_map_index+2]
        pad = weight_address_map[weight_address_map_index+3]
        for s in range(read_size):
            assert(dst_address+s < mm_weight_block_size)
            if pad:
                mm_weight_block[dst_address+s] = 0
            else:
                assert(src_address+s < len(conv_weight))
                mm_weight_block[dst_address+s] = conv_weight[src_address+s]
        weight_address_map_index += 4
    return (mm_weight_block.reshape(mm_weight_block_shape), weight_address_map_index)

def blocked_mm_with_conv_act(conv_act,
                            mm_weight,
                            act_address_map,
                            weight_address_map,
                            num_blocks_act_h,
                            num_blocks_act_w,
                            num_blocks_weight_w,
                            act_block_h,
                            act_block_w,
                            weight_block_w):
    # act refers to conv activation tensor
    # weight refers to conv weight tensor
    mm_output_shape = [1,1,num_blocks_act_h*act_block_h*32,num_blocks_weight_w*weight_block_w*32]
    ret = torch.zeros(mm_output_shape, dtype=torch.bfloat16).float()
    mm_output_block_shape = [1,1,act_block_h*32, weight_block_w*32]
    act_address_map_index = 0
    weight_address_map_index = 0
    weight_block_h = act_block_w
    num_groups = act_address_map[act_address_map_index]
    assert(num_groups == num_blocks_act_h * num_blocks_act_w * num_blocks_weight_w)
    weight_num_groups = act_address_map[weight_address_map_index]
    assert(weight_num_groups == num_groups);
    act_address_map_index += 1
    weight_address_map_index += 1
    for block_act_h in range(num_blocks_act_h):
        # Reset weight (weight) to the starting tile in this column
        for block_weight_w in range(num_blocks_weight_w):
            output_block = torch.zeros(mm_output_block_shape, dtype=torch.bfloat16).float()
            for block_act_w in range(num_blocks_act_w):
                address_map_this_block_size = act_address_map[act_address_map_index]
                act_address_map_index += 1
                weight_address_map_this_block_size = weight_address_map[weight_address_map_index]
                weight_address_map_index += 1
                (mm_act_block, act_address_map_index) = read_conv_act_into_mm_act_block(conv_act, act_address_map_index,
                                                    act_address_map, address_map_this_block_size, act_block_h, act_block_w)
                (mm_weight_block, weight_address_map_index) = read_conv_weight_into_mm_weight_block(mm_weight, weight_address_map_index,
                                                    weight_address_map, weight_address_map_this_block_size, weight_block_h, weight_block_w)
                # Untilize weight block (this CPU reference does matmul on untilized blocks)
                mm_weight_block = untilize(mm_weight_block)
                for out_h_block in range(act_block_h*32):
                    for out_w_block in range(weight_block_w*32):
                        output_block[0][0][out_h_block][out_w_block] += torch.dot(mm_act_block[0,0,out_h_block,:].reshape(-1), mm_weight_block[0,0,:,out_w_block].reshape(-1))
            start_oh = block_act_h * act_block_h * 32
            start_ow = block_weight_w * weight_block_w * 32
            end_oh = start_oh + (act_block_h * 32)
            end_ow = start_ow + (weight_block_w * 32)
            ret[0,0,start_oh:end_oh,start_ow:end_ow] = output_block

    return ret
