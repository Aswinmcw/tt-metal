from .. import tensor as ttl_tensor, device as ttl_device, profiler as ttl_profiler
import torch
from functools import wraps
from loguru import logger
from contextlib import contextmanager


# Log only once to not pollute output
def check_log_pytorch_warning(arg):
    if not getattr(
        check_log_pytorch_warning, "_pytorch_warning_logged", False
    ) and torch.is_tensor(arg):
        logger.warning(
            "Pytorch tensor was passed as input to fallback op instead of TT tensor. This is currently supported to improve perf but support for this will be deprecated."
        )
        check_log_pytorch_warning._pytorch_warning_logged = True


@contextmanager
def custom_tensor_print_handler(tensor_cls):
    def custom_tt_tensor_to_str_fn(tensor):
        # We just report that this was a tt tensor and its shape as detailed information is already reported in other columns
        return f"tt_lib.tensor.Tensor({'_'.join(map(str, tensor.shape()))})"

    def custom_pt_tensor_to_str_fn(tensor):
        return f"torch.Tensor({'|'.join(['_'.join(map(str, tensor.shape)), str(tensor.layout), str(tensor.dtype), str(tensor.device)])})"

    # Save original methods
    tensor_str_og = tensor_cls.__str__
    tensor_repr_og = tensor_cls.__repr__
    if tensor_cls == ttl_tensor.Tensor:
        custom_tensor_to_str_fn = custom_tt_tensor_to_str_fn
    elif tensor_cls == torch.Tensor:
        custom_tensor_to_str_fn = custom_pt_tensor_to_str_fn
    else:
        assert False, f"No custom tensor str fn found for class {tensor_cls}"
    # Replace methods
    tensor_cls.__str__ = custom_tensor_to_str_fn
    tensor_cls.__repr__ = custom_tensor_to_str_fn
    try:
        yield None
    finally:
        # Restore methods
        tensor_cls.__str__ = tensor_str_og
        tensor_cls.__repr__ = tensor_repr_og


def convert_tt_tensor_to_pt_tensor(tt_tensor, host, output_format):
    # Update output_format with format of first encountered arg
    if output_format.get("device", None) is None and tt_tensor.storage_type() == ttl_tensor.StorageType.DEVICE:
        output_format["device"] = tt_tensor.device()

    if output_format.get("dtype", None) is None:
        output_format["dtype"] = tt_tensor.dtype()

    if ttl_profiler.get_profiler_flag():
        ttl_profiler.append_input_data(tt_tensor)
    # Convert to PT Tensor
    if tt_tensor.storage_type() == ttl_tensor.StorageType.DEVICE:
        tt_tensor = tt_tensor.to(host)

    if tt_tensor.layout() != ttl_tensor.Layout.ROW_MAJOR:
        tt_tensor = tt_tensor.to(ttl_tensor.Layout.ROW_MAJOR)

    return torch.Tensor(tt_tensor.data()).reshape(tt_tensor.shape())


def convert_pt_tensor_to_tt_tensor(pt_tensor, output_format):
    output_shape = pt_tensor.shape
    if len(output_shape) < 4:
        output_shape = [1] * (4 - len(output_shape)) + output_shape
    tt_tensor = ttl_tensor.Tensor(
        pt_tensor.reshape(-1).tolist(),
        output_shape,
        output_format["dtype"],
        ttl_tensor.Layout.ROW_MAJOR,
    )

    if output_format["layout"] == ttl_tensor.Layout.TILE:
        if (
            tt_tensor.shape()[2] % 32 == 0 and tt_tensor.shape()[3] % 32 == 0
        ):  # Restore tile layout only if legal or else leave as RM
            tt_tensor = tt_tensor.to(ttl_tensor.Layout.TILE)
    else:
        if output_format["layout"] != ttl_tensor.Layout.ROW_MAJOR:
            tt_tensor = tt_tensor.to(output_format["layout"])

    if isinstance(output_format["device"], ttl_device.Device):
        if (
            tt_tensor.layout() == ttl_tensor.Layout.TILE
            or tt_tensor.layout() == ttl_tensor.Layout.ROW_MAJOR
            and tt_tensor.shape()[3] % 2 == 0
            or tt_tensor.layout() == ttl_tensor.Layout.CHANNELS_LAST
            and tt_tensor.shape()[1] % 2 == 0
        ):
            tt_tensor = tt_tensor.to(output_format["device"])
    if ttl_profiler.get_profiler_flag():
        ttl_profiler.append_output_data(tt_tensor)
    return tt_tensor


def convert_tt_tensors_to_pt_tensors(args, host, output_format):
    check_log_pytorch_warning(args)
    if isinstance(args, ttl_tensor.Tensor):
        return convert_tt_tensor_to_pt_tensor(args, host, output_format)
    elif isinstance(args, dict):
        outputs = {}
        for key, value in args.items():
            if isinstance(value, ttl_tensor.Tensor):
                outputs[key] = convert_tt_tensor_to_pt_tensor(
                    value, host, output_format
                )
            elif isinstance(value, (list, tuple, dict)):
                outputs[key] = convert_tt_tensors_to_pt_tensors(
                    value, host, output_format
                )
            else:
                check_log_pytorch_warning(value)
                outputs[key] = value
        return outputs
    elif isinstance(args, (list, tuple, dict)):
        outputs = []
        for arg in args:
            if isinstance(arg, ttl_tensor.Tensor):
                outputs.append(convert_tt_tensor_to_pt_tensor(arg, host, output_format))
            elif isinstance(arg, (list, tuple, dict)):
                outputs.append(
                    convert_tt_tensors_to_pt_tensors(arg, host, output_format)
                )
            else:
                check_log_pytorch_warning(arg)
                outputs.append(arg)
        return outputs
    else:
        return args


def convert_pt_tensors_to_tt_tensors(args, output_format):
    if isinstance(args, torch.Tensor):
        return convert_pt_tensor_to_tt_tensor(args, output_format)
    elif isinstance(args, dict):
        outputs = []
        for key, value in args.items():
            if isinstance(value, torch.Tensor):
                outputs[key] = convert_pt_tensor_to_tt_tensor(value, output_format)
            elif isinstance(value, (list, tuple, dict)):
                outputs[key] = convert_pt_tensors_to_tt_tensors(value, output_format)
            else:
                outputs[key] = value
        return outputs
    elif isinstance(args, (list, tuple, dict)):
        outputs = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                outputs.append(convert_pt_tensor_to_tt_tensor(arg, output_format))
            elif isinstance(arg, (list, tuple, dict)):
                outputs.append(convert_pt_tensors_to_tt_tensors(arg, output_format))
            else:
                outputs.append(arg)
        return outputs
    else:
        return args


def convert_tt_tensors_wrapper(func):
    host = ttl_device.GetHost()

    @wraps(func)
    def wrap(*args, **kwargs):
        output_format = {"layout": ttl_tensor.Layout.TILE}

        if ttl_profiler.get_profiler_flag():
            ttl_profiler.start_profiling("fallback_op")
            # This is to check if this is a function of a class. We add the object id if it is
            if '.' in func.__qualname__:
                obj_id = id(args[0])
                split_name = func.__qualname__.rsplit(".", 1)
                ttl_profiler.set_preferred_name(f"{split_name[0]}_{obj_id}.{split_name[1]}")
            else:
                ttl_profiler.set_preferred_name(func.__qualname__)

            # Override str functions for PT and TT Tensors to format/report desired values
            with custom_tensor_print_handler(
                ttl_tensor.Tensor
            ), custom_tensor_print_handler(torch.Tensor):
                if args:
                    # This if is to skip the 'self' argument of class methods.
                    # Note that this may not work correctly in all cases
                    ttl_profiler.append_meta_data(
                        f"args:({str(args) if '.' not in func.__qualname__ else str(args[1:])})".replace(
                            ",", ";"
                        ).replace(
                            " ", ""
                        )
                    )
                if kwargs:
                    ttl_profiler.append_meta_data(
                        f"kwargs:({str(kwargs)})".replace(",", "|").replace(" ", "")
                    )

        new_args = convert_tt_tensors_to_pt_tensors(args, host, output_format)

        new_kwargs = convert_tt_tensors_to_pt_tensors(kwargs, host, output_format)

        # Set default output format
        if output_format.get("device", None) is None:
            output_format["device"] = ttl_device.GetDefaultDevice()
        if output_format.get("dtype", None) is None:
            output_format["dtype"] = ttl_tensor.DataType.BFLOAT16

        outputs = func(*new_args, **new_kwargs)

        # Convert pt tensors in outputs to tt tensors
        new_outputs = convert_pt_tensors_to_tt_tensors(outputs, output_format)

        if ttl_profiler.get_profiler_flag():
            # Override str functions for PT and TT Tensors to format/report desired values
            with custom_tensor_print_handler(
                ttl_tensor.Tensor
            ), custom_tensor_print_handler(torch.Tensor):
                ttl_profiler.append_meta_data(
                    f"outputs:({str(new_outputs)})".replace(",", "|").replace(" ", "")
                )
            ttl_profiler.stop_profiling("fallback_op")

        return new_outputs

    return wrap
