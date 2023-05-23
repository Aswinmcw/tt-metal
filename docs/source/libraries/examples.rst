.. _Example:

Examples of Tensor and TT-DNN Use
*********************************

Run one OP from `TT-DDN` on TT Accelerator device
=================================================

In this code example we use TT Accelerator device to execute ``relu`` op from `TT-DNN` library.
These are the steps:

* create and initialize TT Accelerator device and get handle for host machine
* create random PyTorch tensor, convert it to TT Tensor and send to TT Accelerator device
* execute ``relu`` on TT Accelerator device
* move output TT Tensor to host machine and print it

.. code-block::

    import torch
    from libs import tt_lib as tt_lib

    if __name__ == "__main__":
        # Initialize TT Accelerator device on PCI slot 0
        tt_device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(tt_device)

        # Gat handle to host machine
        host = tt_lib.device.GetHost()

        # Create random PyTorch tensor
        py_tensor = torch.randn((1, 1, 32, 32))

        # Create TT tensor from PyTorch Tensor and send it to TT accelerator device
        tt_tensor = tt_lib.tensor.Tensor(
            py_tensor.reshape(-1).tolist(),
            py_tensor.size(),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
            tt_device
        )

        # Run relu on TT accelerator device
        tt_relu_out = tt_lib.tensor.relu(tt_tensor)

        # Move TT Tensor tt_relu_out from TT accelerator device to host
        tt_output = tt_relu_out.to(host)

        # Print TT Tensor
        tt_output.pretty_print()

        # Close TT accelerator device
        tt_lib.device.CloseDevice(tt_device)


Run `TT-DNN` and PyTorch OPs
============================

In this code example we build on previous example and after ``relu`` also execute ``pow``, ``silu``, and ``exp``.

Since ``pow`` is not supported in `TT-DNN` library, we need to move TT Tensor produced by ``relu`` to host machine,
convert it to PyTorch tensor, execut ``pow`` from PyTorch, and then convert the outpout of ``pow`` back to TT Tensor.

After ``pow``, ``silu`` is executed as a fallback op; this means that the operation will actully execute as a PyTorch operation
on host machine. But since ``silu`` is supported as fallback operation in `TT-DNN` library, we can treat it as any other op from `TT-DNN` library and
supply is with TT Tensor as input.

Lastly, we run ``exp`` on TT Accelerator device (suppling it with output from ``silu`` without any conversion).


.. code-block::

    import torch
    from libs import tt_lib as tt_lib
    from libs.tt_lib.fallback_ops import fallback_ops

    if __name__ == "__main__":
        # Initialize TT Accelerator device on PCI slot 0
        tt_device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(tt_device)

        # Gat handle to host machine
        host = tt_lib.device.GetHost()

        # Create random PyTorch tensor
        py_tensor = torch.randn((1, 1, 32, 32))

        # Create TT tensor from PyTorch Tensor and send it to TT accelerator device
        tt_tensor = tt_lib.tensor.Tensor(
            py_tensor.reshape(-1).tolist(),
            py_tensor.size(),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
            tt_device
        )

        # Run relu on TT accelerator device
        tt_relu_out = tt_lib.tensor.relu(tt_tensor)

        # Move TT Tensor tt_relu_out to host and convert it to PyTorch tensor py_relu_out
        tt_relu_out = tt_relu_out.to(host)
        py_relu_out = torch.Tensor(tt_relu_out.data()).reshape(tt_relu_out.shape())

        # Execute pow using PyTorch (since pow is not available from tt_lib)
        py_pow_out = torch.pow(py_relu_out, 2)

        # Create TT Tensor from py_pow_out and move it to TT accelerator device
        tt_pow_out = tt_lib.tensor.Tensor(
            py_pow_out.reshape(-1).tolist(),
            py_pow_out.size(),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
            tt_device
        )

        # Run silu on TT Tensor tt_pow_out
        # This is a fallback op and it will behave like regular ops on TT accelerator device,
        # even though under the hood this op is executed on host.
        tt_silu_out = fallback_ops.silu(tt_pow_out)

        # Run exp on TT accelerator device
        tt_exp_out = tt_lib.tensor.exp(tt_silu_out)

        # Move TT Tensor output from TT accelerator device to host
        tt_output = tt_exp_out.to(host)

        # Print TT Tensor
        tt_output.pretty_print()

        # Close TT accelerator device
        tt_lib.device.CloseDevice(tt_device)

Tensors with odd size of last dim
=================================

We can't create or move to TT Accelerator device a TT Tensor that is in ROW_MAJOR layout and has odd size of last dimension.
This type of TT Tensor can be created on host machine and can be passed to `TT-DNN` operations.

A `TT-DNN` operation will automatically pad the tensor so that the size of last dimension is even, move it to TT Accelerator device,
execute the operation, move output tensor back to host, and finally unpad the output tensor.

To use this functionality, you must call `tt_lib.device.SetDefaultDevice(tt_device)` to set your TT Accelerator device
as the default device that will be used to execute operations on tensors that are on host machine.

So if you want to use a TT Tensor with odd size of last dimension,
the first example with running one operation on TT Accelerator device
can be modified as follow:

.. code-block::

    import torch
    from libs import tt_lib as tt_lib

    if __name__ == "__main__":
        # Initialize TT Accelerator device on PCI slot 0
        tt_device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(tt_device)

        # Set default TT Accelerator device
        # This device will be used to execute TT Tensors that are not assigned to a device
        tt_lib.device.SetDefaultDevice(tt_device)

        # Gat handle to host machine
        host = tt_lib.device.GetHost()

        # Create random PyTorch tensor
        py_tensor = torch.randn((1, 1, 32, 31))

        # Create TT tensor from PyTorch Tensor and leave it on host device
        tt_tensor = tt_lib.tensor.Tensor(
            py_tensor.reshape(-1).tolist(),
            py_tensor.size(),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

        # Run relu on TT accelerator device
        # The ops will padd tensor as needed and send to TT Accelerator device for execution,
        # then it will return result to host and unpad
        tt_relu_out = tt_lib.tensor.relu(tt_tensor)

        # Move TT Tensor output from TT accelerator device to host
        # Note that in this example this call will not do anything since tt_relu_out is already on host machine
        tt_output = tt_relu_out.to(host)

        # Print TT Tensor
        tt_output.pretty_print()

        # Close TT accelerator device
        tt_lib.device.CloseDevice(tt_device)
