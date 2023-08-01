#pragma once

#include <optional>
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"

#include "tt_metal/impl/dispatch/command_queue.hpp"

// To be removed at a later time, but need a global
// command queue for the time being.
inline std::unique_ptr<CommandQueue> HACK_CQ;

/** @file */

/** \mainpage tt-metal Internal C++ Documentation
 *
 * Welcome. Please navigate using the Files menu. All APIs are documented
 * under the files listed in the Files menu.
 *
 * If you want to contribute to the documentation and are looking for a good
 * resource for generating Markdown tables, refer to
 * https://www.tablesgenerator.com/markdown_tables.
 * */

namespace tt {

namespace tt_metal {

class Host;
class Device;
class Program;
class Buffer;

// ==================================================
//                  HOST API: host and device
// ==================================================

/**
 * Instantiates a device object.
 *
 * Return value: Device *
 *
 * | Argument       | Description                                                      | Data type | Valid range                                         | required |
 * |----------------|------------------------------------------------------------------|-----------|-----------------------------------------------------|----------|
 * | device_type    | Type of Tenstorrent device to be used                            | ARCH enum | “tt::ARCH::GRAYSKULL”                               | Yes      |
 * | pcie_slot      | The number of the PCIexpress slot in which the device is located | int       | 0 to 7                                              | Yes      |
 * */
Device *CreateDevice(tt::ARCH arch, int pcie_slot);

/**
 * Initializes a device and memory manager, also puts device into reset.
 * Device can be initialized with different memory allocation schemes which dictate how host divides up addressable memory space.
 * There are currently two supported schemes:
 * - MemoryAllocator::BASIC:\n
 *      Creates one DRAM bank per DRAM channel and one L1 bank per Tensix core.
 *      All of DRAM is addressable whereas total addressable L1 space is in range [UNRESERVED_BASE, total L1 size).
 *      Reserved region holds FW, Kernels, CircularBuffer config, Semaphores, mailboxes etc
 * - MemoryAllocator::L1_BANKING:\n
 *      Create one DRAM bank per DRAM channel that is entirely addressable.
 *      L1 banks are structured such that there is one L1 bank per compute with storage core which is addressable in range [UNRESERVED_BASE, total L1 size)
 *      and (total L1 size) / (L1 bank size) num L1 banks per storage only cores. L1 space of storage only cores are entirely addressable since no kernels are placed on these cores.
 *      Dispatch cores do not have any L1 banks.
 *
 * Return value: bool
 *
 * | Argument                    | Description                                 | Type                 | Valid Range         | Required |
 * |-----------------------------|---------------------------------------------|----------------------|---------------------|----------|
 * | device                      | Pointer to device object                    | Device *             |                     | Yes      |
 * | memory_allocator            | Type of memory allocator scheme to use      | MemoryAllocator enum | BASIC or L1_BANKING | No       |
 */
bool InitializeDevice(Device *device, const MemoryAllocator &memory_allocator = MemoryAllocator::L1_BANKING);

/**
 * Resets device and closes device
 *
 * Return value: bool
 *
 * | Argument | Description                | Type     | Valid Range | Required |
 * |----------|----------------------------|----------|-------------|----------|
 * | device   | Pointer to a device object | Device * |             | True     |
 */
bool CloseDevice(Device *device);

/**
 * Starts a debug print server on core {1,1} in physical core grid coordinates.
*/
void StartDebugPrintServer(Device *device);

/**
 * Starts a debug print server on specified cores (in physical core grid coordinates).
 *
 * |      Argument     |                                                      Description                                                      |        Data type        |    Valid range    | required |
 * |:-----------------:|:---------------------------------------------------------------------------------------------------------------------:|:-----------------------:|:-----------------:|----------|
 * | device            | Device pointer                                                                                                        |                         |                   | Yes      |
 * | cores             | Array of x,y pairs with locations of the Tensix cores (physical coordinates)                                          | const CoreCoord &      | {0, 0} -> {9, 11} | Yes      |
*/
void StartDebugPrintServerOnCores(Device *device, const std::vector<std::vector<int>>& cores);

// ==================================================
//                  HOST API: program & kernels
// ==================================================

/**
 * Creates a single core data movement kernel and adds it to the program.
 *
 * Return value: DataMovementKernel *
 *
 * | Argument       | Description                                                                                                  | Data type                | Valid range                                                    | required |
 * |----------------|--------------------------------------------------------------------------------------------------------------|--------------------------|----------------------------------------------------------------|----------|
 * | program        | The program to which this kernel will be added to                                                            | Program &                |                                                                | Yes      |
 * | file_name      | Name of file containing the kernel                                                                           | const std::string        |                                                                | Yes      |
 * | core           | The location of the Tensix core on which the kernel will execute (Logical co-ordinates)                      | const CoreCoord &       | {0, 0} –> {9, 11}                                              | Yes      |
 * | compile_args   | Compile arguments passed to kernel at compile time                                                           | const std::vector<uint32_t> &       |                                                                | Yes      |
 * | processor_type | The target RISC-V processor on which the kernel will execute, on the given Tensix core (1 kernel per RISC-V) | enum                     | DataMovementProcessor::RISCV_0, DataMovementProcessor::RISCV_1 | Yes      |
 * | noc            | The NoC ID on which the kernel will perform data transfers                                                   | enum                     | RISCV_0_default, RISCV_1_default, NOC_0, NOC_1,                | Yes      |
 */

DataMovementKernel *CreateDataMovementKernel(
    Program &program,
    const std::string &file_name,
    const CoreCoord &core,
    const std::vector<uint32_t> &compile_args,
    DataMovementProcessor processor_type,
    NOC noc);

/**
 * Creates a single core data movement kernel with no default arguments and adds it to the program.
 *
 * Return value: DataMovementKernel *
 *
 * | Argument       | Description                                                                                                  | Data type                | Valid range                                                    | required |
 * |----------------|--------------------------------------------------------------------------------------------------------------|--------------------------|----------------------------------------------------------------|----------|
 * | program        | The program to which this kernel will be added to                                                            | Program &                |                                                                | Yes      |
 * | file_name      | Name of file containing the kernel                                                                           | const std::string        |                                                                | Yes      |
 * | core           | A range of the Tensix cores (inclusive) on which the kernel will execute (Logical co-ordinates)              | const CoreCoord &       | {0, 0} –> {9, 11}                                              | Yes      |
 * | processor_type | The target RISC-V processor on which the kernel will execute, on the given Tensix core (1 kernel per RISC-V) | enum                     | DataMovementProcessor::RISCV_0, DataMovementProcessor::RISCV_1 | Yes      |
 * | noc            | The NoC ID on which the kernel will perform data transfers                                                   | enum                     | RISCV_0_default, RISCV_1_default, NOC_0, NOC_1,                | Yes      |
 */
DataMovementKernel *CreateDataMovementKernel(
    Program &program,
    const std::string &file_name,
    const CoreCoord &core,
    DataMovementProcessor processor_type,
    NOC noc);

/**
 * Creates a multi-core data movement kernel and adds it to the program.
 *
 * Return value: DataMovementKernel *
 *
 * | Argument       | Description                                                                                                  | Data type                | Valid range                                                    | required |
 * |----------------|--------------------------------------------------------------------------------------------------------------|--------------------------|----------------------------------------------------------------|----------|
 * | program        | The program to which this kernel will be added to                                                            | Program &                |                                                                | Yes      |
 * | file_name      | Name of file containing the kernel                                                                           | const std::string        |                                                                | Yes      |
 * | core_range     | The range of the Tensix co-ordinates on which the kernel will execute (Logical co-ordinates)                 | const CoreRange &        | Any range encompassing cores within {0 , 0} –> {9, 11}         | Yes      |
 * | compile_args   | Compile arguments passed to kernel at compile time                                                           | const std::vector<uint32_t> &       |                                                                | Yes      |
 * | processor_type | The target RISC-V processor on which the kernel will execute, on the given Tensix core (1 kernel per RISC-V) | enum                     | DataMovementProcessor::RISCV_0, DataMovementProcessor::RISCV_1 | Yes      |
 * | noc            | The NoC ID on which the kernel will perform data transfers                                                   | enum                     | RISCV_0_default, RISCV_1_default, NOC_0, NOC_1,                | Yes      |
 */
DataMovementKernel *CreateDataMovementKernel(
    Program &program,
    const std::string &file_name,
    const CoreRange &core_range,
    const std::vector<uint32_t> &compile_args,
    DataMovementProcessor processor_type,
    NOC noc);

/**
 * Creates a multi-core data movement kernel with no default arguments and adds it to the program.
 *
 * Return value: DataMovementKernel *
 *
 * | Argument       | Description                                                                                                  | Data type                | Valid range                                                    | required |
 * |----------------|--------------------------------------------------------------------------------------------------------------|--------------------------|----------------------------------------------------------------|----------|
 * | program        | The program to which this kernel will be added to                                                            | Program *                |                                                                | Yes      |
 * | file_name      | Name of file containing the kernel                                                                           | const std::string        |                                                                | Yes      |
 * | core_range     | The range of the Tensix co-ordinates on which the kernel will execute (Logical co-ordinates)                 | const CoreRange &        | Any range encompassing cores within {0 , 0} –> {9, 11}         | Yes      |
 * | processor_type | The target RISC-V processor on which the kernel will execute, on the given Tensix core (1 kernel per RISC-V) | enum                     | DataMovementProcessor::RISCV_0, DataMovementProcessor::RISCV_1 | Yes      |
 * | noc            | The NoC ID on which the kernel will perform data transfers                                                   | enum                     | RISCV_0_default, RISCV_1_default, NOC_0, NOC_1,                | Yes      |
 */
DataMovementKernel *CreateDataMovementKernel(
    Program &program,
    const std::string &file_name,
    const CoreRange &core_range,
    DataMovementProcessor processor_type,
    NOC noc);

/**
 * Creates a multi-core data movement kernel and adds it to the program.
 *
 * Return value: DataMovementKernel *
 *
 * | Argument       | Description                                                                                                  | Data type                | Valid range                                                    | required |
 * |----------------|--------------------------------------------------------------------------------------------------------------|--------------------------|----------------------------------------------------------------|----------|
 * | program        | The program to which this kernel will be added to                                                            | Program &                |                                                                | Yes      |
 * | file_name      | Name of file containing the kernel                                                                           | const std::string        |                                                                | Yes      |
 * | core_range_set    | A set of ranges (inclusive) of Tensix co-ordinates on which the kernel will execute (Logical co-ordinates)   | const CoreRangeSet &     | Ranges encompassing cores within {0 , 0} –> {9, 11}            | Yes      |
 * | compile_args   | Compile arguments passed to kernel at compile time                                                           | const std::vector<uint32_t> &       |                                                                | Yes      |
 * | processor_type | The target RISC-V processor on which the kernel will execute, on the given Tensix core (1 kernel per RISC-V) | enum                     | DataMovementProcessor::RISCV_0, DataMovementProcessor::RISCV_1 | Yes      |
 * | noc            | The NoC ID on which the kernel will perform data transfers                                                   | enum                     | RISCV_0_default, RISCV_1_default, NOC_0, NOC_1,                | Yes      |
 */
DataMovementKernel *CreateDataMovementKernel(
    Program &program,
    const std::string &file_name,
    const CoreRangeSet &core_range_set,
    const std::vector<uint32_t> &compile_args,
    DataMovementProcessor processor_type,
    NOC noc);

/**
 * Creates a multi-core data movement kernel with no default arguments and adds it to the program.
 *
 * Return value: DataMovementKernel *
 *
 * | Argument       | Description                                                                                                  | Data type                | Valid range                                                    | required |
 * |----------------|--------------------------------------------------------------------------------------------------------------|--------------------------|----------------------------------------------------------------|----------|
 * | program        | The program to which this kernel will be added to                                                            | Program *                |                                                                | Yes      |
 * | file_name      | Name of file containing the kernel                                                                           | const std::string        |                                                                | Yes      |
 * | core_range_set    | A set of ranges (inclusive) of Tensix co-ordinates on which the kernel will execute (Logical co-ordinates)   | const CoreRangeSet &     | Ranges encompassing cores within {0 , 0} –> {9, 11}            | Yes      |
 * | processor_type | The target RISC-V processor on which the kernel will execute, on the given Tensix core (1 kernel per RISC-V) | enum                     | DataMovementProcessor::RISCV_0, DataMovementProcessor::RISCV_1 | Yes      |
 * | noc            | The NoC ID on which the kernel will perform data transfers                                                   | enum                     | RISCV_0_default, RISCV_1_default, NOC_0, NOC_1,                | Yes      |
 */
DataMovementKernel *CreateDataMovementKernel(
    Program &program,
    const std::string &file_name,
    const CoreRangeSet &core_range_set,
    DataMovementProcessor processor_type,
    NOC noc);

/**
 * Creates a single core compute kernel object, and adds it to the program.
 *
 * Return value: ComputeKernel *
 *
 * |     Argument     |                                       Description                                       |      Data type      |      Valid range      | required |
 * |:----------------:|:---------------------------------------------------------------------------------------:|:-------------------:|:---------------------:|----------|
 * | program          | The program to which this kernel will be added to                                       | Program &           |                       | Yes      |
 * | file_name        | Name of file containing the kernel                                                      | const std::string   |                       | Yes      |
 * | core             | The location of the Tensix core on which the kernel will execute (Logical co-ordinates) | const CoreCoord &  | {0, 0} –> {9, 11}     | Yes      |
 * | compile_args     | Compile arguments passed to kernel at compile time                                                           | const std::vector<uint32_t> &       |                                                                | Yes      |
 * | math_fidelity    | The percision of the matrix compute engine                                              | enum                | MathFidelity::HiFi4   | Yes      |
 * | fp32_dest_acc_en | Specifies the type of accumulation performed in the matrix compute engine.              | bool                | false (for Grayskull) | Yes      |
 * | math_approx_mode | Used by the vector compute engine. (will be depricated)                                 | bool                | true, false           | Yes      |
 */
ComputeKernel *CreateComputeKernel(
    Program &program,
    const std::string &file_name,
    const CoreCoord &core,
    const std::vector<uint32_t> &compile_args,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode);

/**
 * Creates a multi-core compute kernel object, and adds it to the program.
 *
 * Return value: ComputeKernel *
 *
 * | Argument         | Description                                                                                  | Data type           | Valid range                                            | required |
 * |------------------|----------------------------------------------------------------------------------------------|---------------------|--------------------------------------------------------|----------|
 * | program          | The program to which this kernel will be added to                                            | Program &           |                                                        | Yes      |
 * | file_name        | Name of file containing the kernel                                                           | const std::string   |                                                        | Yes      |
 * | core_range       | The range of the Tensix co-ordinates on which the kernel will execute (Logical co-ordinates) | const CoreRange &   | Any range encompassing cores within {0 , 0} –> {9, 11} | Yes      |
 * | compile_args     | Compile arguments passed to kernel at compile time                                                           | const std::vector<uint32_t> &       |                                                                | Yes      |
 * | math_fidelity    | The percision of the matrix compute engine                                                   | enum                | MathFidelity::HiFi4                                    | Yes      |
 * | fp32_dest_acc_en | Specifies the type of accumulation performed in the matrix compute engine.                   | bool                | false (for Grayskull)                                  | Yes      |
 * | math_approx_mode | Used by the vector compute engine. (will be depricated)                                      | bool                | true, false                                            | Yes      |
 */
ComputeKernel *CreateComputeKernel(
    Program &program,
    const std::string &file_name,
    const CoreRange &core_range,
    const std::vector<uint32_t> &compile_args,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode);

/**
 * Creates a multi-core compute kernel object, and adds it to the program.
 *
 * Return value: ComputeKernel *
 *
 * | Argument         | Description                                                                                                  | Data type           | Valid range                                            | required |
 * |------------------|--------------------------------------------------------------------------------------------------------------|---------------------|--------------------------------------------------------|----------|
 * | program          | The program to which this kernel will be added to                                                            | Program &           |                                                        | Yes      |
 * | file_name        | Name of file containing the kernel                                                                           | const std::string   |                                                        | Yes      |
 * | core_range_set   | A set of ranges (inclusive) of Tensix co-ordinates on which the kernel will execute (Logical co-ordinates)   | const CoreRangeSet &     | Ranges encompassing cores within {0 , 0} –> {9, 11}            | Yes      |
 * | compile_args     | Compile arguments passed to kernel at compile time                                                           | const std::vector<uint32_t> &       |                                                                | Yes      |
 * | math_fidelity    | The percision of the matrix compute engine                                                                   | enum                | MathFidelity::HiFi4                                    | Yes      |
 * | fp32_dest_acc_en | Specifies the type of accumulation performed in the matrix compute engine.                                   | bool                | false (for Grayskull)                                  | Yes      |
 * | math_approx_mode | Used by the vector compute engine. (will be depricated)                                                      | bool                | true, false                                            | Yes      |
 */
ComputeKernel *CreateComputeKernel(
    Program &program,
    const std::string &file_name,
    const CoreRangeSet &core_range_set,
    const std::vector<uint32_t> &compile_args,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode);

// ==================================================
//                  HOST API: data format
// ==================================================



// ==================================================
//                  HOST API: buffers
// ==================================================
/**
 * Creates a Circular Buffer (CBs) in L1 memory at specified address and core and adds it to the program. L1 allocator reserves size_in_bytes bytes at manually specified addresses.
 *
 * Return value: CircularBuffer *
 *
 * | Argument      | Description                                                                    | Type               | Valid Range                             | Required |
 * |---------------|--------------------------------------------------------------------------------|--------------------|-----------------------------------------|----------|
 * | program       | The program to which buffer will be added to.                                  | Program &          |                                         | True     |
 * | buffer_index  | The index/ID of the CB.                                                        | uint32_t           | 0 to 32 DOX-TODO: specify more detail here. | True     |
 * | core          | The location of the Tensix core on which the CB will reside (logical co-ordinates) | const CoreCoord & | DOX-TODO: { , } –> { , }                    | True     |
 * | num_tiles     | Total number of tiles to be stored in the CB                                   | uint32_t           | DOX-TODO: range?                            | True     |
 * | size_in_bytes | Size of CB buffer in Bytes                                                     | uint32_t           | 0 to 1 MB (DOX-TODO: in Bytes)              | True     |
 * | data_format   | The format of the data to be stored in the CB                                  | DataFormat enum    | DataFormat::Float16_b                   | True     |
 * | l1_address    | Address at which the CB buffer will reside                                     | optional<uint32_t>           | 200 kB to 1MB (DOX-TODO: in bytes)          | False     |
 */
const CircularBuffer &CreateCircularBuffer(
    Program &program,
    uint32_t buffer_index,
    const CoreCoord &core,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format,
    std::optional<uint32_t> l1_address = std::nullopt);

/**
 * Creates Circular Buffers (CBs) in L1 memory of all cores within core range (inclusive) at specified address and adds it to the program. L1 allocator reserves size_in_bytes bytes at manually specified addresses.
 *
 * Return value: CircularBuffer *
 *
 * | Argument      | Description                                                                    | Type               | Valid Range                             | Required |
 * |---------------|--------------------------------------------------------------------------------|--------------------|-----------------------------------------|----------|
 * | program       | The program to which buffer will be added to.                                  | Program *          |                                         | True     |
 * | buffer_index  | The index/ID of the CB.                                                        | uint32_t           | 0 to 32 DOX-TODO: specify more detail here. | True     |
 * | core_range    | Range of the Tensix co-ordinates where buffer will reside (Logical co-ordinates)  | const CoreRange & (std::pair<CoreCoord, CoreCoord>) | DOX-TODO: { , } –> { , }                    | True     |
 * | num_tiles     | Total number of tiles to be stored in the CB                                   | uint32_t           | DOX-TODO: range?                            | True     |
 * | size_in_bytes | Size of CB buffer in Bytes                                                     | uint32_t           | 0 to 1 MB (DOX-TODO: in Bytes)              | True     |
 * | data_format   | The format of the data to be stored in the CB                                  | DataFormat enum    | DataFormat::Float16_b                   | True     |
 * | l1_address    | Address at which the CB buffer will reside                                     | optional<uint32_t>           | 200 kB to 1MB (DOX-TODO: in bytes)          | False     |
 */
const CircularBuffer &CreateCircularBuffers(
    Program &program,
    uint32_t buffer_index,
    const CoreRange &core_range,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format,
    std::optional<uint32_t> l1_address = std::nullopt);

/**
 * Creates Circular Buffers (CBs) in L1 memory of all cores within set of core ranges (inclusive) at specified address and adds it to the program. L1 allocator reserves size_in_bytes bytes at manually specified addresses.
 *
 * Return value: CircularBuffer *
 *
 * | Argument      | Description                                                                    | Type               | Valid Range                             | Required |
 * |---------------|--------------------------------------------------------------------------------|--------------------|-----------------------------------------|----------|
 * | program       | The program to which buffer will be added to.                                  | Program *          |                                         | True     |
 * | buffer_index  | The index/ID of the CB.                                                        | uint32_t           | 0 to 32 DOX-TODO: specify more detail here. | True     |
 * | core_range_set   | Ranges of the Tensix co-ordinates where buffer will reside (Logical co-ordinates)  | const CoreRangeSet & (std::set<CoreRange>) | DOX-TODO: { , } –> { , }                    | True     |
 * | num_tiles     | Total number of tiles to be stored in the CB                                   | uint32_t           | DOX-TODO: range?                            | True     |
 * | size_in_bytes | Size of CB buffer in Bytes                                                     | uint32_t           | 0 to 1 MB (DOX-TODO: in Bytes)              | True     |
 * | data_format   | The format of the data to be stored in the CB                                  | DataFormat enum    | DataFormat::Float16_b                   | True     |
 * | l1_address    | Address at which the CB buffer will reside                                     | optional<uint32_t>           | 200 kB to 1MB (DOX-TODO: in bytes)          | False     |
 */
const CircularBuffer &CreateCircularBuffers(
    Program &program,
    uint32_t buffer_index,
    const CoreRangeSet &core_range_set,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format,
    std::optional<uint32_t> l1_address = std::nullopt);

/**
 * Creates Circular Buffers (CBs) in L1 memory of all cores within set of core ranges (inclusive) at specified address and adds it to the program. L1 allocator reserves size_in_bytes bytes at manually specified addresses.
 *
 * Return value: CircularBuffer *
 *
 * | Argument      | Description                                                                    | Type               | Valid Range                             | Required |
 * |---------------|--------------------------------------------------------------------------------|--------------------|-----------------------------------------|----------|
 * | program       | The program to which buffer will be added to.                                  | Program *          |                                         | True     |
 * | buffer_indices  | Indices/IDs of the CB.                                                        | const std::set<uint32_t> &           | 0 to 32 DOX-TODO: specify more detail here. | True     |
 * | core_range_set   | Ranges of the Tensix co-ordinates where buffer will reside (Logical co-ordinates)  | const CoreRangeSet & (std::set<CoreRange>) | DOX-TODO: { , } –> { , }                    | True     |
 * | num_tiles     | Total number of tiles to be stored in the CB                                   | uint32_t           | DOX-TODO: range?                            | True     |
 * | size_in_bytes | Size of CB buffer in Bytes                                                     | uint32_t           | 0 to 1 MB (DOX-TODO: in Bytes)              | True     |
 * | data_format   | The format of the data to be stored in the CB                                  | DataFormat enum    | DataFormat::Float16_b                   | True     |
 * | l1_address    | Address at which the CB buffer will reside                                     | optional<uint32_t>           | 200 kB to 1MB (DOX-TODO: in bytes)          | True     |
 */
const CircularBuffer &CreateCircularBuffers(
    Program &program,
    const std::set<uint32_t> &buffer_indices,
    const CoreRangeSet &core_range_set,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format,
    std::optional<uint32_t> l1_address = std::nullopt);

/**
 * Initializes semaphore on all cores within core range (inclusive). Each core can have up to four 32B semaphores.
 *
 * Return value: Semaphore address (uint32_t)
 *
 * | Argument      | Description                                          | Type                                                  | Valid Range                                              | Required |
 * |---------------|------------------------------------------------------|-------------------------------------------------------|----------------------------------------------------------|----------|
 * | program       | The program to which semaphore will be added to      | Program &                                             |                                                          | Yes      |
 * | device        | The device where the semaphore resides               | Device *                                              |                                                          | Yes      |
 * | core_range    | Range of the Tensix co-ordinates using the semaphore | const CoreRange & (std::pair<CoreCoord, CoreCoord>)   | Pair of logical coords where first coord <= second coord | Yes      |
 * | initial_value | Initial value of the semaphore                       | uint32_t                                              |                                                          | Yes      |
 */
uint32_t CreateSemaphore(Program &program, const CoreRange &core_range, uint32_t initial_value);

/**
 * Initializes semaphore on all cores within core range (inclusive). Each core can have up to four 32B semaphores.
 *
 * Return value: Semaphore address (uint32_t)
 *
 * | Argument       | Description                                                 | Type                   | Valid Range                                               | Required |
 * |----------------|-------------------------------------------------------------|------------------------|-----------------------------------------------------------|----------|
 * | program        | The program to which semaphore will be added to             | Program &              |                                                           | Yes      |
 * | device         | The device where the semaphore resides                      | Device *               |                                                           | Yes      |
 * | core_range_set    | Set of Range of the Tensix co-ordinates using the semaphore | const CoreRangeSet &   | Pairs of logical coords where first coord <= second coord | Yes      |
 * | initial_value  | Initial value of the semaphore                              | uint32_t               |                                                           | Yes      |
 */
uint32_t CreateSemaphore(Program &program, const CoreRangeSet &core_range_set, uint32_t initial_value);

/**
* Copies data from a host buffer into the specified buffer
*
* Return value: void
*
* | Argument    | Description                                     | Data type               | Valid range                                      | Required |
* |-------------|-------------------------------------------------|-------------------------|--------------------------------------------------|----------|
* | buffer      | Buffer to send data to                          | const Buffer &          |                                                  | Yes      |
* | host_buffer | Buffer on host to copy data from                | std::vector<uint32_t> & | Host buffer size must match buffer               | Yes      |
*/
void WriteToBuffer(const Buffer &buffer, const std::vector<uint32_t> &host_buffer);

/**
* Copies data from a buffer into a host buffer
*
* Return value: void
*
* | Argument    | Description                                     | Data type               | Valid range                                      | Required |
* |-------------|-------------------------------------------------|-------------------------|--------------------------------------------------|----------|
* | buffer      | Buffer to read data from                        | const Buffer &          |                                                  | Yes      |
* | host_buffer | Buffer on host to copy data into                | std::vector<uint32_t> & |                                                  | Yes      |
*/
void ReadFromBuffer(const Buffer &buffer, std::vector<uint32_t> &host_buffer);

/**
*  Deallocates buffer from device by marking its memory as free.
*
*  Return value: void
*
*  | Argument | Description                          | Type     | Valid Range | Required |
*  |----------|--------------------------------------|----------|-------------|----------|
*  | buffer   | The buffer to deallocate from device | Buffer & |             | Yes      |
*/
void DeallocateBuffer(Buffer &buffer);



// ==================================================
//           COMPILE & EXECUTE KENRNELS
//
// ==================================================

/**
 *  Compiles all kernels within the program, and generates binaries that are written to `$TT_METAL_HOME/built/kernels/<kernel name>/<kernel hash>`
 *  Blank data movement kernel targeting RISCV_0 are placed onto cores that are missing a RISCV_0 kernel because RISCV_0 processor needs to run to enable Compute and RISCV_1 processors
 *
 *  To speed up compilation there is a kernel compilation cache that skips over generating binaries for the previously compiled kernels.
 *  Kernel uniqueness is determined by the kernel hash which is computed based on compile time args, defines, and kernel type specific attributes such as NOC for data movement kernels and math fidelity for compute kernels
 *  TODO: Kernel hash needs to account for device architecture as binaries are not the same across architectures.
 *  On cache hits the kernel is not recompiled if the output binary directory exists, otherwise the kernel is compiled.
 *  This cache is static is enabled for the duration of the running process.
 *  By default the cache does not persistent across runs, but can be enabled by calling EnablePersistentKernelCache(). Setting this will skip compilation when output binary directory exists.
 *
 *  Return value: bool
 *
 * | Argument       | Description                                                      | Type      | Valid Range                                        | Required |
 * |----------------|------------------------------------------------------------------|-----------|----------------------------------------------------|----------|
 * | device         | Which device the program is compiled for                         | Device *  | Must be initialized via tt_metal::InitializeDevice | Yes      |
 * | program        | The program to compile                                           | Program & |                                                    | Yes      |
 */
bool CompileProgram(Device *device, Program &program);

// Configures a given device with a given program.
// - Loads all kernel binaries into L1s of assigned Tensix cores
// - Configures circular buffers (inits regs with buffer data)
// - Takes the device out of reset
bool ConfigureDeviceWithProgram(Device *device, const Program &program);

/**
 * Set runtime args for a kernel that are sent to the core during runtime. This API needs to be called to update the runtime args for the kernel.
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                                                      | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------------------------------------|----------|
 * | kernel       | The kernel that will receive the runtime args                          | Kernel *                      |                                                                  | Yes      |
 * | logical_core | The location of the Tensix core where the runtime args will be written | const CoreCoord &             | Any logical Tensix core coordinate on which the kernel is placed | Yes      |
 * | runtime_args | The runtime args to be written                                         | const std::vector<uint32_t> & |                                                                  | Yes      |
 */
void SetRuntimeArgs(Kernel *kernel, const CoreCoord &logical_core, const std::vector<uint32_t> &runtime_args);

/**
 * Set runtime args for a kernel that are shared amongst a range of cores. Runtime args are sent to cores during runtime. This API needs to be called to update the runtime args for the kernel.
 *
 * Return value: void
 *
 * | Argument     | Description                                                                                            | Type                          | Valid Range                                                             | Required |
 * |--------------|--------------------------------------------------------------------------------------------------------|-------------------------------|-------------------------------------------------------------------------|----------|
 * | kernel       | The kernel that will receive the runtime args                                                          | Kernel *                      |                                                                         | Yes      |
 * | core_range   | The range of the Tensix co-ordinates which receive the runtime args (Logical co-ordinates)             | const CoreRange &             | A range of any logical Tensix core coordinate on which the kernel is placed | Yes      |
 * | runtime_args | The runtime args to be written to the core range                                                       | const std::vector<uint32_t> & |                                                                         | Yes      |
 */
void SetRuntimeArgs(Kernel *kernel, const CoreRange &core_range, const std::vector<uint32_t> &runtime_args);

/**
 * Set runtime args for a kernel that are shared amongst a CoreRangeSet. Runtime args are sent to cores during runtime. This API needs to be called to update the runtime args for the kernel.
 *
 * Return value: void
 *
 * | Argument       | Description                                                                                            | Type                          | Valid Range                        | Required |
 * |----------------|--------------------------------------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | kernel         | The kernel that will receive the runtime args                                                          | Kernel *                      |                                    | Yes      |
 * | core_range_set | Set of ranges of Tensix co-ordinates which receive the runtime args (Logical co-ordinates)             | const CoreRangeSet &          | Ranges of any logical Tensix core coordinate on which the kernel is placed | Yes      |
 * | runtime_args   | The runtime args to be written to the core ranges                                                      | const std::vector<uint32_t> & |                                    | Yes      |
 */
void SetRuntimeArgs(Kernel *kernel, const CoreRangeSet &core_range_set, const std::vector<uint32_t> &runtime_args);

/**
 * Get the runtime args for a kernel.
 *
 * Return value: std::vector<uint32_t>
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | kernel       | The kernel that will receive the runtime args                          | Kernel *                      |                                    | Yes      |
 * | logical_core | The location of the Tensix core where the runtime args will be written | const CoreCoord &             | Any logical Tensix core coordinate | Yes      |
 */
std::vector<uint32_t> GetRuntimeArgs(Kernel *kernel, const CoreCoord &logical_core);

/**
 * Writes runtime args that are saved in the program to device
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | device       | The device to whcih runtime args will be written                       | Device *                      |                                    | Yes      |
 * | program      | The program holding the runtime args                                   | const Program &               |                                    | Yes      |
 */
void WriteRuntimeArgsToDevice(Device *device, const Program &program);

// Launches all kernels on cores specified with kernels in the program.
// All kernels on a given Tensix core must be launched.
bool LaunchKernels(Device *device, const Program &program, bool stagger_start = false);

void Synchronize();

}  // namespace tt_metal

}  // namespace tt


/**
 * Reads a buffer from the device
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | cq           | The command queue object which dispatches the command to the hardware  | CommandQueue &                |                                    | Yes      |
 * | buffer       | The device buffer we are reading from                                  | Buffer &                      |                                    | Yes      |
 * | dst          | The vector where the results that are read will be stored              | vector<u32> &                 |                                    | Yes      |
 * | blocking     | Whether or not this is a blocking operation                            | bool                          |                                    | Yes      |
 */
void EnqueueReadBuffer(CommandQueue& cq, Buffer& buffer, vector<u32>& dst, bool blocking);

/**
 * Writes a buffer to the device
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | cq           | The command queue object which dispatches the command to the hardware  | CommandQueue &                |                                    | Yes      |
 * | buffer       | The device buffer we are writing to                                    | Buffer &                      |                                    | Yes      |
 * | dst          | The vector we are writing to the device                                | vector<u32> &                 |                                    | Yes      |
 * | blocking     | Whether or not this is a blocking operation                            | bool                          |                                    | Yes      |
 */
void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buffer, vector<u32>& src, bool blocking);

/**
 * Writes a program to the device and launches it
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | cq           | The command queue object which dispatches the command to the hardware  | CommandQueue &                |                                    | Yes      |
 * | program      | The program we are writing to the device                               | Program &                     |                                    | Yes      |
 * | blocking     | Whether or not this is a blocking operation                            | bool                          |                                    | Yes      |
 */
void EnqueueProgram(CommandQueue& cq, Program& program, bool blocking);

/**
 * Blocks until all previously dispatched commands on the device have completed
 *
 * Return value: void
 *
 * | Argument     | Description                                                            | Type                          | Valid Range                        | Required |
 * |--------------|------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | cq           | The command queue object which dispatches the command to the hardware  | CommandQueue &                |                                    | Yes      |
 */
void Finish(CommandQueue& cq);
