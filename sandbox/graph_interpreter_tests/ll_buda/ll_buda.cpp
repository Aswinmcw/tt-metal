#include <algorithm>
#include <filesystem>
#include <unordered_set>
#include <string>

#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"
#include "ll_buda/host_api.hpp"
#include "llrt/tt_debug_print_server.hpp"

namespace tt {

namespace ll_buda {

static Profiler ll_buda_profiler = Profiler();

void dumpProfilerResults(std::string name_append)
{
    ll_buda_profiler.dumpResults(name_append);
}

Device *CreateDevice(tt::ARCH device_type, int pcie_slot) {
    TT_ASSERT(device_type == tt::ARCH::GRAYSKULL, "Only Grayskull is supported!");
    return new Device(device_type, pcie_slot);
}

bool InitializeDevice(Device *device) { return device->initialize(); }

bool CloseDevice(Device *device) { return device->close(); }

DataMovementKernelArgs *InitializeCompileTimeDataMovementKernelArgs(const CoreCoord &logical_core, const std::vector<uint32_t> &compile_time_args) {
    DataMovementKernelArgs *kernel_args = new DataMovementKernelArgs(logical_core, compile_time_args);
    return kernel_args;
}

DataMovementKernelArgs *InitializeCompileTimeDataMovementKernelArgs(const CoreRange &core_range, const std::vector<uint32_t> &compile_time_args) {
    CoreBlocks core_blocks = {core_range};
    DataMovementKernelArgs *kernel_args = new DataMovementKernelArgs(core_blocks, {compile_time_args});
    return kernel_args;
}

DataMovementKernelArgs *InitializeCompileTimeDataMovementKernelArgs(const CoreBlocks &core_blocks, const std::vector<std::vector<uint32_t>> &compile_time_args_spec) {
    DataMovementKernelArgs *kernel_args = new DataMovementKernelArgs(core_blocks, compile_time_args_spec);
    return kernel_args;
}

ComputeKernelArgs *InitializeCompileTimeComputeKernelArgs(const CoreCoord &logical_core, void *compile_time_args, size_t compile_time_args_size) {
    ComputeKernelArgs *kernel_args = new ComputeKernelArgs(logical_core, compile_time_args, compile_time_args_size);
    return kernel_args;
}

ComputeKernelArgs *InitializeCompileTimeComputeKernelArgs(const CoreRange &core_range, void *compile_time_args, size_t compile_time_args_size) {
    CoreBlocks core_blocks = {core_range};
    ComputeKernelArgs *kernel_args = new ComputeKernelArgs(core_blocks, {compile_time_args}, compile_time_args_size);
    return kernel_args;
}

ComputeKernelArgs *InitializeCompileTimeComputeKernelArgs(const CoreBlocks &core_blocks, const std::vector<void *> &compile_time_args, size_t compile_time_args_size) {
    ComputeKernelArgs *kernel_args = new ComputeKernelArgs(core_blocks, compile_time_args, compile_time_args_size);
    return kernel_args;
}

DataMovementKernel *CreateDataMovementKernel(
    Program *program,
    const std::string &file_name,
    const CoreCoord &core,
    DataMovementKernelArgs *kernel_args,
    DataMovementProcessor processor_type,
    NOC noc) {
    DataMovementKernel *kernel = new DataMovementKernel(file_name, core, kernel_args, processor_type, noc);
    program->add_kernel(kernel);
    return kernel;
}

DataMovementKernel *CreateDataMovementKernel(
    Program *program,
    const std::string &file_name,
    const CoreCoord &core,
    DataMovementProcessor processor_type,
    NOC noc) {
    auto kernel_args = new DataMovementKernelArgs();
    DataMovementKernel *kernel = new DataMovementKernel(file_name, core, kernel_args, processor_type, noc);
    program->add_kernel(kernel);
    return kernel;
}

ComputeKernel *CreateComputeKernel(
    Program *program,
    const std::string &file_name,
    const CoreCoord &core,
    ComputeKernelArgs *kernel_args,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode) {
    ComputeKernel *kernel = new ComputeKernel(
        file_name,
        core,
        kernel_args,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);
    program->add_kernel(kernel);
    return kernel;
}

DataMovementKernel *CreateDataMovementKernel(
    Program *program,
    const std::string &file_name,
    const CoreRange &core_range,
    DataMovementKernelArgs *kernel_args,
    DataMovementProcessor processor_type,
    NOC noc) {
    DataMovementKernel *kernel = new DataMovementKernel(file_name, core_range, kernel_args, processor_type, noc);
    program->add_kernel(kernel);
    return kernel;
}

DataMovementKernel *CreateDataMovementKernel(
    Program *program,
    const std::string &file_name,
    const CoreRange &core_range,
    DataMovementProcessor processor_type,
    NOC noc) {
    auto kernel_args = new DataMovementKernelArgs();
    DataMovementKernel *kernel = new DataMovementKernel(file_name, core_range, kernel_args, processor_type, noc);
    program->add_kernel(kernel);
    return kernel;
}

ComputeKernel *CreateComputeKernel(
    Program *program,
    const std::string &file_name,
    const CoreRange &core_range,
    ComputeKernelArgs *kernel_args,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode) {
    ComputeKernel *kernel = new ComputeKernel(
        file_name,
        core_range,
        kernel_args,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);
    program->add_kernel(kernel);
    return kernel;
}

DramBuffer *CreateDramBuffer(Device *device, int dram_channel, uint32_t size_in_bytes) {
    TT_ASSERT(dram_channel == 0, "Currently only DRAM channel 0 is supported");
    uint32_t buffer_address = device->allocate_buffer(dram_channel, size_in_bytes);
    DramBuffer *buffer = new DramBuffer(dram_channel, size_in_bytes, buffer_address);
    return buffer;
}

DramBuffer *CreateDramBuffer(Device *device, int dram_channel, uint32_t size_in_bytes, uint32_t address) {
    TT_ASSERT(dram_channel >= 0 and dram_channel <= 7, "Valid range for DRAM channle is [0, 7]");
    DramBuffer *buffer = new DramBuffer(device, dram_channel, size_in_bytes, address);
    return buffer;
}

L1Buffer *CreateL1Buffer(Program *program, Device *device, const CoreCoord &core, uint32_t size_in_bytes, uint32_t address) {
    L1Buffer *l1_buffer = new L1Buffer(device, core, size_in_bytes, address);
    program->add_l1_buffer(l1_buffer);
    return l1_buffer;
}

CircularBuffer *CreateCircularBuffer(
    Program *program,
    Device *device,
    uint32_t buffer_index,
    const CoreCoord &core,
    uint32_t number_of_tiles,
    uint32_t size_in_bytes,
    uint32_t l1_address,
    DataFormat data_format) {
    CircularBuffer *circular_buffer =
        new CircularBuffer(device, core, buffer_index, number_of_tiles, size_in_bytes, l1_address, data_format);
    program->add_circular_buffer(circular_buffer);
    return circular_buffer;
}

bool GenerateBinaries(
    Device *device,
    build_kernel_for_riscv_options_t *build_kernel_for_riscv_options,
    const std::string &op_path,
    const KernelGroup &kernel_group,
    const CoreCoord &logical_core) {
    std::string arch_name = tt::get_string_lowercase(device->device_type());

    if (kernel_group.compute != nullptr) {
        generate_binaries_for_triscs(build_kernel_for_riscv_options, op_path, arch_name, true);
    }

    std::thread br_comp([build_kernel_for_riscv_options, op_path, arch_name, kernel_group, logical_core]() {
        generate_binary_for_brisc(
            build_kernel_for_riscv_options,
            op_path,
            arch_name,
            kernel_group.riscv_0->noc(),
            kernel_group.riscv_0->compile_time_args(logical_core));
    });

    std::thread nc_comp([build_kernel_for_riscv_options, op_path, arch_name, kernel_group, logical_core]() {
        generate_binary_for_ncrisc(
            build_kernel_for_riscv_options,
            op_path,
            arch_name,
            kernel_group.riscv_1->noc(),
            kernel_group.riscv_1->compile_time_args(logical_core));
    });
    br_comp.join();
    nc_comp.join();
    return true;
}

bool BlankKernelBinariesExist(const std::string &blank_op_path) {
    bool binaries_exist = std::filesystem::exists(blank_op_path + "/brisc/brisc.hex");
    binaries_exist &= std::filesystem::exists(blank_op_path + "/ncrisc/ncrisc.hex");
    for (int trisc_id = 0; trisc_id <= 2; trisc_id++) {
        std::string trisc_id_str = std::to_string(trisc_id);
        std::string trisc_hex_name = blank_op_path + "/tensix_thread" + trisc_id_str + "/tensix_thread" + trisc_id_str + ".hex";
        binaries_exist &= std::filesystem::exists(trisc_hex_name);
    }
    return binaries_exist;
}

void CompileBlankKernel(Device *device, const std::string &out_dir_path) {
    build_kernel_for_riscv_options_t blank_build_options("blank_op", "blank_op");
    // Crude way to check if blank_op needs to be compiled or not
    if (BlankKernelBinariesExist(out_dir_path + blank_build_options.name)) {
        return;
    }
    struct hlk_args_t {
        std::int32_t dummy;
    };
    void *hlk_args = new hlk_args_t{
        .dummy = 0,
    };
    blank_build_options.set_hlk_args_all_cores(hlk_args, sizeof(hlk_args_t));
    blank_build_options.set_hlk_file_name_all_cores("kernels/compute/blank.cpp");
    blank_build_options.ncrisc_kernel_file_name = "kernels/dataflow/blank.cpp";
    blank_build_options.brisc_kernel_file_name = "kernels/dataflow/blank.cpp";
      // TODO: Hardcoded to false for now
    std::string arch_name = tt::get_string_lowercase(device->device_type());

    generate_binaries_params_t default_params;
    generate_binaries_all_riscs(&blank_build_options, out_dir_path + blank_build_options.name, arch_name, default_params);
}

void SetCircularBufferDataFormat(
    Program *program, const CoreCoord &logical_core, build_kernel_for_riscv_options_t *build_kernel_for_riscv_options, const std::string &op_path) {
    for (auto circular_buffer : program->circular_buffers_on_core(logical_core)) {
        build_kernel_for_riscv_options->set_cb_dataformat_all_cores(
            static_cast<CB>(circular_buffer->buffer_index()), circular_buffer->data_format());
    }
    std::filesystem::create_directories(op_path);
    generate_data_format_descriptors(build_kernel_for_riscv_options, op_path);
}

void ValidateL1Buffers(Device *device, Program *program, const CoreCoord &logical_core) {
    auto l1_buffers_on_core = program->l1_buffers_on_core(logical_core);
    uint32_t total_l1_buffer_size_in_bytes = 0;
    // TODO: Pull this based on device type! - should account for reserved space as well!
    uint32_t max = 1 * 1024 * 1024;
    for (auto l1_buffer : l1_buffers_on_core) {
        total_l1_buffer_size_in_bytes += l1_buffer->size();
        if (total_l1_buffer_size_in_bytes > max) {
            TT_THROW("Size of L1 buffers on " + logical_core.str() + " exceed maximum size of " + std::to_string(max) + " bytes");
        }
    }
}

void ValidateKernelGroup(const KernelGroup &kernel_group, const CoreCoord &logical_core) {
    if (kernel_group.riscv_0 != nullptr and kernel_group.riscv_1 != nullptr) {
        if (kernel_group.riscv_0->noc() == kernel_group.riscv_1->noc()) {
            TT_THROW("Data movement kernels on RISCV_0 and RISCV_1 on core " + logical_core.str() + " cannot use the same NOC, doing so results in a hang!");
        }
    }
}

// Gets all kernels running on a specific core and creates blank kernels for RISCV0 and RISCV1 if the data movement
// processors do not have a kernel
void PopulateKernelGroupWithDataMovementKernels(Program *program, KernelGroup &kernel_group, const CoreCoord &logical_core) {
    // Toggle NOC
    std::function<NOC(DataMovementKernel *, NOC)> get_noc_id = [&](DataMovementKernel *existing_dm_kernel, NOC default_noc) {
        if (existing_dm_kernel != nullptr) {
            uint8_t toggled_noc =  !(existing_dm_kernel->noc());
            return static_cast<NOC>(toggled_noc);
        }
        return default_noc;
    };

    DataMovementKernelArgs *empty_kernel_args = new DataMovementKernelArgs();
    if (kernel_group.riscv_0 == nullptr) {
        NOC riscv_0_noc = get_noc_id(kernel_group.riscv_1, NOC::RISCV_0_default);
        auto riscv_0_kernel = CreateDataMovementKernel(
            program,
            "kernels/dataflow/blank.cpp",
            logical_core,
            empty_kernel_args,
            DataMovementProcessor::RISCV_0,
            riscv_0_noc);
        kernel_group.riscv_0 = riscv_0_kernel;
    }

    if (kernel_group.riscv_1 == nullptr) {
        NOC riscv_1_noc = get_noc_id(kernel_group.riscv_0, NOC::RISCV_1_default);
        auto riscv_1_kernel = CreateDataMovementKernel(
            program,
            "kernels/dataflow/blank.cpp",
            logical_core,
            empty_kernel_args,
            DataMovementProcessor::RISCV_1,
            riscv_1_noc);
        kernel_group.riscv_1 = riscv_1_kernel;
    }
}

std::string GetOpName(const KernelGroup &kernel_group) {
    std::string dummy_op_name;
    std::vector<Kernel *> kernels = {kernel_group.compute, kernel_group.riscv_0, kernel_group.riscv_1};
    for (auto kernel_index = 0; kernel_index < kernels.size(); kernel_index++) {
        auto kernel = kernels.at(kernel_index);
        if (kernel == nullptr) {
            continue;
        }
        dummy_op_name += kernel->name();
        if (kernel_index != kernels.size() - 1) {
            dummy_op_name += "_";
        }
    }
    return dummy_op_name;
}

size_t KernelGroupCompileHash(const KernelGroup &kernel_group, const CoreCoord &logical_core, const std::string &op_name) {
    size_t kg_compile_hash = 0;
    if (kernel_group.compute != nullptr) {
        tt::utils::hash_combine(kg_compile_hash, kernel_group.compute->compile_time_args_hash(logical_core));
        tt::utils::hash_combine(kg_compile_hash, kernel_group.compute->define_args_hash(logical_core));
    }
    tt::utils::hash_combine(kg_compile_hash, kernel_group.riscv_0->compile_time_args_hash(logical_core));
    tt::utils::hash_combine(kg_compile_hash, kernel_group.riscv_1->compile_time_args_hash(logical_core));
    tt::utils::hash_combine(kg_compile_hash, std::hash<std::string>{}(op_name));
    return kg_compile_hash;
}

bool CompileProgram(Device *device, Program *program) {
    bool pass = true;
    ll_buda_profiler.markStart("CompileProgram");

    std::string out_dir_path = tt::utils::get_root_dir() + "/built_kernels/";
    CompileBlankKernel(device, out_dir_path);

    // Compute kernels generate dependencies for data movement kernels
    // Kernels running on a core need to be grouped together for compilation
    // The same group of kernels shouldn't be compiled multiple times
    std::unordered_set<size_t> compiled_hashes;
    auto op_idx = 0;
    for (auto &[logical_core, kernel_group] : program->core_to_kernel_group()) {
        ValidateL1Buffers(device, program, logical_core);

        ValidateKernelGroup(kernel_group, logical_core);

        // Modifies kernel_group to have blank data movement kernels if they are not present
	    PopulateKernelGroupWithDataMovementKernels(program, kernel_group, logical_core);

        auto dummy_op_name = GetOpName(kernel_group);
        build_kernel_for_riscv_options_t dummy_op("dummy_type", dummy_op_name + std::to_string(op_idx++));

        auto kernel_group_hash = KernelGroupCompileHash(kernel_group, logical_core, dummy_op_name);
        std::string op_path = out_dir_path + dummy_op_name + "/" + std::to_string(kernel_group_hash);

        SetCircularBufferDataFormat(program, logical_core, &dummy_op, op_path);

        ConfigureForCompilation(kernel_group.compute, &dummy_op, logical_core, op_path);
        ConfigureForCompilation(kernel_group.riscv_0, &dummy_op, logical_core, op_path);
        ConfigureForCompilation(kernel_group.riscv_1, &dummy_op, logical_core, op_path);

        if (compiled_hashes.find(kernel_group_hash) != compiled_hashes.end()) {
            continue;
        }

        GenerateBinaries(device, &dummy_op, op_path, kernel_group, logical_core);
        compiled_hashes.insert(kernel_group_hash);
    }

    ll_buda_profiler.markStop("CompileProgram");
    return pass;
}

void ConfigureKernelGroup(const KernelGroup &kernel_group, Device *device, const CoreCoord &logical_core) {
    // No need to check if kernel_group.riscv_0 and kernel_group.riscv_1 are null because compilation
    // creates blank data movement kernels for riscs0/1 if there is no kernel on them
    if (kernel_group.compute != nullptr) {
        kernel_group.compute->configure(device, logical_core);
    }
    kernel_group.riscv_1->configure(device, logical_core);
    kernel_group.riscv_0->configure(device, logical_core);
}

bool ConfigureDeviceWithProgram(Device *device, Program *program, bool doStartPrintfServer) {
    bool pass = true;

    ll_buda_profiler.markStart("ConfigureDeviceWithProgram");
    std::vector<CoreCoord> worker_cores;
    auto cluster = device->cluster();
    auto pcie_slot = device->pcie_slot();

    for (auto &[logical_core, kernel_group] : program->core_to_kernel_group()) {
        auto worker_core = device->worker_core_from_logical_core(logical_core);
        worker_cores.push_back(worker_core);

        // CircularBufferConfigVec -- common across all kernels, so written once to the core
        llrt::CircularBufferConfigVec circular_buffer_config_vec = llrt::create_circular_buffer_config_vector();

        // Load firmware into L1 of worker core
        llrt::disable_ncrisc(cluster, pcie_slot, worker_core);
        llrt::disable_triscs(cluster, pcie_slot, worker_core);

        ConfigureKernelGroup(kernel_group, device, logical_core);

        // Initialize registers to INVALID
        constexpr static uint32_t INVALID = 0x4321;
        uint32_t stream_register_address = STREAM_REG_ADDR(0, 24);
        llrt::write_hex_vec_to_core(cluster, pcie_slot, worker_core, {INVALID}, stream_register_address);

        auto cbs_on_core = program->circular_buffers_on_core(logical_core);
        for (auto circular_buffer : cbs_on_core) {
            llrt::set_config_for_circular_buffer(
                circular_buffer_config_vec,
                circular_buffer->buffer_index(),
                circular_buffer->address(),
                circular_buffer->size(),
                circular_buffer->num_tiles());
        }

        llrt::write_circular_buffer_config_vector_to_core(cluster, pcie_slot, worker_core, circular_buffer_config_vec);
    }

    // Setup printf host server
    if (doStartPrintfServer)
    {
        int hart_mask = DPRINT_HART_NC | DPRINT_HART_BR;

        std::string log_name = Profiler::getKernelProfilerLogName();
        tt_start_debug_print_server(cluster, {pcie_slot}, worker_cores, hart_mask, log_name.c_str());
    }

    // Take device out of reset
    const llrt::TensixRiscsOptions riscs_options = llrt::TensixRiscsOptions::ALL_RISCS;
    llrt::internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(
        cluster, pcie_slot, riscs_options, worker_cores);

    ll_buda_profiler.markStop("ConfigureDeviceWithProgram");
    return pass;
}

bool WriteRuntimeArgsToDevice(Device *device, DataMovementKernel *kernel, const CoreCoord &logical_core, const std::vector<uint32_t> &runtime_args) {
    ll_buda_profiler.markStart("WriteRuntimeArgsToDevice");
    bool pass = true;
    kernel->kernel_args()->set_runtime_args(logical_core, runtime_args);
    kernel->write_runtime_args_to_device(device, logical_core);
    ll_buda_profiler.markStop("WriteRuntimeArgsToDevice");
    return pass;
}

bool WriteRuntimeArgsToDevice(Device *device, DataMovementKernel *kernel, const CoreRange &core_range, const std::vector<uint32_t> &runtime_args) {
    ll_buda_profiler.markStart("WriteRuntimeArgsToDeviceCoreRange");
    CoreBlocks core_blocks = {core_range};
    return WriteRuntimeArgsToDevice(device, kernel, core_blocks, {runtime_args});
    ll_buda_profiler.markStop("WriteRuntimeArgsToDeviceCoreRange");
}

bool WriteRuntimeArgsToDevice(Device *device, DataMovementKernel *kernel, const CoreBlocks &core_blocks, const std::vector<std::vector<uint32_t>> &runtime_args_spec) {
    ll_buda_profiler.markStart("WriteRuntimeArgsToDeviceCoreBlocks");
    bool pass = true;
    TT_ASSERT(core_blocks.size() == runtime_args_spec.size());
    for (auto index = 0; index < core_blocks.size(); index++) {
        auto core = core_blocks.at(index);
        auto runtime_args = runtime_args_spec.at(index);
        std::visit(overloaded_core {
            [device, kernel, runtime_args](CoreCoord single_core) {
                WriteRuntimeArgsToDevice(device, kernel, single_core, runtime_args);
            },
            [device, kernel, runtime_args](CoreRange core_range) {
                auto start_core = core_range.first;
                auto end_core = core_range.second;
                for (auto x = start_core.x; x <= end_core.x; x++) {
                    for (auto y = start_core.y; y <= end_core.y; y++) {
                        auto core_in_range = CoreCoord(x, y);
                        WriteRuntimeArgsToDevice(device, kernel, core_in_range, runtime_args);
                    }
                }
            }
        }, core);
    }
    ll_buda_profiler.markStop("WriteRuntimeArgsToDeviceCoreBlocks");
    return pass;
}

void stopPrintfServer()
{
    tt_stop_debug_print_server();
}

bool core_runs_ncrisc(Program *program, const CoreCoord &logical_core) {
    auto kernel_group = program->kernels_on_core(logical_core);
    return kernel_group.riscv_1 != nullptr;
}

bool core_runs_triscs(Program *program, const CoreCoord &logical_core) {
    auto kernel_group = program->kernels_on_core(logical_core);
    return kernel_group.compute != nullptr;
}

llrt::TensixRiscsOptions GetRiscOptionFromCoreConfig(bool core_runs_ncrisc, bool core_runs_triscs) {
    auto risc_option = llrt::TensixRiscsOptions::BRISC_ONLY;
    if (core_runs_ncrisc and not core_runs_triscs) {
        risc_option = llrt::TensixRiscsOptions::BRISC_NCRISC;
    } else if (not core_runs_ncrisc and core_runs_triscs) {
        risc_option = llrt::TensixRiscsOptions::BRISC_TRISCS;
    } else if (core_runs_ncrisc and core_runs_triscs) {
        risc_option = llrt::TensixRiscsOptions::ALL_RISCS;
    }
    return risc_option;
}

bool LaunchKernels(Device *device, Program *program) {

    ll_buda_profiler.markStart("LaunchKernels");
    bool pass = true;

    auto cluster = device->cluster();
    auto pcie_slot = device->pcie_slot();

    // Cores have to be enabled before BRISC reset is de-asserted
    auto logical_cores_used_in_program = program->cores();
    auto worker_cores = device->worker_cores_from_logical_cores(logical_cores_used_in_program);
    llrt::internal_::enable_cores(cluster, pcie_slot, worker_cores);  // BRISC FW waits for this enable to run
    llrt::deassert_brisc_reset_for_all_chips_all_cores(cluster);

    bool riscs_are_done = false;
    while (not riscs_are_done) {
        riscs_are_done = true;
        for (const auto &logical_core : logical_cores_used_in_program) {
            // Check if all the riscs on the core are done
            bool ncrisc_runs = core_runs_ncrisc(program, logical_core);
            bool triscs_run = core_runs_triscs(program, logical_core);
            auto risc_option = GetRiscOptionFromCoreConfig(ncrisc_runs, triscs_run);
            auto worker_core = device->worker_core_from_logical_core(logical_core);
            riscs_are_done &=
                llrt::internal_::check_if_riscs_on_specified_core_done(cluster, pcie_slot, risc_option, worker_core);
        }
    }

    // Reset the mailboxes on each core to enable multiple launches of the same program
    // without needing to re-configure the device
    for (const auto &logical_core : logical_cores_used_in_program) {
        bool ncrisc_runs = core_runs_ncrisc(program, logical_core);
        bool triscs_run = core_runs_triscs(program, logical_core);
        auto risc_option = GetRiscOptionFromCoreConfig(ncrisc_runs, triscs_run);
        auto worker_core = device->worker_core_from_logical_core(logical_core);
        llrt::internal_::setup_riscs_on_specified_core(cluster, pcie_slot, risc_option, worker_core);
    }

    // Reset the device that was running
    cluster->broadcast_remote_tensix_risc_reset(pcie_slot, TENSIX_ASSERT_SOFT_RESET);

    ll_buda_profiler.markStop("LaunchKernels");
    return pass;
}

// Copies data from a host buffer into a buffer within the device DRAM channel
bool WriteToDeviceDRAM(
    DramBuffer *dram_buffer, std::vector<uint32_t> &host_buffer) {
    ll_buda_profiler.markStart("WriteToDeviceDRAM");
    bool pass = true;
    dram_buffer->device()->cluster()->write_dram_vec(
        host_buffer, tt_target_dram{dram_buffer->device()->pcie_slot(), dram_buffer->dram_channel(), 0}, dram_buffer->address());
    ll_buda_profiler.markStop("WriteToDeviceDRAM");
    return pass;
}

// Copy data from a device DRAM channel to a host buffer
bool ReadFromDeviceDRAM(
    DramBuffer *dram_buffer,
    std::vector<uint32_t> &host_buffer) {
    ll_buda_profiler.markStart("ReadFromDeviceDRAM");
    bool pass = true;
    dram_buffer->device()->cluster()->read_dram_vec(
        host_buffer, tt_target_dram{dram_buffer->device()->pcie_slot(), dram_buffer->dram_channel(), 0}, dram_buffer->address(), size);
    ll_buda_profiler.markStop("ReadFromDeviceDRAM");
    return pass;
}

// Copies data from a host buffer into the device DRAM channel
bool WriteToDeviceDRAMChannel(
    Device *device, int dram_channel, std::vector<uint32_t> &host_buffer, uint32_t dram_address) {
    ll_buda_profiler.markStart("WriteToDeviceDRAMChannel");
    bool pass = true;
    device->cluster()->write_dram_vec(
        host_buffer, tt_target_dram{device->pcie_slot(), dram_channel, 0}, dram_address);
    ll_buda_profiler.markStop("WriteToDeviceDRAMChannel");
    return pass;
}

// Copy data from a device DRAM channel to a host buffer
bool ReadFromDeviceDRAMChannel(
    Device *device,
    int dram_channel,
    uint32_t device_address,
    std::vector<uint32_t> &host_buffer,
    uint32_t size) {
    ll_buda_profiler.markStart("ReadFromDeviceDRAMChannel");
    bool pass = true;
    device->cluster()->read_dram_vec(
        host_buffer, tt_target_dram{device->pcie_slot(), dram_channel, 0}, device_address, size);
    ll_buda_profiler.markStop("ReadFromDeviceDRAMChannel");
    return pass;
}

bool WriteToDeviceL1(
    Device *device,
    const CoreCoord &core,
    std::vector<uint32_t> &host_buffer,
    uint32_t buffer_address) {
    ll_buda_profiler.markStart("WriteToDeviceL1");
    bool pass = true;
    auto worker_core = device->worker_core_from_logical_core(core);
    llrt::write_hex_vec_to_core(
        device->cluster(), device->pcie_slot(), worker_core, host_buffer, buffer_address);
    ll_buda_profiler.markStop("WriteToDeviceL1");
    return pass;
}

bool WriteToDeviceL1(
    Device *device,
    const CoreCoord &core,
    op_info_t op_info,
    int op_idx) {
    ll_buda_profiler.markStart("WriteToDeviceL1");
    bool pass = true;
    auto worker_core = device->worker_core_from_logical_core(core);
    llrt::write_graph_interpreter_op_info_to_core(
        device->cluster(), device->pcie_slot(), worker_core, op_info, op_idx);
    ll_buda_profiler.markStop("WriteToDeviceL1");
    return pass;
}

bool ReadFromDeviceL1(
    Device *device, const CoreCoord &core, int device_buffer_addess, std::vector<uint32_t> &host_buffer, int size) {
    ll_buda_profiler.markStart("ReadFromDeviceL1");
    bool pass = true;
    auto worker_core = device->worker_core_from_logical_core(core);
    host_buffer =
        llrt::read_hex_vec_from_core(device->cluster(), device->pcie_slot(), worker_core, device_buffer_addess, size);
    ll_buda_profiler.markStop("ReadFromDeviceL1");
    return pass;
}

}  // namespace ll_buda

}  // namespace tt
