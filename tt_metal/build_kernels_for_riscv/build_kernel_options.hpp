#pragma once

#include "common/core_coord.h"
#include "kernels/hostdevcommon/kernel_structs.h"
#include "hlk_desc.hpp"

namespace tt
{

class build_kernel_for_riscv_options_t
{
    public:

    // general config
    std::string name;

    // HLK config
    tt::tt_hlk_desc hlk_desc;

    // We can keep for future WH support, otherwise not used in GS
    bool fp32_dest_acc_en;

    // BRISC config
    std::string brisc_kernel_file_name;

    // NCRISC config
    std::string ncrisc_kernel_file_name;

    std::map<std::string, std::string> hlk_defines; // preprocessor defines for HLK
    std::map<std::string, std::string> ncrisc_defines;
    std::map<std::string, std::string> brisc_defines;

    build_kernel_for_riscv_options_t(std::string type, std::string name);

    void set_hlk_file_name_all_cores(std::string file_name) ;
    void set_hlk_math_fidelity_all_cores(MathFidelity math_fidelity) ;
    void set_hlk_math_approx_mode_all_cores(bool approx_mode);
    void set_hlk_args_all_cores(void *args, size_t size) ;

    void set_cb_dataformat_all_cores(CB cb_id, DataFormat data_format);
    // old API name
    void set_hlk_operand_dataformat_all_cores(HlkOperand op_id, DataFormat data_format);
};


} // end namespace tt
