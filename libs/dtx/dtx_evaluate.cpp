#include "dtx.hpp"
#include "dtx_passes.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"

vector<float> evaluate(vector<float> data, DataTransformations * dtx) {
    assert(dtx->transformations.size() == 2);
    assert(dtx->transformations.back()->groups[0]->transfers.size() > 0);
    // copy transfer addresses into a vector
    std::vector<uint32_t> address_map;

    // Generate address map
    for(auto transfer : dtx->transformations.back()->groups[0]->transfers){
        address_map.push_back(transfer->src_address);
        address_map.push_back(transfer->dst_address);
        address_map.push_back(transfer->size);
        address_map.push_back(transfer->pad);
    }

    auto data_transformed_size = vector_product(dtx->transformations.back()->groups[0]->shape);
    vector<float> data_transformed(data_transformed_size, 0);
    for(uint32_t i = 0; i < address_map.size(); i+=4) {
        auto src_address = address_map[i];
        auto dst_address = address_map[i+1];
        auto transfer_size = address_map[i+2];
        auto pad = address_map[i+3];
        for (uint32_t s = 0; s < transfer_size; s++) {
            assert(dst_address+s < data_transformed.size());
            // if(src_address+s >= data.size()) {
            //     std::cout << "src_address=" << src_address << std::endl;
            //     std::cout << "s=" << s << std::endl;
            //     std::cout << "data size=" << data.size() << std::endl;
            //     std::cout << "pad=" << pad << std::endl;
            //     std::cout << "dst_address=" << dst_address << std::endl;
            //     std::cout << "transfer_size=" << transfer_size << std::endl;
            // }
            if (pad) {
                data_transformed[dst_address+s] = 0;
            }
            else {
                assert(src_address+s < data.size());
                data_transformed[dst_address+s] = data[src_address+s];
            }
        }
    }
    return data_transformed;
}
