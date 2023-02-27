#include "dtx.hpp"
#include "util_vector_of_ints.hpp"
#include "util.hpp"

bool collapse_transformations(DataTransformations * dtx) {
    bool DEBUG = true;

    if (DEBUG) cout << "\n----- Start Resolving Transoformations -----\n" << endl;


    if (DEBUG) dtx->print();

    // Identify the last node (which we are keeping, and using to drive the resolvemnt of overlaps)
    TransformationNode * consumer_node = dtx->transformations.back();
    if (DEBUG) cout << s(2) << "consumer_node = " << consumer_node->opcode << endl;
    int spaces = 0;
    int c = 0;
    while (dtx->transformations.size() > 2) {

        if (DEBUG) cout << s(4) << "There are more than 2 tx. Starting to resolve." << endl;

        // The node being currently processed - to be deleted. It's always the second from the back
        TransformationNode * producer_node = dtx->transformations[dtx->transformations.size()-2];
        if (DEBUG) cout << s(4) << "producer_node = " << producer_node->opcode << endl;

        // Sweep over groups in the consumer node
        for (int consumer_group_idx = 0; consumer_group_idx < consumer_node->groups.size(); consumer_group_idx++) {
            if (DEBUG) cout << s(6) << "consumer_group_idx = " << consumer_group_idx << endl;

            // Newly formed TensorPairs as a result of the overlaps. One for each consumer group
            vector<TensorPair *> resolved_tensor_pairs;

            // Sweep over groups in the producer node
            for (int producer_group_idx = 0; producer_group_idx < producer_node->groups.size(); producer_group_idx++) {
                if (DEBUG) cout << s(8) << "producer_group_idx = " << producer_group_idx << endl;

                // Sweep over all TensorPairs for a given producer group & consumer group
                for (int consumer_tp_idx=0; consumer_tp_idx<consumer_node->groups[consumer_group_idx]->tensor_pairs.size(); consumer_tp_idx++) {
                    for (int producer_tp_idx=0; producer_tp_idx<producer_node->groups[producer_group_idx]->tensor_pairs.size(); producer_tp_idx++) {      // node1 dst tensor

                        if (DEBUG) cout << s(10) << "producer_tp_idx = " << producer_tp_idx << ",   consumer_tp_idx = " << consumer_tp_idx << endl;

                        Tensor * overlap = calculate_tensor_overlap_in_nd(producer_node->groups[producer_group_idx]->tensor_pairs[producer_tp_idx]->dst_tensor, consumer_node->groups[consumer_group_idx]->tensor_pairs[consumer_tp_idx]->src_tensor);

                         //c++;

                        // if(c >= 4) {
                        //     exit(0);
                        // }
                        if (has_overlap(overlap)) {
                            c++;
                            // Part 1: Calculating the new SRC tensor

                            cout << "Producer src tensor - " << endl;
                            producer_node->groups[producer_group_idx]->tensor_pairs[producer_tp_idx]->src_tensor->print();
                            cout << "Producer dst tensor - " << endl;
                            producer_node->groups[producer_group_idx]->tensor_pairs[producer_tp_idx]->dst_tensor->print();

                            vector<int> producer_offset = vector_subtraction(producer_node->groups[producer_group_idx]->tensor_pairs[producer_tp_idx]->dst_tensor->str,
                                                                            producer_node->groups[producer_group_idx]->tensor_pairs[producer_tp_idx]->src_tensor->str);
                            if (DEBUG) cout << s(12) << "producer_offset = " << v2s(producer_offset) << endl;

                            vector<int> new_src_str = vector_subtraction(overlap->str, producer_offset);
                            vector<int> new_src_end = vector_subtraction(overlap->end, producer_offset);
                            Tensor * new_src = new Tensor(new_src_str, new_src_end);

                            if (DEBUG) cout << s(12) << "offset overlap = " << new_src->get_string() << endl;

                            // Part 2: Calculating the new DST tensor
                            vector<int> consumer_offset = vector_subtraction(consumer_node->groups[consumer_group_idx]->tensor_pairs[consumer_tp_idx]->src_tensor->str,
                                                                            consumer_node->groups[consumer_group_idx]->tensor_pairs[consumer_tp_idx]->dst_tensor->str);
                            if (DEBUG) cout << s(12) << "consumer_offset = " << v2s(consumer_offset) << endl;

                            vector<int> new_dst_str = vector_subtraction(overlap->str, consumer_offset);
                            vector<int> new_dst_end = vector_subtraction(overlap->end, consumer_offset);
                            Tensor * new_dst = new Tensor(new_dst_str, new_dst_end);

                            int new_src_group = producer_node->groups[producer_group_idx]->tensor_pairs[producer_tp_idx]->src_group;

                            // Store results
                            resolved_tensor_pairs.push_back(new TensorPair(new_src, new_src_group, new_dst));
                            cout << s(6) << "new resolved tp  " << resolved_tensor_pairs.back()->get_string() << endl;
                            exit(0);
                        }
                    }
                }
            } // for-loop: producer_groupidx

            // Update all TensorPairs in the consumer node, for this group
            consumer_node->groups[consumer_group_idx]->tensor_pairs = resolved_tensor_pairs;
        }

        dtx->transformations.erase(dtx->transformations.end() - 2);
        if (DEBUG) dtx->print(4);
        //exit(0);

    } // while: transformations > 2

    /*
    Next steps:
        - actually delete resolved nodes, and add the new ones
            - dont need to create a whole new node. we can just create a new vector of tensor pairs, and then replace the old one in the group.
            = all the attributes remain in the group. dont have to manipulate them.
        - add support for groups (used for parallelization & streaming)
        - API for creating nodes.
            - nodes generate their own tensor maps automatically under the hood
                - generic tensor slicing node (for eltwise unary, or just for data movement)
        - actually generate the final addresses for kernel transfers

        - 2 categories: 1) infra generic stuff, and 2) dtx nodes that automatically do things under the hood

    */

    if (DEBUG) cout << "\n----- End Resolving Transoformations -----\n" << endl;

    return true;
}
