#include "tt_dnn/op_library/operation.hpp"
#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_numpy/functions.hpp"

#include <queue>

namespace tt {

namespace lazy {

using LazyOperation = std::function<tt::tt_metal::Tensor(const std::vector<tt::tt_metal::Tensor>& input_tensors)>;

struct Graph {
    std::map<tt::tt_metal::UniqueId, std::vector<tt::tt_metal::UniqueId>> predecessors;
    std::map<tt::tt_metal::UniqueId, LazyOperation> operations;
};

std::shared_ptr<Graph> merge_graphs(const Graph& graph_a, const Graph& graph_b) {
    auto predecessors = graph_a.predecessors;
    predecessors.insert(std::begin(graph_b.predecessors), std::end(graph_b.predecessors));
    auto operations = graph_a.operations;
    operations.insert(std::begin(graph_b.operations), std::end(graph_b.operations));
    return std::make_shared<Graph>(Graph{predecessors, operations});
}

struct LazyTensor {
    tt::tt_metal::UniqueId unique_id;
    std::shared_ptr<Graph> graph;
    std::size_t output_index;

    tt::tt_metal::Shape shape;
    tt::tt_metal::DataType dtype;
    tt::tt_metal::Layout layout;
    std::optional<tt::tt_metal::ShardSpec> shard_spec;
};

static tt::tt_metal::UniqueId get_next_unique_id() {
    return tt::tt_metal::UniqueId{tt::tt_metal::GLOBAL_UNIQUE_ID++};
}

static LazyTensor as_lazy(
    const tt::tt_metal::Tensor& tensor
) {
    auto unique_id = get_next_unique_id();
    auto operation = [&](const std::vector<tt::tt_metal::Tensor>& input_tensors){
        return tensor;
    };

    auto graph = std::make_shared<Graph>();
    graph->operations.insert({unique_id, operation});
    graph->predecessors.insert({unique_id, {}});

    return LazyTensor{
        .unique_id=unique_id,
        .graph=graph,
        .output_index=0,
    };
}

static LazyTensor ones(
    const tt::tt_metal::Shape& shape
) {
    auto unique_id = get_next_unique_id();
    auto operation = [&](const std::vector<tt::tt_metal::Tensor>& input_tensors){
        return tt::numpy::ones(shape);
    };

    auto graph = std::make_shared<Graph>();
    graph->operations.insert({unique_id, operation});
    graph->predecessors.insert({unique_id, {}});

    return LazyTensor{
        .unique_id=unique_id,
        .graph=graph,
        .output_index=0,
    };
}

static LazyTensor matmul(
    const LazyTensor& input_tensor_a,
    const LazyTensor& input_tensor_b,
    const tt::tt_metal::MemoryConfig& mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG
) {
    auto unique_id = get_next_unique_id();
    auto operation = [&](const std::vector<tt::tt_metal::Tensor>& input_tensors) {
        return tt::tt_metal::matmul(input_tensors.at(0), input_tensors.at(1), mem_config);
    };

    auto graph = merge_graphs(*input_tensor_a.graph, *input_tensor_b.graph);
    graph->operations.insert({unique_id, operation});
    graph->predecessors.insert({unique_id, {input_tensor_a.unique_id, input_tensor_b.unique_id}});

    return LazyTensor{
        .unique_id=unique_id,
        .graph=graph,
        .output_index=0,
    };
}

class Queue {

    std::queue<LazyTensor> queue;
    std::map<tt::tt_metal::UniqueId, std::vector<tt::tt_metal::Tensor>> tensors;

  public:

    auto size() const {
        return this->queue.size();
    }

    LazyTensor push(const LazyTensor& lazy_tensor) {
        this->queue.push(lazy_tensor);
        return lazy_tensor;
    }

    void finish() {
        while (this->size()) {
            auto lazy_tensor = this->queue.front();
            this->queue.pop();

            auto graph = lazy_tensor.graph;
            auto operation = graph->operations.at(lazy_tensor.unique_id);
            auto predecessors = graph->predecessors.at(lazy_tensor.unique_id);
            auto pred_input_tensors = std::vector<tt::tt_metal::Tensor>{};
            for (auto pred_unique_id : predecessors) {
                auto input_tensors = tensors.at(pred_unique_id);
                pred_input_tensors.push_back(input_tensors.at(0));
            }
            auto output_tensor = operation(pred_input_tensors);
            tensors[lazy_tensor.unique_id] = {output_tensor};
        }
    }

    tt::tt_metal::Tensor get_tensor(const LazyTensor& lazy_tensor) {
        return tensors.at(lazy_tensor.unique_id).at(lazy_tensor.output_index);
    }
};



}  // namespace lazy

}  // namespace tt
