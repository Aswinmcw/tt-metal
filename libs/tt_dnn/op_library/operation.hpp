#pragma once

#include <libs/tensor/tensor.hpp>
#include "tt_dnn/op_library/auto_pad.hpp"
#include "tt_dnn/op_library/types.hpp"

#include <experimental/type_traits>

#include "tt_metal/tools/profiler/op_profiler.hpp"

namespace tt::tt_metal {

namespace operation {

// TODO: move 'NotImplemented' to a library file
class NotImplemented : public std::logic_error
{
public:
    NotImplemented(const std::string& message) : std::logic_error(message) { };
};


template<class T, class... Args>
using hashable_t = decltype(std::declval<T>().compute_program_hash(std::declval<Args>()...));

template<class T>
constexpr bool implements_compute_program_hash() {
    return std::experimental::is_detected<hashable_t, T, const std::vector<std::reference_wrapper<const Tensor>>>{};
}

class Operation {
    struct Interface {
        virtual ~Interface() {}

        virtual ProgramHash compute_program_hash(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const = 0;
        virtual void validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const = 0;
        virtual std::vector<Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const = 0;
        virtual std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const = 0;
        virtual Program create_program(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors, std::vector<Tensor> &output_tensors) const = 0;
        virtual std::string get_op_name() const = 0 ;
    };

    template< typename T >
    struct Implementation : Interface {

        Implementation(const T& t) : object(t) {}

        ProgramHash compute_program_hash(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override {
            if constexpr (implements_compute_program_hash<T>()) {
                return this->object.compute_program_hash(input_tensors);
            } else {
                throw NotImplemented("this operation does not implement compute_program_hash!");
            }
        }

        void validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override {
            return this->object.validate(input_tensors);
        }

        std::vector<Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override {
            return this->object.compute_output_shapes(input_tensors);
        }

        std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override {
            return this->object.create_output_tensors(input_tensors);
        }

        Program create_program(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors, std::vector<Tensor> &output_tensors) const override {
            return this->object.create_program(input_tensors, output_tensors);
        }

        std::string get_op_name() const {
            return typeid(T).name();
        }

      private:
        T object;
    };

    std::shared_ptr<const Interface> implementation_;

  public:
    template <typename T>
    Operation(T&& operation): implementation_(std::make_shared<Implementation<T>>(std::forward<T>(operation))) {}

    ProgramHash compute_program_hash(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
        return this->implementation_->compute_program_hash(input_tensors);
    }

    void validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
        return this->implementation_->validate(input_tensors);
    }

    std::vector<Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
        return this->implementation_->compute_output_shapes(input_tensors);
    }

    std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
        return this->implementation_->create_output_tensors(input_tensors);
    }

    Program create_program(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors, std::vector<Tensor> &output_tensors) const {
        return this->implementation_->create_program(input_tensors, output_tensors);
    }

    std::string get_op_name() const {
        return this->implementation_->get_op_name();
    }


};

}
}
