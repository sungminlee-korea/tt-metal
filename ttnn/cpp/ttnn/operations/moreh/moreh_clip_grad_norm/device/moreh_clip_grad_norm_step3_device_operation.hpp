
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/decorators.hpp"

#define DEFINE_PROGRAM_FACTORY3(FactoryName)                                                \
    struct FactoryName {                                                                    \
        struct shared_variables_t {                                                         \
            KernelHandle reader_kernels_id;                                                 \
            KernelHandle writer_kernels_id;                                                 \
            std::size_t num_cores_to_be_used;                                               \
            std::size_t num_cores_y;                                                        \
        };                                                                                  \
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>; \
        static cached_program_t create(                                                     \
            const operation_attributes_t& operation_attributes,                             \
            const tensor_args_t& tensor_args,                                               \
            tensor_return_value_t& output);                                                 \
        static void override_runtime_arguments(                                             \
            cached_program_t& cached_program,                                               \
            const operation_attributes_t& operation_attributes,                             \
            const tensor_args_t& tensor_args,                                               \
            tensor_return_value_t& output);                                                 \
    };

namespace ttnn::operations::moreh::moreh_clip_grad_norm {

struct MorehClipGradNormStep3Operation {
    struct operation_attributes_t {};

    struct tensor_args_t {
        const std::vector<Tensor>& inputs;
        const Tensor& clip_coef_clamped;
    };

    using shape_return_value_t = std::vector<Shape>;
    using tensor_return_value_t = std::vector<Tensor>;

    DEFINE_PROGRAM_FACTORY3(ProgramFactory)

    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_inputs(const operation_attributes_t&, const tensor_args_t&);
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const std::vector<Tensor>& inputs, const Tensor& clip_coef_clamped);
};

}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm

namespace ttnn::prim {
constexpr auto moreh_clip_grad_norm_step3 = ttnn::register_operation<
    "ttnn::prim::moreh_clip_grad_norm_step3",
    ttnn::operations::moreh::moreh_clip_grad_norm::MorehClipGradNormStep3Operation>();
}
