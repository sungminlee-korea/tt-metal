
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/decorators.hpp"

#define DEFINE_PROGRAM_FACTORY2(FactoryName)                                                \
    struct FactoryName {                                                                    \
        struct shared_variables_t {                                                         \
            KernelHandle reader_kernels_id;                                                 \
            KernelHandle writer_kernels_id;                                                 \
            KernelHandle compute_kernels_id;                                                \
            CoreCoord single_core;                                                          \
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

struct MorehClipGradNormStep2Operation {
    struct operation_attributes_t {
        float norm_type;
    };

    struct tensor_args_t {
        const Tensor& tmp_pow_sum;
        const Tensor& total_norm;
    };

    using shape_return_value_t = std::vector<Shape>;
    using tensor_return_value_t = std::vector<Tensor>;

    DEFINE_PROGRAM_FACTORY2(ProgramFactory)

    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_inputs(const operation_attributes_t&, const tensor_args_t&);
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& tmp_pow_sum, float norm_type, const Tensor& total_norm);
};

}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm

namespace ttnn::prim {
constexpr auto moreh_clip_grad_norm_step2 = ttnn::register_operation<
    "ttnn::prim::moreh_clip_grad_norm_step2",
    ttnn::operations::moreh::moreh_clip_grad_norm::MorehClipGradNormStep2Operation>();
}
