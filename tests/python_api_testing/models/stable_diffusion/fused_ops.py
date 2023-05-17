
from libs import tt_lib as ttl
from libs.tt_lib import tensor
from libs.tt_lib.fallback_ops import fallback_ops



def Linear(in_features: int, out_features: int, weight, bias):
    """
    Returns a function that performs a Linear operation with optional bias.

    ``weight`` must be the weight as a tilized list of values.
    """

    weight = weight
    bias = bias

    def linear_(activation):
        weight_T = tensor.transpose(weight)
        output = fallback_ops.matmul(activation, weight_T)
        # output = tensor.matmul(activation, weight_T)

        if bias is not None:
            output_plus_bias = tensor.bcast(output, bias, tensor.BcastOpMath.ADD, tensor.BcastOpDim.H)
            return output_plus_bias

        return output

    return linear_
