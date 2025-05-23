#pragma once


#include "cuda_common.h"
#include "../cuda_source/convolution.cuh"
#include "../cuda_source/nn_dw_conv.cuh"
#include "../cuda_source/nn_upsample.cuh"
#include "../cuda_source/nn_batch_normalize.cuh"
#include "../cuda_source/nn_concat.cuh"
#include "../cuda_source/special_act.cuh"
#include "../cuda_source/matmul.cuh"
#include "../cuda_source/maxpool.cuh"
#include "../cuda_source/relu.cuh"
#include "../cuda_source/sigmoid.cuh"
#include "../cuda_source/softmax.cuh"
#include "nn_ops.h"
#include "flatten.h"
#include "nn_model.h"
