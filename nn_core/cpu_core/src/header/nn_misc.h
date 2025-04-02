#pragma once
#include "nn_common.h"
#include "nn_tensor.h"


template <typename _T>
void transpose(const NN_Tensor<_T>& src, const NN_Shape& rank_orders, NN_Tensor<_T>& dst);


template <typename _T>
void transpose(const NN_Tensor<_T>& src, const NN_Shape& rank_orders, NN_Tensor<_T>& dst) {
	const NN_Shape& src_shape = src.get_shape();
	const NN_Shape& dst_shape = dst.get_shape();

	const size_t src_len = src_shape.total_size();
	const size_t dst_len = dst_shape.total_size();
	const size_t order_len = rank_orders.total_size();

	if (src_len != order_len || src_len != dst_len) {
		ErrorExcept(
			"[transpose] src, dst and rank_orders are different size. src: %ld, dst: %ld, order: %ld",
			src_len, dst_len, order_len
		);
	}

	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, order_len),
		[&src, &dst, &src_shape, &rank_orders, &dst_shape](const tbb::blocked_range<size_t>& q) {
			const _T* p_src = src.get_shared_ptr().get();
			_T* p_dst = dst.get_shared_ptr().get();

			for (size_t i = q.begin(); i != q.end(); ++i) {
				tbb::parallel_for(
					tbb::blocked_range<int>(0, src_shape[i]),
					[&src, &dst, &rank_orders, &dst_shape](const tbb::blocked_range<int>& p) {

					}
				);
			}
		},
		tbb::auto_partitioner()
	);
}

void padding_dilation(
	const NN_Tensor<nn_type>& input,
	NN_Tensor<nn_type>& output,
	const int stride_x,
	const int stride_y,
	const int offset_x,
	const int offset_y
);

void conv2d(
	const NN_Tensor<nn_type>& input_tensor,
	const NN_Tensor<nn_type>& kernel_tensor,
	NN_Tensor<nn_type>& output_tensor,
	cuint* indices,
	const int stride_h,
	const int stride_w
);