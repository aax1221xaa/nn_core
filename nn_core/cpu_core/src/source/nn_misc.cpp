#include "../header/nn_misc.h"
#include <tbb/tbb.h>

using namespace tbb;


void padding_dilation(
	const NN_Tensor<nn_type>& input,
	NN_Tensor<nn_type>& output,
	const int stride_x,
	const int stride_y,
	const int offset_x,
	const int offset_y
) {
	const NN_Shape in_shape = input.get_shape();
	const NN_Shape out_shape = output.get_shape();

	parallel_for(
		blocked_range3d<uint>(
			0, in_shape[1],
			0, in_shape[2], 
			0, in_shape[3]
		),
		[&](const blocked_range3d<uint>& q) {
			const NN_Shape4D in = in_shape.get_4dims();
			const NN_Shape4D out = out_shape.get_4dims();
			
			const nn_type* in_data = input.get_ptr();
			nn_type* out_data = output.get_ptr();

			for (uint h = q.pages().begin(); h < q.pages().end(); ++h) {
				cuint h_idx = stride_y * h + offset_y;
				cuint out_h = out._c * out._w * h_idx;
				cuint in_h = in._c * in._w * h_idx;
				for (uint w = q.rows().begin(); w < q.rows().end(); ++w) {
					cuint w_idx = stride_x * w + offset_x;
					cuint out_w = out._c * w_idx;
					cuint in_w = in._c * w_idx;
					for (uint c = q.cols().begin(); c < q.cols().end(); ++c) {
						out_data[out_h + out_w + c] = in_data[in_h + in_w + c];
					}
				}
			}
		},
		auto_partitioner()
	);
}

class __Conv2D {
	const nn_type* _in_data;
	const nn_type* _k_data;
	cuint* _indices;

public:
	nn_type _sum_val;

	__Conv2D(
		const nn_type* in_data,
		const nn_type* k_data,
		cuint* indices
	) : _in_data(in_data), _k_data(k_data), _indices(indices), _sum_val(0) {}

	__Conv2D(const __Conv2D& p, split) :
		_in_data(p._in_data), _k_data(p._k_data), _indices(p._indices), _sum_val(0) {}

	void join(const __Conv2D& p) {
		_sum_val += p._sum_val;
	}

	void operator()(const blocked_range<uint>& q) {
		for (uint i = q.begin(); i < q.end(); ++i) {
			_sum_val += _in_data[_indices[i]] * _k_data[i];
		}
	}
};

void conv2d(
	const NN_Tensor<nn_type>& input_tensor,
	const NN_Tensor<nn_type>& kernel_tensor,
	NN_Tensor<nn_type>& output_tensor,
	cuint* indices,
	const int stride_h,
	const int stride_w
) {
	const NN_Shape in_shape = input_tensor.get_shape();
	const NN_Shape k_shape = kernel_tensor.get_shape();
	const NN_Shape out_shape = output_tensor.get_shape();

	const nn_type* in_data = input_tensor.get_ptr();
	const nn_type* k_data = kernel_tensor.get_ptr();
	nn_type* out_data = output_tensor.get_ptr();

	const NN_Shape4D in_dims = in_shape.get_4dims();
	const NN_Shape4D k_dims = k_shape.get_4dims();
	const NN_Shape4D out_dims = out_shape.get_4dims();
	uint* p_indices = new uint[k_dims._n * k_dims._h * k_dims._w];

	for (uint h = 0; h < k_dims._n; ++h) {
		cuint h_idx = h * k_dims._w * k_dims._h;
		cuint h_in_idx = h * in_dims._w * in_dims._c;
		for (uint w = 0; w < k_dims._h; ++w) {
			cuint w_idx = w * k_dims._w;
			cuint w_in_idx = w * in_dims._c;
			for (uint c = 0; c < k_dims._w; ++c) {
				p_indices[h_idx + w_idx + c] = h_in_idx + w_in_idx + c;
			}
		}
	}

	parallel_for(
		blocked_range3d<uint>(
			0, out_dims._n,
			0, out_dims._h,
			0, out_dims._c * out_dims._w
		),
		[&](const blocked_range3d<uint>& q) {
			for (uint n = q.pages().begin(); n < q.pages().end(); ++n) {
				const nn_type* n_in_data = in_data + (n * in_dims._h * in_dims._w * in_dims._c);
				nn_type* n_out_data = out_data + (n * out_dims._h * out_dims._w * out_dims._c);
				for (uint h = q.rows().begin(); h < q.rows().end(); ++h) {
					const nn_type* h_in_data = n_in_data + (h * stride_h * in_dims._w * in_dims._c);
					nn_type* h_out_data = n_out_data + (h * out_dims._w * out_dims._c);
					for (uint wc = q.cols().begin(); wc < q.cols().end(); ++wc) {
						cuint w = wc / out_dims._w;
						cuint c = wc % out_dims._w;
						
						__Conv2D conv(
							h_in_data + (w * in_dims._c + c),
							k_data,
							p_indices
						);

						parallel_reduce(
							blocked_range<uint>(0, k_dims._n * k_dims._h * k_dims._w),
							conv,
							auto_partitioner()
						);
						h_out_data[out_dims._c * w + c] = conv._sum_val;
					}
				}
			}
		},
		auto_partitioner()
	);

	delete p_indices;
}