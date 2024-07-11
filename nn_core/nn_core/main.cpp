#include "cpp_source/nn_core.h"
#include "cpp_source/mnist.h"
#include <exception>

#ifdef _DEBUG
#include "vld.h"
#endif
 



int main() {
	std::cout.precision(6);

	try {
		NN_Manager nn;

		auto convert_func = [](const void* src, void* dst, const tbb::blocked_range<size_t>& q) {
			for (size_t i = q.begin(); i < q.end(); ++i) {
				((nn_type*)dst)[i] = (nn_type)((const uchar*)src)[i] / 255.f - 0.5f;
			}
		};

		Layer_t x_input = nn.input(NN_Shape({ 1, 28, 28 }), -1, "input", convert_func);

		Layer_t x = nn(NN_Conv2D(32, { 5, 5 }, { 1, 1 }, Pad::VALID, "conv_1"))(x_input);
		x = nn(NN_ReLU("relu_1"))(x);
		x = nn(NN_Maxpool2D({ 2, 2 }, { 2, 2 }, Pad::VALID, "maxpool_1"))(x);
		
		x = nn(NN_Conv2D(64, { 5, 5 }, { 1, 1 }, Pad::VALID, "conv_2"))(x);
		x = nn(NN_ReLU("relu_2"))(x);
		x = nn(NN_Maxpool2D({ 2, 2 }, { 2, 2 }, Pad::VALID, "maxpool_2"))(x);

		x = nn(NN_Flatten("flat"))(x);

		x = nn(NN_Dense(256, "dense_1"))(x);
		x = nn(NN_Sigmoid("sigmoid_3"))(x);
		x = nn(NN_Dense(128, "dense_2"))(x);
		x = nn(NN_Sigmoid("sigmoid_4"))(x);
		x = nn(NN_Dense(10, "dense_3"))(x);
		x = nn(NN_Softmax("softmax"))(x);

		Model model(nn, x_input, x, "model_1");

		MNIST mnist("E:\\data_set\\mnist");
		std::vector<DataSet<uchar, uchar>> samples = mnist.get_samples();
		DataSet<uchar, uchar>& train = samples[0];
		DataSet<uchar, uchar>& test = samples[1];

		Tensor<uchar> mx = Tensor<uchar>::expand_dims(test._x[0], 1);
		//Tensor<uchar> my = Tensor<uchar>::expand_dims(train._y[0], 1);

		std::vector<Tensor<uchar>> _x = { mx };

		model.summary();
		model.load_weights("E:\\data_set\\mnist\\mnist.h5");
		
		NN_List<Tensor<nn_type>> result = model.predict(_x, 16, 1);

		
		for (NN_List<Tensor<nn_type>>& m_result : result) {
			for (NN_List<Tensor<nn_type>>& p_result : m_result) {
				std::cout << std::endl << p_result.val();
			}
		}
		
	}
	catch (NN_Exception& e) {
		e.put();
		cudaDeviceReset();

		return -1;
	}

	cudaDeviceReset();

	return 0;
}