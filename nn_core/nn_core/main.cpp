#include "cpp_source/nn_core.h"
#include "cpp_source/mnist.h"

#ifdef _DEBUG
#include "vld.h"
#endif
 


int main() {
	std::cout.precision(6);

	try {
		MNIST mnist("E:\\data_set\\mnist");
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


		Sample<uchar, uchar> sample = mnist.get_test_samples(16, 1, false);

		model.summary();
		model.load_weights("E:\\data_set\\mnist\\mnist.h5");
		
		std::vector<std::vector<Tensor<nn_type>>> result = model.evaluate(sample);

		
		for (std::vector<Tensor<nn_type>>& m_result : result) {
			for (Tensor<nn_type>& p_result : m_result) {
				std::cout << std::endl << p_result;
			}
		}
		
		for (const DataSet<uchar, uchar>& data : sample) {
			const uchar* p_y = data._y.get_ptr();
			std::cout << std::endl;
			for (int i = 0; i < 16; ++i) {
				std::cout << (int)p_y[i] << ", ";
			}
			std::cout << std::endl;
		}
		
	}
	catch (NN_Exception& e) {
		e.put();
	}

	cudaDeviceReset();

	return 0;
}