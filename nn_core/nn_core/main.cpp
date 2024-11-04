#include "cpp_source/nn_core.h"
#include "cpp_source/mnist.h"
#include <exception>

#ifdef _DEBUG
#include "vld.h"
#endif
 



int main() {
	try {
		NN_Manager nn;

		auto convert_func = [](const void* src, void* dst, const tbb::blocked_range<size_t>& q) {
			for (size_t i = q.begin(); i < q.end(); ++i) {
				((nn_type*)dst)[i] = (nn_type)((const uchar*)src)[i] / 255.f - 0.5f;
			}
		};

		Layer_t x_input = nn.input({ 28, 28, 1 }, -1, "input_1", convert_func);

		Layer_t x = nn(NN_Conv2D(32, { 5, 5 }, { 1, 1 }, "valid", "conv_1"))(x_input);
		x = nn(NN_ReLU("relu_1"))(x);
		x = nn(NN_Maxpool2D({ 2, 2 }, { 2, 2 }, "valid", "maxpool_1"))(x);
		x = nn(NN_Conv2D(64, { 5, 5 }, { 1, 1 }, "valid", "conv_2"))(x);
		x = nn(NN_ReLU("relu_2"))(x);
		x = nn(NN_Maxpool2D({ 2, 2 }, { 2, 2 }, "valid", "maxpool_2"))(x);

		x = nn(NN_Flatten("flat"))(x);

		x = nn(NN_Dense(512, "dense_1"))(x);
		x = nn(NN_Sigmoid("sigmoid_1"))(x);
		x = nn(NN_Dense(256, "dense_2"))(x);
		x = nn(NN_Sigmoid("sigmoid_3"))(x);
		x = nn(NN_Dense(10, "dense_3"))(x);
		Layer_t y = nn(NN_Softmax("softmax"))(x);

		Model model(nn, x_input, y, "model_1");

		model.stand_by(SGD(0.01, 0.1), { NN_Loss("loss") });

		//MNIST mnist("E:\\data_set\\mnist");
		//std::vector<DataSet<uchar, uchar>> samples = mnist.get_samples();
		//DataSet<uchar, uchar>& train = samples[0];
		//DataSet<uchar, uchar>& test = samples[1];

		//Tensor<uchar> mx = Tensor<uchar>::expand_dims(test._x[0], 1);
		//Tensor<uchar> my = Tensor<uchar>::expand_dims(train._y[0], 1);

		//std::vector<Tensor<uchar>> _x = { mx };

		model.summary();
		std::cout << "==============================================" << std::endl;
	}
	catch (NN_Exception& e) {
		e.put();
		cudaDeviceReset();

		return -1;
	}

	cudaDeviceReset();

	return 0;
}