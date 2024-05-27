#include <vld.h>

#include "cpp_source/nn_model.h"
#include "cpp_source/nn_layer.h"
#include "cpp_source/mnist.h"


int main() {
	try {
		MNIST mnist("E:\\data_set\\mnist");
		NN_Manager nn;

		Layer_t x_input = nn.input({ 1, 28, 28 }, -1, "input");

		Layer_t x = nn(NN_Conv2D(32, { 3, 3 }, { 1, 1 }, Pad::VALID, "conv_1"))(x_input);
		x = nn(NN_ReLU("relu_1"))(x);
		x = nn(NN_Maxpool2D({ 2, 2 }, { 2, 2 }, Pad::VALID, "maxpool_1"))(x);
		
		x = nn(NN_Conv2D(64, { 5, 5 }, { 1, 1 }, Pad::VALID, "conv_2"))(x);
		x = nn(NN_ReLU("relu_2"))(x);
		x = nn(NN_Maxpool2D({ 2, 2 }, { 2, 2 }, Pad::VALID, "maxpool_2"))(x);

		x = nn(NN_Flat("flat"))(x);

		x = nn(NN_Dense(512, "dense_1"))(x);
		x = nn(NN_ReLU("relu_3"))(x);
		x = nn(NN_Dense(128, "dense_2"))(x);
		x = nn(NN_ReLU("relu_4"))(x);
		x = nn(NN_Dense(10, "dense_2"))(x);
		x = nn(NN_ReLU("relu_5"))(x);

		Model model(nn, x_input, x, "model_1");

		Sample<uchar, uchar> sample = mnist.get_train_samples(32, 3);

		std::vector<std::vector<Tensor<nn_type>>> _y = model.predict(sample);
		
		std::cout << std::fixed;
		std::cout.precision(6);

		for (std::vector<Tensor<nn_type>>& my : _y) {
			std::cout << my[0];
		}
	}
	catch (Exception& e) {
		e.put();
	}

	cudaDeviceReset();

	return 0;
}