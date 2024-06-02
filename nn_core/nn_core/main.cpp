#include <vld.h>
#include <time.h>

#include "cpp_source/nn_model.h"
#include "cpp_source/nn_layer.h"
#include "cpp_source/mnist.h"


int main() {
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

		x = nn(NN_Flat("flat"))(x);

		x = nn(NN_Dense(256, "dense_1"))(x);
		x = nn(NN_ReLU("relu_3"))(x);
		x = nn(NN_Dense(128, "dense_2"))(x);
		x = nn(NN_ReLU("relu_4"))(x);
		x = nn(NN_Dense(10, "dense_2"))(x);
		x = nn(NN_ReLU("relu_5"))(x);

		Model model(nn, x_input, x, "model_1");


		Sample<uchar, uchar> sample = mnist.get_train_samples(64, 938);

		clock_t start = clock();
		std::vector<std::vector<Tensor<nn_type>>> _y = model.predict(sample);
		clock_t end = clock();
		std::cout << end - start << "ms" << std::endl;
	}
	catch (Exception& e) {
		e.put();
	}

	cudaDeviceReset();

	return 0;
}