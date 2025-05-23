#include "cpp_source/nn_core.h"
#include "cpp_source/mnist.h"
#include <exception>
#include <time.h>

#ifdef _DEBUG
#include "vld.h"
#endif



int main() {
	try {

		NN_Manager nn;
		MNIST mnist("E:\\data_set\\mnist");
		Sample<uchar, uchar> train_x = mnist.get_train_samples(64, 1, false);

		Layer_t x_input = nn.input({ 28, 28, 1 }, -1, "input");
		Layer_t x = nn(NN_Div(255.f, "div"))(x_input);
		x = nn(NN_Sub(0.5f, "sub"))(x);
		x = nn(NN_Conv2D(32, { 5, 5 }, { 1, 1 }, "valid", true, "conv_1"))(x);		// {24, 24, 32}
		x = nn(NN_ReLU("relu_1"))(x);
		x = nn(NN_Maxpool2D({ 2, 2 }, { 2, 2 }, "valid", "maxpool_1"))(x);		// {12, 12, 32}
		x = nn(NN_Conv2D(64, { 5, 5 }, { 1, 1 }, "valid", true, "conv_2"))(x);		// {8, 8, 64}
		x = nn(NN_ReLU("relu_2"))(x);
		x = nn(NN_Maxpool2D({ 2, 2 }, { 2, 2 }, "valid", "maxpool_2"))(x);		// {4, 4, 64}
		x = nn(NN_Flatten("flatten"))(x);
		x = nn(NN_Dense(256, "dense_1"))(x);
		x = nn(NN_Sigmoid("sigmoid_1"))(x);
		x = nn(NN_Dense(128, "dense_2"))(x);
		x = nn(NN_Sigmoid("sigmoid_2"))(x);
		x = nn(NN_Dense(10, "dense_3"))(x);
		x = nn(NN_Softmax("softmax"))(x);
		
		Model model(nn, x_input, x, "model_1");
		model.summary();

		model.load_weights("e:/data_set/mnist/mnist.h5");

		clock_t start = clock();
		NN_List<Tensor<nn_type>> result = model.predict(train_x[0]._x, 64, 1);
		clock_t end = clock();

		std::cout << "Elepsed time: " << end - start << "ms" << std::endl;
	}
	catch (NN_Exception& e) {
		e.put();
		cudaDeviceReset();

		return -1;
	}

	cudaDeviceReset();

	return 0;
}