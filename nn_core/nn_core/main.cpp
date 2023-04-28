#include <iostream>
#include <random>
#include <time.h>
#include <vld.h>

#include "cpp_source/nn_model.h"
#include "cpp_source/nn_layer.h"
#include "cpp_source/mnist.h"

#ifdef FIX_MODE
int main() {
	try {
		MNIST mnist("E:\\data_set\\mnist", 64);
		NN_Manager nn;

		Layer_t x_input = Input({ 784 }, -1, "input");
		
		Layer_t x = NN_Creater(NN_Dens(512, "Dense_7"))(x_input);
		x = NN_Creater(NN_ReLU("ReLU_7"))(x);
		x = NN_Creater(NN_Dens(256, "Dens_1"))(x);
		x = NN_Creater(NN_ReLU("ReLU_1"))(x);
		x = NN_Creater(NN_Dens(128, "Dens_2"))(x);
		x = NN_Creater(NN_ReLU("ReLU_8"))(x);
		x = NN_Creater(NN_Dens(64, "Dens_6"))(x);
		x = NN_Creater(NN_ReLU("ReLU_2"))(x);
		x = NN_Creater(NN_Dens(10, "Dens_5"))(x);
		x = NN_Creater(NN_ReLU("ReLU_5"))(x);

		Model model = Model(x_input, x, "model_1");
		model.summary();

		Tensor<nn_type> x_tensor({ 60000, 784 });

		for (uint i = 0; i < x_tensor._len; ++i) x_tensor._data[i] = mnist.train_x._data[i] / 255.f;
		
		clock_t start = clock();
		std::vector<Tensor<nn_type>> output = model.predict<nn_type>(x_tensor, 128, 60000 / 128);
		clock_t end = clock();

		printf("elapsed time: %ld ms.\n", end - start);
		

		/*
		for (uint i = 0; i < 128; ++i) {
			for (uint j = 0; j < 784; ++j) {
				sample._data[i * 784 + j] = ((float)mnist.train_x._data[i * 784 + j]) / 255.f;
			}
		}
		clock_t start = clock();
		std::vector<Tensor<nn_type>> output = model.predict({ sample });
		clock_t end = clock();

		printf("\n\nelapsed time: %ld ms.\n", end - start);
		*/
	}
	catch (const Exception& e) {
		e.Put();
	}

	return 0;
}

#endif

#ifndef FIX_MODE
int main() {
	
	NN_Manager nn;
	
	Layer_t x_input_1 = Input({ 28, 28, 1 }, 32, "input_1");
	Layer_t x_input_2 = Input({ 28, 28, 1 }, 32, "input_2");
	
	Layer_t x = NN_Creater(NN_Test("test_1_1"))(x_input_1);
	x = NN_Creater(NN_Test("test_1_2"))(x);
	x = NN_Creater(NN_Test("test_1_3"))(x);
	Layer_t branch_1 = NN_Creater(NN_Test("test_1_4"))(x);

	x = NN_Creater(NN_Test("test_2_1"))(x_input_2);
	x = NN_Creater(NN_Test("test_2_2"))(x);
	Layer_t branch_2 = NN_Creater(NN_Test("test_2_3"))(x);
	
	x = NN_Creater(NN_Test("concat"))({ branch_1, branch_2 });
	
	x = NN_Creater(NN_Test("test_3_1"))(x);
	x = NN_Creater(NN_Test("test_3_2"))(x);
	Layer_t branch_3 = NN_Creater(NN_Test("test_3_3"))(x);

	x = NN_Creater(NN_Test("test_7_1"))(branch_3);
	Layer_t y_output_1 = NN_Creater(NN_Test("test_7_2"))(x);

	x = NN_Creater(NN_Test("test_8_1"))(branch_3);
	Layer_t y_output_2 = NN_Creater(NN_Test("test_8_2"))(x);

	Model model({ x_input_1, x_input_2 }, { y_output_1, y_output_2 }, "model_1");
	
	x_input_1 = Input({ 28, 28, 1 }, 32, "input_1_1");
	x_input_2 = Input({ 28, 28, 1 }, 32, "input_1_2");

	x = NN_Creater(NN_Test("test_4_1"))(x_input_1);
	x = NN_Creater(NN_Test("test_4_2"))(x);
	Layer_t feature_1 = NN_Creater(NN_Test("test_4_3"))(x);

	x = NN_Creater(NN_Test("test_5_1"))(x_input_2);
	x = NN_Creater(NN_Test("test_5_2"))(x);
	Layer_t feature_2 = NN_Creater(NN_Test("test_5_3"))(x);

	x = model({ feature_1, feature_2 });

	y_output_1 = NN_Creater(NN_Test("test_6_1"))(x[0]);
	y_output_2 = NN_Creater(NN_Test("test_6_2"))(x[1]);

	Model model_2({ x_input_1, x_input_2 }, { y_output_1, y_output_2 }, "model_2");

	model.summary();
	model_2.summary();

	return 0;
}
#endif