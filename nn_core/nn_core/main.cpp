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
		/*
		NN_Manager nn;

		Layer_t x_input = Input({ 784 }, -1, "input");
		
		Layer_t x = NN_Creater(NN_Dense(256, "Dens_1"))(x_input);
		x = NN_Creater(NN_ReLU("ReLU_1"))(x);
		x = NN_Creater(NN_Dense(128, "Dens_2"))(x);
		x = NN_Creater(NN_ReLU("ReLU_2"))(x);
		x = NN_Creater(NN_Dense(64, "Dens_3"))(x);
		x = NN_Creater(NN_ReLU("ReLU_3"))(x);
		x = NN_Creater(NN_Dense(32, "Dens_4"))(x);
		x = NN_Creater(NN_ReLU("ReLU_4"))(x);
		x = NN_Creater(NN_Dense(10, "Dens_5"))(x);
		x = NN_Creater(NN_ReLU("ReLU_5"))(x);

		Model model = Model(x_input, x, "model_1");

		model.summary();
		*/
		MNIST mnis("E:\\data_set\\mnist", 64);
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