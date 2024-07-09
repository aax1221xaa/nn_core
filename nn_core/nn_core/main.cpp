#include "cpp_source/nn_core.h"
#include "cpp_source/mnist.h"
#include <exception>

#ifdef _DEBUG
#include "vld.h"
#endif
 

#define SELECT				0


#if (SELECT == 0)

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

		train._x[0] = Tensor<uchar>::expand_dims(train._x[0], 1);
		train._y[0] = Tensor<uchar>::expand_dims(train._y[0], 1);

		model.summary();
		model.load_weights("E:\\data_set\\mnist\\mnist.h5");
		
		NN_List<Tensor<nn_type>> result = model.evaluate(sample);

		
		for (NN_List<Tensor<nn_type>>& m_result : result) {
			for (NN_List<Tensor<nn_type>>& p_result : m_result) {
				std::cout << std::endl << p_result.val();
			}
		}
		
		for (const DataSet<uchar, uchar>& data : sample) {
			const uchar* p_y = data._y[0].get_ptr();
			std::cout << std::endl;
			for (int i = 0; i < 16; ++i) {
				std::cout << (int)p_y[i] << ", ";
			}
			std::cout << std::endl;
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

#elif (SELECT == 1)

int main() {
	try {
		NN_Manager nn;

		Layer_t x_input_1 = nn.input({ 3, 3 }, -1, "input_1", NULL);
		Layer_t x_input_2 = nn.input({ 3, 3 }, -1, "input_2", NULL);
		Layer_t x_input_3 = nn.input({ 3, 3 }, -1, "input_3", NULL);

		Layer_t x1 = nn(NN_Add(1.f, "Add_1"))(x_input_1);
		Layer_t x2 = nn(NN_Add(2.f, "Add_2"))(x_input_2);
		Layer_t x3 = nn(NN_Add(3.f, "Add_3"))(x_input_3);

		Layer_t x4 = nn(NN_Sum("Sum_1"))({ x1, x2 });
		x4 = nn(NN_Add(1.f, "Add_4"))(x4);
		Layer_t x5 = nn(NN_Sum("Sum_2"))({ x4, x3 });

		Model model_1(nn, { x_input_1, x_input_2, x_input_3 }, { x4, x5 }, "model_1");

		Layer_t x_input_4 = nn.input({ 3, 3 }, -1, "input_4", NULL);
		Layer_t x_input_5 = nn.input({ 3, 3 }, -1, "input_5", NULL);
		Layer_t x_input_6 = nn.input({ 3, 3 }, -1, "input_6", NULL);

		Layer_t x6 = model_1({ x_input_4, x_input_5, x_input_6 });
		Layer_t x7 = nn(NN_Add(1.f, "Add_5"))(x6[0]);
		Layer_t x8 = nn(NN_Add(1.f, "Add_6"))(x6[1]);

		Model model_2(nn, { x_input_4, x_input_5, x_input_6 }, { x7, x8 }, "model_2");
		
		Model model_3(nn, { x_input_1, x_input_2 }, x4, "model_3");

		model_1.summary();
		model_2.summary();
		model_3.summary();

		Tensor<nn_type> a({ 1, 3, 3 });
		Tensor<nn_type> b({ 1, 3, 3 });
		Tensor<nn_type> c({ 1, 3, 3 });

		a = b = c = 1;

		NN_List<Tensor<nn_type>> samples({ a, b, c });

		NN_List<Tensor<nn_type>> result = model_1.predict(samples);

		std::cout << std::endl;
		std::cout << result << std::endl;
	}
	catch (const NN_Exception& e) {
		e.put();

		return -1;
	}
	catch (const std::out_of_range& e) {
		std::cout << e.what() << std::endl;

		return -1;
	}
	catch (const std::length_error& e) {
		std::cout << e.what() << std::endl;

		return -1;
	}
	catch (const std::exception& e) {
		std::cout << e.what() << std::endl;

		return -1;
	}

	return 0;
}

#elif (SELECT == 2)

int main() {
	try {
		MNIST mnist("E:\\data_set\\mnist");
		Sample<MNIST::x_type, MNIST::y_type> test = mnist.get_test_samples(16, 1, true);

		cv::namedWindow("IMAGE");
		int key = 0;

		for (const DataSet<MNIST::x_type, MNIST::y_type>& sample : test) {
			const Tensor<MNIST::x_type>& x = sample._x[0].val();
			const Tensor<MNIST::y_type>& y = sample._y[0].val();

			for (int i = 0; i < 16; ++i) {
				Tensor<MNIST::x_type> x_image(x[i].get_shape());
				Tensor<MNIST::y_type> y_label(y[i].get_shape());

				x_image = x[i];
				y_label = y[i];

				const cv::Mat m_image(28, 28, CV_8UC1, (void*)x_image.get_ptr());
				const int m_label = *(y_label.get_ptr());

				cv::imshow("IMAGE", m_image);
				std::cout << m_label << std::endl;

				key = cv::waitKey();

				if (key == 27) break;
			}
			if (key == 27) break;
		}
	}
	catch (const NN_Exception& e) {
		e.put();
	}

	cv::destroyAllWindows();

	return 0;
}

#endif