#include "cpp_source/nn_core.h"
#include "cpp_source/voc2012.h"
#include <opencv2\opencv.hpp>


class MobileNet {
public:
	const NN_Shape _input_size = { 416, 416, 3 };
	const std::vector<std::string> _classes = {
		"person",
		"bird",
		"cat",
		"cow",
		"dog",
		"horse",
		"sheep",
		"aeroplane",
		"bicycle",
		"boat",
		"bus",
		"car",
		"motorbike",
		"train",
		"bottle",
		"chair",
		"diningtable",
		"pottedplant",
		"sofa",
		"tvmonitor"
	};
	const int _anchors[2][3][2] = {
		{
			{
				23,
				27
			},
			{
				37,
				58
			},
			{
				81,
				82
			}
		},
		{
			{
				81,
				82
			},
			{
				135,
				169
			},
			{
				344,
				319
			}
		}
	};
	const int anchor_scale[2] = {
		16, 32
	};

public:
	NN_Manager _nn;
	Layer_t _x_input, _branch_1, _branch_2;
	Model _model;

	MobileNet(NN_Manager& nn, int batch);

	auto conv_bn(const int& n_filters, const NN_Shape& filter_size, const NN_Shape& strides) {
		return [&](Layer_t& x) {
			Layer_t mx = _nn(NN_Conv2D(n_filters, filter_size, strides, "same", false))(x);
			mx = _nn(NN_BatchNormalize())(mx);
			mx = _nn(NN_ReLU())(mx);

			return mx;
		};
	}
};

MobileNet::MobileNet(NN_Manager& nn, int batch) :
	_nn(nn)
{
	_x_input = _nn.input(_input_size, batch, "input");												// input
	Layer_t x = _nn(NN_Div(255.f, "normal"))(_x_input);												// normal

	x = conv_bn(16, { 3, 3 }, { 1, 1 })(x);															// conv2d, relu, batch_normal
	x = _nn(NN_Maxpool2D({ 2, 2 }, { 2, 2 }, "same"))(x);											// max_pool2d
	x = conv_bn(32, { 3, 3 }, { 1, 1 })(x);															// conv2d_1, relu_1, batch_normal_1
	x = _nn(NN_Maxpool2D({ 2, 2 }, { 2, 2 }, "same"))(x);											// max_pool2d_1
	x = conv_bn(64, { 3, 3 }, { 1, 1 })(x);															// conv2d_2, relu_2, batch_normal_2
	x = _nn(NN_Maxpool2D({ 2, 2 }, { 2, 2 }, "same"))(x);											// max_pool2d_2
	x = conv_bn(128, { 3, 3 }, { 1, 1 })(x);														// conv2d_3, relu_3, batch_normal_3
	x = _nn(NN_Maxpool2D({ 2, 2 }, { 2, 2 }, "same"))(x);											// max_pool2d_3
	_branch_1 = conv_bn(256, { 3, 3 }, { 1, 1 })(x);												// conv2d_4, relu_4, batch_normal_4
	x = _nn(NN_Maxpool2D({ 2, 2 }, { 2, 2 }, "same"))(_branch_1);									// max_pool2d_4
	x = conv_bn(512, { 3, 3 }, { 1, 1 })(x);														// conv2d_5, relu_5, batch_normal_5
	x = _nn(NN_Maxpool2D({ 2, 2 }, { 1, 1 }, "same"))(x);											// max_pool2d_5
	x = conv_bn(1024, { 3, 3 }, { 1, 1 })(x);														// conv2d_6, relu_6, batch_normal_6
	_branch_2 = conv_bn(256, { 1, 1 }, { 1, 1 })(x);												// conv2d_7, relu_7, batch_normal_7

	x = conv_bn(512, { 3, 3 }, { 1, 1 })(_branch_2);												// conv2d_8, relu_8, batch_normal_8
	x = _nn(NN_Conv2D(3 * (4 + 1 + _classes.size()), { 1, 1 }, { 1, 1 }, "same", true))(x);			// conv2d_9

	Layer_t y1 = _nn(SpecialAct(_classes.size(), "y1"))(x);											// y1

	x = conv_bn(128, { 1, 1 }, { 1, 1 })(_branch_2);												// conv2d_10, relu_9, batch_normal_9
	x = _nn(NN_UpSample2D({ 2, 2 }))(x);															// up_sample2d
	x = _nn(NN_Concat(3))({ x, _branch_1 });											// concat
	x = conv_bn(256, { 3, 3 }, { 1, 1 })(x);														// conv2d_11 relu_10, batch_normal_10
	x = _nn(NN_Conv2D(3 * (4 + 1 + _classes.size()), { 1, 1 }, { 1, 1 }, "same", true))(x);			// conv2d_12
	Layer_t y2 = _nn(SpecialAct(_classes.size(), "y2"))(x);											// y2

	_model = Model(_nn, _x_input, { y2, y1 });
}



int main() {
	try {
		const std::string img_path = "E:\\data_set\\test_image.jpg";
		NN_Manager nn;
		MobileNet mobile(nn, 1);
		
		cv::Mat mat_image = cv::imread(img_path);
		Tensor<uchar> tensor_img = mat_image;
		std::vector<Tensor<uchar>> x;

		x.push_back(tensor_img.expand_dims());

		Model& model = mobile._model;
		
		model.summary();
		model.load_weights("weight\\yolov3_tiny02.h5");

		NN_List<Tensor<nn_type>> y = model.predict(x, 1, 1);

		Tensor<nn_type> tensor = y[0][0].val();

		std::cout << std::fixed;
		std::cout.precision(8);
		std::cout << tensor[0](0, 5)(0, 5)(0)(0);
		std::cout.unsetf(std::ios::fixed);
	}
	catch (const NN_Exception& e) {
		e.put();
	}

	return 0;
}