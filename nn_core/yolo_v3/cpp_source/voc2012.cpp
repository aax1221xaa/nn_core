#include "voc2012.h"
#include <fstream>



void _str_split_lines(const std::string& src, std::vector<std::string>& dst) {
	size_t begin = 0;
	size_t end = src.find('\n');

	while (end != std::string::npos) {
		dst.push_back(src.substr(begin, end - begin));
		begin = end + 1;
		end = src.find('\n', begin);
	}
}

void _read_text_lines(const std::string& path, std::vector<std::string>& lines) {
	std::ifstream ifs(path);

	if (!ifs.is_open()) {
		ErrorExcept(
			"This path can't open. %s",
			path.c_str()
		);
	}

	ifs.seekg(0, std::ios::end);

	const size_t str_len = ifs.tellg();
	std::string str(str_len, '\0');

	ifs.seekg(0, std::ios::beg);
	ifs.read(&str[0], str_len);

	_str_split_lines(str, lines);
}

Voc2012::Generator::Generator() {

}

void Voc2012::Generator::generate_sample(const std::vector<int>& indices, Sample<xt, yt>& dst) const {
	std::vector<std::string> current_path;
	Tensor<uchar> images({ _batch, _fr_height, _fr_width, 3 });

	dst._x.clear();

	for (const int& index : indices) 
		current_path.push_back(_image_dir + "\\" + _image_names[index] + ".jpg");

	int i = 0;
	for (const std::string& path : current_path) {
		cv::Mat image = cv::imread(path);
		cv::Mat out_img = cv::Mat::zeros(_fr_width, _fr_height, image.type());

		const int img_h = image.rows;
		const int img_w = image.cols;
		int new_w = 0;
		int new_h = 0;

		if (img_h < img_w) {
			const float fact = (float)_fr_width / img_w;
			
			new_w = _fr_width;
			new_h = (int)roundf(img_h * fact);
		}
		else {
			const float fact = (float)_fr_height / img_h;

			new_w = (int)roundf(img_w * fact);
			new_h = _fr_height;
		}

		cv::Rect roi;

		roi.x = (_fr_width - new_w) / 2;
		roi.y = (_fr_height - new_h) / 2;
		roi.width = new_w;
		roi.height = new_h;

		cv::resize(image, out_img(roi), cv::Size(new_w, new_h));
		
		images[i++] = out_img;
	}

	dst._x.push_back(images);
}

Voc2012::Voc2012(const std::string& path, const NN_Shape& img_shape, int batch, const std::string& mode) {
	std::string m_path = path;
	std::string image_set_path = m_path + "\\ImageSets\\Main\\";
	std::string image_path = m_path + "\\JPEGImages\\";

	_read_text_lines(image_set_path + "labels.txt", _labels);
	_read_text_lines(image_set_path + mode + ".txt", _gen._image_names);

	const int img_size = (int)_gen._image_names.size();
	const int max_index = img_size / batch;

	_gen._image_dir = image_path;
	_gen.set_params(false, img_size, batch, max_index);
	_gen._fr_width = img_shape[0];
	_gen._fr_height = img_shape[1];
}

const Voc2012::Generator& Voc2012::get_generator() const {
	return _gen;
}