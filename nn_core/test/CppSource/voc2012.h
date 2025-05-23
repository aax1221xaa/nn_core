#pragma once
#include "cuda_common.h"
#include "nn_sample.h"



class Voc2012 {
	std::vector<std::string> _labels;

public:
	class Generator : public SampleGenerator<uchar, int> {
	public:
		std::string _image_dir;
		std::vector<std::string> _image_names;

		int _fr_width;
		int _fr_height;

		Generator();
		void generate_sample(const std::vector<int>& indices, Sample<xt, yt>& dst) const;
	};

private:
	Generator _gen;

public:
	Voc2012(const std::string& path, const NN_Shape& img_shape, int batch, const std::string& mode);

	const Generator& get_generator() const;
};