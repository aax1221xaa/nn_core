#include "../nn_core/cpp_source/mnist.h"

#ifdef _DEBUG
#include "vld.h"
#endif



int main() {
	try {
		const int batch = 3;

		MNIST mnist("E:\\data_set\\mnist");
		Sample<uchar, uchar> samples = mnist.get_train_samples(batch, 100);

		int key = -1;
		cv::namedWindow("train");
		for (const DataSet<uchar, uchar>& data : samples) {
			for (int i = 0; i < batch; ++i) {
				cv::Mat image(28, 28, CV_8UC1, (void*)(data._x.get_ptr() + (28 * 28 * i)));

				std::cout << (int)*(data._y.get_ptr() + i) << std::endl;
				cv::imshow("train", image);
				key = cv::waitKey();

				if (key == 27) break;
			}
			if (key == 27) break;
		}

		cv::destroyAllWindows();
	}
	catch (const Exception& e) {
		e.put();
	}

	return 0;
}

