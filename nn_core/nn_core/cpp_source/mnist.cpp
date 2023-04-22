#include "mnist.h"


MNIST::MNIST(const char* dir, int batch) {
	try {
		char train_image[128] = { '\0', };
		char train_label[128] = { '\0', };
		char test_image[128] = { '\0', };
		char test_label[128] = { '\0', };

		strcat_s(train_image, dir);
		strcat_s(train_image, "\\train-images.idx3-ubyte");

		strcat_s(train_label, dir);
		strcat_s(train_label, "\\train-labels.idx1-ubyte");

		strcat_s(test_image, dir);
		strcat_s(test_image, "\\t10k-images.idx3-ubyte");

		strcat_s(test_label, dir);
		strcat_s(test_label, "\\t10k-labels.idx1-ubyte");

		load_file(train_image, train_label, train_x, train_y);
		load_file(test_image, test_label, test_x, test_y);

	}
	catch (const Exception& e) {
		e.Put();
	}
}

void MNIST::load_file(const char* image_file, const char* label_file, Tensor<uchar>& image, Tensor<uchar>& truth) {
	FILE* image_fp = NULL;
	FILE* label_fp = NULL;

	try {
		errno_t err = fopen_s(&image_fp, image_file, "rb");

		if (err < 0) {
			ErrorExcept(
				"[MNIST::MNIST] can't read file: %s",
				image_file
			);
		}

		err = fopen_s(&label_fp, label_file, "rb");

		if (err < 0) {
			ErrorExcept(
				"[MNIST::MNIST] can't read file: %s",
				label_file
			);
		}

		union LittleEndian {
			uchar buff8[4];
			int buff32;
		}image_param[4], label_param[2];

		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				fread_s(&image_param[i].buff8[3 - j], sizeof(uchar), sizeof(uchar), 1, image_fp);
			}
		}
		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < 4; ++j) {
				fread_s(&label_param[i].buff8[3 - j], sizeof(uchar), sizeof(uchar), 1, image_fp);
			}
		}

		const int sample_cnt = image_param[1].buff32;
		const int width = image_param[2].buff32;
		const int height = image_param[3].buff32;

		if (sample_cnt == label_param[1].buff32) {
			ErrorExcept("[MNIST::MNIST()] mismatched image and label amounts. image: %d, label: %d", sample_cnt, label_param[1].buff32);
		}

		nn_shape shape({ sample_cnt, height, width, 1 });
		image.set(shape);
		truth.set({ sample_cnt, 1 });

		fread_s(image._data, sizeof(uchar) * image._len, sizeof(uchar), image._len, image_fp);
		fread_s(truth._data, sizeof(uchar) * truth._len, sizeof(uchar), truth._len, label_fp);

		fclose(image_fp);
		fclose(label_fp);

		std::cout << "======================================" << std::endl;
		std::cout << "sample amounts: " << sample_cnt << std::endl;
		std::cout << "image height: " << height << std::endl;
		std::cout << "image width: " << width << std::endl;
	}
	catch (const Exception& e) {
		fclose(image_fp);
		fclose(label_fp);

		throw e;
	}
}