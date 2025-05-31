#include "mat_dtype.h"
#include <iostream>
#include <opencv2\opencv.hpp>
#include "Exception.h"


int get_type(const char& dummy, int channels) {
	return CV_8SC(channels);
}

int get_type(const uchar& dummy, int channels) {
	return CV_8UC(channels);
}

int get_type(const short& dummy, int channels) {
	return CV_16SC(channels);
}

int get_type(const ushort& dummy, int channels) {
	return CV_16UC(channels);
}

int get_type(const int& dummy, int channels) {
	return CV_32SC(channels);
}

int get_type(const float& dummy, int channels) {
	return CV_32FC(channels);
}

int get_type(const double& dummy, int channels) {
	return CV_64FC(channels);
}


int what_type(const char& dummy) {
	return CV_8S;
}

int what_type(const uchar& dummy) {
	return CV_8U;
}

int what_type(const short& dummy) {
	return CV_16S;
}

int what_type(const ushort& dummy) {
	return CV_16U;
}

int what_type(const int& dummy) {
	return CV_32S;
}

int what_type(const float& dummy) {
	return CV_32F;
}

int what_type(const double& dummy) {
	return CV_64F;
}


size_t get_type_size(int flag) {
	size_t size = 0;

	switch (flag)
	{
	case CV_8S:
		size = sizeof(char);
		break;
	case CV_8U:
		size = sizeof(uchar);
		break;
	case CV_16S:
		size = sizeof(short);
		break;
	case CV_16U:
		size = sizeof(ushort);
		break;
	case CV_32S:
		size = sizeof(int);
		break;
	case CV_32F:
		size = sizeof(float);
		break;
	case CV_64F:
		size = sizeof(double);
		break;
	default:
		ErrorExcept(
			"This flag is wrong."
		);
		break;
	}

	return size;
}