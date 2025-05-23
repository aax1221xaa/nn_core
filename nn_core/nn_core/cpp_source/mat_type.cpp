#include "mat_dtype.h"


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