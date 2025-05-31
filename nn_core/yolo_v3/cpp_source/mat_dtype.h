#pragma once
#include "cuda_common.h"
#include <string>


int get_type(const char& dummy, int channels);
int get_type(const uchar& dummy, int channels);
int get_type(const short& dummy, int channels);
int get_type(const ushort& dummy, int channels);
int get_type(const int& dummy, int channels);
int get_type(const float& dummy, int channels);
int get_type(const double& dummy, int channels);

int what_type(const char& dummy);
int what_type(const uchar& dummy);
int what_type(const short& dummy);
int what_type(const ushort& dummy);
int what_type(const int& dummy);
int what_type(const float& dummy);
int what_type(const double& dummy);

size_t get_type_size(int flag);