#pragma once
#include <stdio.h>



class Exception{
protected:
	const char* message;
	const char* file;
	int line;

public:
	Exception(const char* message_, const char* file_, int line_);
	void Put();
};


#define ErrorExcept(format, ...) __ErrorException(__FILE__, __LINE__, format, ##__VA_ARGS__)

template <typename... T>
void __ErrorException(const char* file, int line, const char* format, T... args) {
	char buffer[256] = { '\0', };

	sprintf_s(buffer, format, args...);

	throw Exception(buffer, file, line);
}