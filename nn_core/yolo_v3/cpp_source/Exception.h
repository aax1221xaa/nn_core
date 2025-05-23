#pragma once
#include <iostream>
#include <string>


/**********************************************/
/*                                            */
/*                  NN_Check                  */
/*                                            */
/**********************************************/

class NN_Check {
	static bool _is_error;

public:
	static void set();
	static void clear();
	static bool get_flag();
};


class NN_Exception{
protected:
	const std::string message;
	const std::string file;
	int line;

public:
	NN_Exception(const std::string& message_, const std::string& file_, int line_);
	void put() const;
};


#define ErrorExcept(format, ...) __ErrorException(__FILE__, __LINE__, format, ##__VA_ARGS__)

template <typename... T>
void __ErrorException(const std::string& file, int line, const std::string& format, T... args) {
	char buff[200];

	sprintf_s(buff, format.c_str(), args...);

	throw NN_Exception(buff, file, line);
}
