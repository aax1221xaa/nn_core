#include "Exception.h"

 
/**********************************************/
/*                                            */
/*                  NN_Check                  */
/*                                            */
/**********************************************/

bool NN_Check::_is_valid = true;

void NN_Check::set_flag(bool is_valid) {
	_is_valid = is_valid;
}

const bool& NN_Check::get_flag() {
	return _is_valid;
}


Exception::Exception(const std::string& message_, const std::string& file_, int line_) :
	message(message_),
	file(file_),
	line(line_)
{
	NN_Check::set_flag(false);
}

void Exception::put() const {
	std::cout << "================================================================\n";
	std::cout << "오류 내용: " << message << std::endl;
	std::cout << "파일명: " << file << std::endl;
	std::cout << "줄수: " << line << std::endl;
}


