#include "Exception.h"

 
/**********************************************/
/*                                            */
/*                  NN_Check                  */
/*                                            */
/**********************************************/

bool NN_Check::_is_error = true;

void NN_Check::set() {
	_is_error = true;
}

void NN_Check::clear() {
	_is_error = false;
}

bool NN_Check::get_flag() {
	return _is_error;
}


NN_Exception::NN_Exception(const std::string& message_, const std::string& file_, int line_) :
	message(message_),
	file(file_),
	line(line_)
{
	NN_Check::set();
}

void NN_Exception::put() const {
	std::cout << "================================================================\n";
	std::cout << "Error script: " << message << std::endl;
	std::cout << "File path: " << file << std::endl;
	std::cout << "Line: " << line << std::endl;
}


