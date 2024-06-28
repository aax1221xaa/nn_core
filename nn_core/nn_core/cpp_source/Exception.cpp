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


NN_Exception::NN_Exception(const std::string& message_, const std::string& file_, int line_) :
	message(message_),
	file(file_),
	line(line_)
{
	NN_Check::set_flag(false);
}

void NN_Exception::put() const {
	std::cout << "================================================================\n";
	std::cout << "Error script: " << message << std::endl;
	std::cout << "File path: " << file << std::endl;
	std::cout << "Line: " << line << std::endl;
}


