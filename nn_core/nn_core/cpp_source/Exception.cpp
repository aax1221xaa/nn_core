#include "Exception.h"



Exception::Exception(const char* message_, const char* file_, int line_) :
	message(message_),
	file(file_),
	line(line_)
{
}

void Exception::Put() {
	printf("================================================================\n");
	printf("오류 내용: %s\n", message);
	printf("파일명: %s\n", file);
	printf("줄수: %d\n", line);
}


