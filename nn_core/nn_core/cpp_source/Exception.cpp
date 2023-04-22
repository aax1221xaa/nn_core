#include "Exception.h"



Exception::Exception(const char* message_, const char* file_, int line_) :
	message(message_),
	file(file_),
	line(line_)
{
}

void Exception::Put() const {
	printf("================================================================\n");
	printf("���� ����: %s\n", message);
	printf("���ϸ�: %s\n", file);
	printf("�ټ�: %d\n", line);
}


