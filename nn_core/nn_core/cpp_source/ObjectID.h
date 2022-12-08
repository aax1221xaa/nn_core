#ifndef OBJECT_ID_H
#define OBJECT_ID_H

#include <stdlib.h>



struct Object_ID{
	int nCpy;

	Object_ID *prev;
	Object_ID *next;
};


class Object_Linker{
	Object_ID *head;

public:
	Object_Linker();
	Object_ID* Create();
	void Erase(Object_ID *currObj_ID);
	~Object_Linker();
};


#endif
