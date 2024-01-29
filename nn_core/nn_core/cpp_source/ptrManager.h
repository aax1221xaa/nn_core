#ifndef OBJECT_ID_H
#define OBJECT_ID_H

#include <stdlib.h>


struct ptrRef{
	int ref_cnt;

	ptrRef *prev;
	ptrRef *next;
};


class ptrManager{
	ptrRef *head;

public:
	ptrManager();
	ptrRef* create();
	void erase(ptrRef *node);
	~ptrManager();
};

#endif
