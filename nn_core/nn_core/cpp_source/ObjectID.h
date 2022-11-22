#ifndef OBJECT_ID_H
#define OBJECT_ID_H

struct Object_ID{
	int nCpy;

	Object_ID *prev;
	Object_ID *next;
};


class Object_ID_List{
	Object_ID *head;

public:
	Object_ID_List();
	Object_ID* Create();
	void Erase(Object_ID *currObj_ID);
	~Object_ID_List();
};


#endif
