#pragma once
#include "ObjectID.h"


class NN_Manager {
protected:
	static Object_Linker linker;
	Object_ID* id;

	void create();
	virtual void destroy();
	virtual void clear_param();
	void copy_id(const NN_Manager& p);

public:
	NN_Manager();
	NN_Manager(const NN_Manager& p);
	~NN_Manager();

	void clear();
};