#include "ObjectID.h"


Object_Linker::Object_Linker(){
	head = new Object_ID;

	head->prev = head;
	head->next = head;
	
	head->ref_cnt = 0;
}


Object_ID* Object_Linker::Create(){
	Object_ID *current = new Object_ID;

	Object_ID *before = head;
	Object_ID *after = head->next;

	before->next = current;
	after->prev = current;
	current->prev = before;
	current->next = after;

	current->ref_cnt = 1;

	return current;
}

void Object_Linker::Erase(Object_ID *currObj_ID){
	if(currObj_ID == head) return;

	Object_ID *before = currObj_ID->prev;
	Object_ID *after = currObj_ID->next;

	before->next = after;
	after->prev = before;

	delete currObj_ID;
}


Object_Linker::~Object_Linker(){
	Object_ID *current = head->next;
	Object_ID *tmp = NULL;

	while(current != head){
		tmp = current->next;
		delete current;
		current = tmp;
	}
	delete head;
}