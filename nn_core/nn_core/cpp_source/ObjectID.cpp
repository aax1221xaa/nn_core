#include "ObjectID.h"
#include <stdlib.h>


Object_ID_List::Object_ID_List(){
	head = new Object_ID;

	head->prev = head;
	head->next = head;
	
	head->nCpy = 0;
}


Object_ID* Object_ID_List::Create(){
	Object_ID *current = new Object_ID;

	Object_ID *before = head;
	Object_ID *after = head->next;

	before->next = current;
	after->prev = current;
	current->prev = before;
	current->next = after;

	current->nCpy = 1;

	return current;
}

void Object_ID_List::Erase(Object_ID *currObj_ID){
	if(currObj_ID == head) return;

	Object_ID *before = currObj_ID->prev;
	Object_ID *after = currObj_ID->next;

	before->next = after;
	after->prev = before;

	delete currObj_ID;
}


Object_ID_List::~Object_ID_List(){
	Object_ID *current = head->next;
	Object_ID *tmp = NULL;

	while(current != head){
		tmp = current->next;
		delete current;
		current = tmp;
	}
	delete head;
}