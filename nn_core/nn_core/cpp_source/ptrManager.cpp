#include "ptrManager.h"


ptrManager::ptrManager(){
	head = (ptrRef*)malloc(sizeof(ptrRef));

	head->prev = head;
	head->next = head;
	
	head->ref_cnt = 0;
}


ptrRef* ptrManager::create(){
	ptrRef *current = (ptrRef*)malloc(sizeof(ptrRef));

	ptrRef *before = head;
	ptrRef *after = head->next;

	before->next = current;
	after->prev = current;
	current->prev = before;
	current->next = after;

	current->ref_cnt = 1;

	return current;
}

void ptrManager::erase(ptrRef **node){
	if(*node == head) return;

	ptrRef *before = (*node)->prev;
	ptrRef *after = (*node)->next;

	before->next = after;
	after->prev = before;

	free(*node);
	*node = NULL;
}


ptrManager::~ptrManager(){
	ptrRef *current = head->next;
	ptrRef *tmp = NULL;

	while(current != head){
		tmp = current->next;
		free(current);
		current = tmp;
	}
	free(head);
}