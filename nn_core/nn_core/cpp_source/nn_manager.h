#pragma once
#include "nn_link.h"


class NN_Manager {
protected:
	static bool init_flag;

	static vector<NN_Ptr<NN_Link>> links;

public:
	static void add_link(NN_Ptr<NN_Link>& link);
	static void clear_links();
	static void clear_select_flag();
	static vector<NN_Ptr<NN_Link>>& get_links();
	static void set_linked_count();

	NN_Manager();
	~NN_Manager();
};