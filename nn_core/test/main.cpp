#include "../nn_core/cpp_source/dimension.h"
#include <time.h>
#include "tbb/tbb.h"
#include "vld.h"

using namespace tbb;

class A {
public:
	virtual void print();
};

class B : public A {
public:
	virtual void print();
};

void A::print() {
	printf("AAAAAAAAAAAA\n");
}

void B::print() {
	printf("BBBBBBBBBBBB\n");
}

int main() {
	try {
		A a;
		B b;

		a.print();

		A& c = b;

		c.print();

		A d = b;
		d.print();
	}
	catch (Exception& p) {
		p.Put();
	}

	return 0;
}