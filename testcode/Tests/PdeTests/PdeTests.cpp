// PdeTests.cpp : Defines the entry point for the application.
//

#include "PdeTests.h"

using namespace std;

typedef Vector<double, 2> point;
typedef Vector<double, 3> point3d;

int main()
{
	printf("Testing the complex class implementation\n");
	complex c= 1., d(3.,4.);
	printf("%f \n",c-d);
	printf("%f \n",c/d);
	printf("Complex class implementation test finished.\n");

	printf("Testing the point class implementation\n");
	point twoDimPoint(0);
	twoDimPoint.set(0, 1);
	twoDimPoint.set(1, 3);

	
	printf("two dim = (%f, %f)\n", twoDimPoint[0], twoDimPoint[1]);
	//printf("anotherone = (%f, %f)\n", anotherOne[0], anotherOne[1]);
	printf("Test done");
	return 0;
}
