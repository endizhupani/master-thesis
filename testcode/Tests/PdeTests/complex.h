#include <stdio.h>
#pragma once
class complex
{
	double real;
	double imagine;
public: 
	complex(double r=0., double i=0.);
	complex(const complex&c);
	~complex();
	double re() const;
	double im() const;
	const complex&operator=(const complex&c);
	const complex&operator+=(const complex&c);
	const complex&operator-=(const complex&c);
	const complex&operator*=(const complex&c);
	const complex&operator/=(double d);
	friend complex operator! (const complex&c);
	friend double abs2(const complex&c);
	const complex&operator/=(const complex&c);
	const complex operator-(const complex&c);
	const friend complex operator-(const complex&lhs, const complex&rhs);
	const friend complex operator+(const complex&lhs, const complex&rhs);
	const friend complex operator*(const complex&lhs, const complex&rhs);
	const friend complex operator/(const complex&lhs, const complex&rhs);
};
