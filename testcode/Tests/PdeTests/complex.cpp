#include "complex.h"

complex::complex(double r, double i): real(r), imagine(i)
{
}

complex::complex(const complex& c): real(c.real), imagine(c.imagine)
{
}

complex::~complex()
{
}

double complex::re() const
{
	return real;
}

double complex::im() const
{
	return imagine;
}

const complex& complex::operator=(const complex& c)
{
	real = c.real;
	imagine = c.imagine;
	return *this;
}

const complex& complex::operator+=(const complex& c)
{
	real += c.real;
	imagine += c.imagine;
	return *this;
}

const complex& complex::operator-=(const complex& c)
{
	real -= c.real;
	imagine -= c.imagine;
	return *this;
}

const complex& complex::operator*=(const complex& c)
{
	double keepreal = real;
	real = real * c.real - imagine * c.imagine;
	imagine *= keepreal * c.imagine + imagine * c.real;
	return *this;
}

const complex& complex::operator/=(double d)
{
	real /= d;
	imagine /= d;
	return *this;
}

const complex& complex::operator/=(const complex& c)
{
	return *this *= (!c) /= abs2(c);
}

const complex complex::operator-(const complex& c) { return complex(-c.re(), -c.im()); }

complex operator!(const complex& c)
{
	return complex();
}

double abs2(const complex& c)
{
	return c.re() * c.re() + c.im() * c.im(); 
}

const complex operator-(const complex& lhs, const complex& rhs) {
  return complex(lhs.re() - rhs.re(), lhs.im() - rhs.im());
}

const complex operator+(const complex& lhs, const complex& rhs) {
  return complex(lhs.re() + rhs.re(), lhs.im() + rhs.im());
}

const complex operator*(const complex& lhs, const complex& rhs) {
  return complex(lhs)*=rhs;
}

const complex operator/(const complex& lhs, const complex& rhs) {
  return complex(lhs)/=rhs;
}


