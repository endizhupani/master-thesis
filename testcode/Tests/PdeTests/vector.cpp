#include <stdio.h>
template <class T, int N>
class Vector {
  T component[N];

 public:
  Vector(const T&);
  Vector(const Vector&);
  const Vector& operator=(const Vector&);
  const Vector& operator=(const T&);
  const Vector& operator+=(const Vector&);
  const Vector& operator-=(const Vector&);
  const Vector& operator*=(const T&);
  const Vector& operator/=(const T&);

  const T& operator[](int i) const { return component[i]; }
  void set(int i, const T& a) { component[i] = a; }



  template <class T, int N>
  T squaredNorm(const Vector<T, N>& u) {
    return u * u;
  }

  template <class T, int N>
  const Vector<T, N>& operator+(const Vector<T, N>& u) {
    return u;
  }

  template <class T, int N>
  const Vector<T, N>& operator-(const Vector<T, N>& u) {
    return Vector<T, N>(u) *= -1;
  }

  template <class T, int N>
  void print(const Vector<T, N> &v){
    printf("(");
    for (int i = 0; i < N; i++) {
        printf("v[%d] = ", i);
        printf(v[i]);
    }
    printf(")\n");
  }

  ~Vector() { }
};

  template <class T, int N>
  const Vector<T, N> operator+(const Vector<T, N> &u, const Vector<T, N> &v) {
    return Vector<T, N>(u) += v;
  }

  template <class T, int N>
  const T operator*(const Vector<T, N>& u, const Vector<T, N>& v) {
    T sum = 0;
    for (int i = 0; i < N; i++) {
      sum += u[i] * v[i]
    }

    return sum;
  }

template <class T, int N>
Vector<T, N>::Vector(const T& a = 0) {
  for (int i = 0; i < N; i++) component[i] = a;
}

template <class T, int N>
Vector<T, N>::Vector(const Vector<T, N>& v) {
  for (int i = 0; i < N; i++) {
    component[i] = v.component[i];
  }
}

template <class T, int N>
const Vector<T, N>& Vector<T, N>::operator=(const Vector<T, N>& v) {
  if (this != &v)
    for (int i = 0; i < N; i++) {
      component[i] = v.component[i];
    }
  return *this;
}

template <class T, int N>
const Vector<T, N>& Vector<T, N>::operator=(const T& a) {
  for (int i = 0; i < N; i++) {
    component[i] = a;
  }

  return *this;
}

template <class T, int N>
const Vector<T, N>& Vector<T, N>::operator+=(const Vector<T, N>& v) {
  for (int i = 0; i < N; i++) {
    component[i] += v[i];
  }

  return *this;
}

template <class T, int N>
const Vector<T, N>& Vector<T, N>::operator-=(const Vector<T, N>& v) {
  for (int i = 0; i < N; i++) {
    component[i] -= v[i];
  }

  return *this;
}

template <class T, int N>
const Vector<T, N>& Vector<T, N>::operator*=(const T& i) {
  for (int i = 0; i < N; i++) {
    component[i] *= i;
  }

  return *this;
}

template <class T, int N>
const Vector<T, N>& Vector<T, N>::operator/=(const T& i) {
  for (int i = 0; i < N; i++) {
    component[i] /= i;
  }

  return *this;
}
