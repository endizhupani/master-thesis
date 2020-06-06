template <int N>
class point {
  double coordinate[N];

 public:
     point(double coords[N]);
  point(const point&P);
  double operator[](int i) const { return coordinate[i]; }
  ~point(){
      delete [] coordinate;
  }
};

template <int N>
point<N>::point(double coords[N]) {
  for (int i = 0; i < N; i++) {
      coordinate[i] = coords[i];
  }
}

template <int N>
point<N>::point(const point&P) {
for (int i = 0; i < N; i++) {
    coordinate[i] = P.coordinate[i];
  }
}