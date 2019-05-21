#ifndef ARRAY_HPP
#define ARRAY_HPP

template<typename T>
class Array {
 public:
  std::vector<T> arr;
  Array(int n) : arr(n) {}
  Array(int n, T v) : arr(n, v) {}
  Array(std::vector<T> vec) : arr(vec.begin(), vec.end()) {}
  Array(std::initializer_list<T> l) : arr(l) {}
  int size() const { return arr.size(); }
  std::vector<T> to_a() { return arr; }
  void range(int s, int e) {
    assert(s <= e);
    arr.resize(e - s);
    std::iota(arr.begin(), arr.end(), s);
  }
  T operator[](int i) const { return arr[i]; }
  T& operator[](int i) { return arr[i]; }
  Array<T> transpose() {
    Array<T> res(arr[0].size(), T(arr.size()));
    for (int i = 0; i < arr.size(); ++i) {
      for (int j = 0; j < arr[0].size(); ++j) {
        res.arr[j][i] = arr[i][j];
      }
    }
    return res;
  }
  Array<T> convolve(const Array<T> &f) { // pass the filter in
    assert(f.size() == f[0].size());
    assert(f.size() & 1);
    Array<T> r(arr.size(), T(arr[0].size()));
    for (int i = 0; i < arr.size(); ++i) {
      for (int j = 0; j < arr[0].size(); ++j) {
        int m = f.size() / 2;
        for (int x = -m; x < m + 1; ++x) {
          for (int y = -m; y < m + 1; ++y) {
            int s = i - x;
            int t = j - y;
            if (0 <= s && s < arr.size() && 0 <= t && t < arr[0].size()) {
              r[i][j] += f[m + x][m + y] * arr[s][t];
            }
          }
        }
      }
    }
    return r;
  }
  template<typename F>
  Array<T> select(const F &f) {
    Array<T> res(arr.size());
    std::copy_if(arr.begin(), arr.end(), res.begin(), f);
    return res;
  }
  template<typename F>
  Array<typename std::result_of<F(T)>::type> map(const F &f) {
    Array<typename std::result_of<F(T)>::type> res(arr.size());
    std::transform(arr.begin(), arr.end(), res.arr.begin(), f);
    return res;
  }
  template<typename F, typename S>
  S reduce(const F &f, S s) {
    for (const T &v : arr) s = f(s, v);
    return s;
  }
  template<typename F>
  T reduce(const F &f) {
    T s = arr[0];
    for (auto it = arr.begin() + 1; it != arr.end(); ++it) s = f(s, *it);
    return s;
  }
  template<typename F>
  bool all(const F &f) { return all_of(arr.begin(), arr.end(), f); }
  T min() { return *min_element(arr.begin(), arr.end()); }
  T max() { return *max_element(arr.begin(), arr.end()); }
};

#endif
