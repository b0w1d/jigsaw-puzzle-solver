#ifndef ARRAY_HPP
#define ARRAY_HPP

template<typename T>
class Array {
 public:
  std::vector<T> arr;
  Array(int n = 0) : arr(n) {}
  Array(int n, T v) : arr(n, v) {}
  Array(std::vector<T> vec) : arr(vec.begin(), vec.end()) {}
  Array(std::initializer_list<T> l) : arr(l) {}
  int size() const { return arr.size(); }
  operator std::vector<T>() const { return arr; }
  typename std::vector<T>::iterator begin() { return arr.begin(); }
  typename std::vector<T>::const_iterator begin() const { return arr.begin(); }
  typename std::vector<T>::iterator end() { return arr.end(); }
  typename std::vector<T>::const_iterator end() const { return arr.end(); }
  static Array<T> range(int s, int e) {
    assert(s <= e);
    Array<T> res(e - s);
    std::iota(res.arr.begin(), res.arr.end(), s);
    return res;
  }
  T operator[](int i) const { return arr[i]; }
  T& operator[](int i) { return arr[i]; }
  void push_back(T v) { arr.push_back(v); }
  Array<T> slice(int s, int sz = -1) const {
    assert(s + std::max(sz, 0) <= size());
    if (sz < 0) sz = size() - s;
    Array<T> res(sz);
    std::copy(arr.begin() + s, arr.begin() + s + sz, res.begin());
    return res;
  }
  Array<T> transpose() const {
    Array<T> res(arr[0].size(), T(arr.size()));
    for (int i = 0; i < arr.size(); ++i) {
      for (int j = 0; j < arr[0].size(); ++j) {
        res.arr[j][i] = arr[i][j];
      }
    }
    return res;
  }
  Array<T> convolve(const Array<T> &f) const { // pass the filter in
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
  Array<T> select(const F &f) const {
    Array<T> res(arr.size());
    std::copy_if(arr.begin(), arr.end(), res.begin(), f);
    return res;
  }
  template<typename F>
  Array<typename std::result_of<F(T)>::type> map(const F &f) const {
    Array<typename std::result_of<F(T)>::type> res(arr.size());
    std::transform(arr.begin(), arr.end(), res.arr.begin(), f);
    return res;
  }
  template<typename F, typename S>
  S reduce(const F &f, S s) const {
    for (const T &v : arr) s = f(s, v);
    return s;
  }
  template<typename F>
  T reduce(const F &f) const {
    T s = arr[0];
    for (auto it = arr.begin() + 1; it != arr.end(); ++it) s = f(s, *it);
    return s;
  }
  template<typename F>
  bool all(const F &f) const { return all_of(arr.begin(), arr.end(), f); }
  T min() const { return *min_element(arr.begin(), arr.end()); }
  T max() const { return *max_element(arr.begin(), arr.end()); }
};

#endif
