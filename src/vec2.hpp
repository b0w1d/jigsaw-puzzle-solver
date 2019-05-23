#ifndef VEC2_HPP
#define VEC2_HPP

namespace vec2 {
  template<typename T>
  std::vector<T> slice(const std::vector<T> &v, int s, int sz = -1) {
    assert(s + std::max(sz, 0) <= v.size());
    if (sz < 0) sz = v.size() - s;
    std::vector<T> res(sz);
    std::copy(v.begin() + s, v.begin() + s + sz, res.begin());
    return res;
  }
  template<typename T>
  std::vector<T> transpose(const std::vector<T> &v) {
    std::vector<T> res(v[0].size(), T(v.size()));
    for (int i = 0; i < v.size(); ++i) {
      for (int j = 0; j < v[0].size(); ++j) {
        res[j][i] = v[i][j];
      }
    }
    return res;
  }
  template<typename T>
  std::vector<T> convolve(const std::vector<T> &v, const std::vector<T> &f) {
    assert(f.size() == f[0].size());
    assert(f.size() & 1);
    std::vector<T> r(v.size(), T(v[0].size()));
    for (int i = 0; i < v.size(); ++i) {
      for (int j = 0; j < v[0].size(); ++j) {
        int m = f.size() / 2;
        for (int x = -m; x < m + 1; ++x) {
          for (int y = -m; y < m + 1; ++y) {
            int s = i - x;
            int t = j - y;
            if (0 <= s && s < v.size() && 0 <= t && t < v[0].size()) {
              r[i][j] += f[m + x][m + y] * v[s][t];
            }
          }
        }
      }
    }
    return r;
  }
  template<typename T, typename F>
  std::vector<T> select(const std::vector<T> &v, const F &f) {
    std::vector<T> res;
    std::copy_if(v.begin(), v.end(), std::back_inserter(res), f);
    return res;
  }
  template<typename T, typename F>
  std::vector<typename std::result_of<F(T)>::type> map(const std::vector<T> &v, const F &f) {
    std::vector<typename std::result_of<F(T)>::type> res(v.size());
    std::transform(v.begin(), v.end(), res.begin(), f);
    return res;
  }
  template<typename T, typename F, typename S>
  S reduce(const std::vector<T> &v, const F &f, S s) {
    for (const T &u : v) s = f(s, u);
    return s;
  }
  template<typename T, typename F>
  T reduce(const std::vector<T> &v, const F &f) {
    T s = v[0];
    for (auto it = v.begin() + 1; it != v.end(); ++it) s = f(s, *it);
    return s;
  }
};

#endif
