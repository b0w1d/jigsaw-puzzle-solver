#ifndef KIKA_HPP
#define KIKA_HPP

namespace kika {
typedef std::complex<double> cod;
const double EPS = 1e-9;
const double PI = acos(-1);
int dcmp(double x) {
  if (abs(x) < EPS) return 0;
  return x > 0 ? 1 : -1;
}
bool less(cod a, cod b) {
  return real(a) < real(b) || (real(a) == real(b) && imag(a) < imag(b));
}
double cross(cod a, cod b) {
  return imag(conj(a) * b);
}
cod rotate(cod a, double rad) {
  return a * cod(cos(rad), sin(rad));
}
std::vector<cod> convex_hull(std::vector<cod> p) {
  std::sort(p.begin(), p.end(), less);
  p.erase(std::unique(p.begin(), p.end()), p.end());
  int n = p.size(), m = 0;
  std::vector<cod> ch(n + 1);
  for (int i = 0; i < n; ++i) {  // note that border is cleared
    while (m > 1 and
           dcmp(cross(ch[m - 1] - ch[m - 2], p[i] - ch[m - 2])) < 0) {
      --m;
    }
    ch[m++] = p[i];
  }
  for (int i = n - 2, k = m; i >= 0; --i) {
    while (m > k and
           dcmp(cross(ch[m - 1] - ch[m - 2], p[i] - ch[m - 2])) < 0) {
      --m;
    }
    ch[m++] = p[i];
  }
  ch.erase(ch.begin() + m - (n > 1), ch.end());
  return ch;
}
};  // namespace kika

#endif
