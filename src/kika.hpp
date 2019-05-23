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
double dot(cod a, cod b) {
  return real(conj(a) * b);
}
bool less(cod a, cod b) {
  return real(a) < real(b) || (real(a) == real(b) && imag(a) < imag(b));
}
double cross(cod a, cod b) {
  return imag(conj(a) * b);
}
double dist2(cod a, cod b) {
  double dx = std::real(a) - std::real(b);
  double dy = std::imag(a) - std::imag(b);
  return dx * dx + dy * dy;
}
cod rotate(cod a, double rad) {
  return a * cod(cos(rad), sin(rad));
}
double angle(cod a, cod b) {
  return acos(dot(a, b) / abs(a) / abs(b));
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
std::vector<cod> full_convex_hull(std::vector<cod> p) {
  std::vector<cod> ch = convex_hull(p);
  std::vector<cod> ch_full;
  for (int i = 0; i < ch.size(); ++i) {
    int lx = real(ch[i]);
    int ly = imag(ch[i]);
    int ux = real(ch[(i + 1) % ch.size()]);
    int uy = imag(ch[(i + 1) % ch.size()]);
    int dx = ux - lx;
    int dy = uy - ly;
    std::vector<kika::cod> seq;
    if (std::abs(dy) < std::abs(dx)) {
      double t = 1.0 * dy / dx;
      for (int j = std::min(0, dx); j <= std::max(0, dx); ++j) {
        seq.emplace_back(int(lx + j), int(ly + t * j));
      }
    } else {
      double t = 1.0 * dx / dy;
      for (int j = std::min(0, dy); j <= std::max(0, dy); ++j) {
        seq.emplace_back(int(lx + t * j), int(ly + j));
      }
    }
    if (2 < kika::dist2(seq[0], ch[i])) {
      std::reverse(seq.begin(), seq.end());
    }
    assert(kika::dist2(seq[0], ch[i]) < 3);
    ch_full.insert(ch_full.end(), seq.begin(), seq.end());
  }
  return ch_full;
}
};  // namespace kika

#endif
