#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <algorithm>
#include <numeric>
#include <functional>
#include <iterator>
#include <type_traits>
#include <initializer_list>
#include <cmath>
#include <cassert>
#include <utility>
#include <tuple>
#include "opencv2/opencv.hpp"
#include "src/vec2.hpp"
#include "src/kika.hpp"

void flood(
  int R, int C, // #row, #col
  int sx, int sy, int c, // start x, y, color to fill
  std::function<bool(int, int)> invalid, // (x, y).invalid?
  std::vector<std::vector<int>> &col // to color on, initialize with 0
) {
  assert(-1 < sx && sx < R);
  assert(-1 < sy && sy < C);
  std::queue<int> que;
  que.push(sx * C + sy);
  col[sx][sy] = c;
  while (que.size()) {
    int x = que.front() / C;
    int y = que.front() % C;
    que.pop();
    for (int d = 0; d < 4; ++d) {
      static const int dx[] = {0, 1, 0, -1};
      static const int dy[] = {1, 0, -1, 0};
      int nx = x + dx[d];
      int ny = y + dy[d];
      if (nx < 0 || R <= nx || ny < 0 || C <= ny) continue;
      if (~col[nx][ny]) continue;
      if (invalid(nx, ny)) continue;
      que.push(nx * C + ny);
      col[nx][ny] = c;
    }
  }
};

std::vector<std::vector<kika::cod>> extractPieces(const cv::Mat &img) {
  cv::Mat img_gray;
  cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

  std::vector<std::vector<int>> mat(img.rows, std::vector<int>(img.cols));
  for (int i = 0; i < img.rows; ++i) {
    for (int j = 0; j < img.cols; ++j) {
      mat[i][j] = img_gray.at<uchar>(i, j);
    }
  }
  std::vector<std::vector<int>> sobel_filter_x = {
    {-1, 0, +1},
    {-2, 0, +2},
    {-1, 0, +1}
  };
  std::vector<std::vector<int>> sobel_filter_y = vec2::transpose(sobel_filter_x);
  std::vector<std::vector<int>> gx = vec2::convolve(mat, sobel_filter_x);
  std::vector<std::vector<int>> gy = vec2::convolve(mat, sobel_filter_y);
  cv::Mat res(img.rows, img.cols, CV_8U);
  for (int i = 1; i + 1 < img.rows; ++i) {
    for (int j = 1; j + 1 < img.cols; ++j) {
      int magnitude = std::sqrt(gx[i][j]*gx[i][j] + gy[i][j]*gy[i][j]);
      int threshold = 70;
      res.at<uchar>(i, j) = (threshold < magnitude) * magnitude;
    }
  }
  int col_cnt = 0; // assumes between every piece there is some space, as well as the corner
  std::vector<std::vector<int>> col(res.rows, std::vector<int>(res.cols, -1));
  flood(res.rows, res.cols, 0, 0, col_cnt,
        [&](int x, int y) { return 0 < res.at<uchar>(x, y); }, col);
  for (int i = 0; i < col.size(); ++i) {
    for (int j = 0; j < col[i].size(); ++j) {
      if (~col[i][j]) continue;
      ++col_cnt;
      flood(res.rows, res.cols, i, j, col_cnt, [](int x, int y) { return false; }, col);
    }
  }
  std::vector<std::vector<kika::cod>> col2pos(col_cnt + 1);
  for (int i = 0; i < col.size(); ++i) {
    for (int j = 0; j < col[i].size(); ++j) {
      if (col[i][j]) {
        col2pos[col[i][j]].push_back(kika::cod(i, j));
      }
    }
  }
  col2pos.erase(col2pos.begin());
  return col2pos;
}

std::tuple<double, double, double, double, std::vector<kika::cod>>
normalize_pnts(std::vector<kika::cod> pnts) {
  int min_x = std::real(pnts[0]);
  int max_x = std::real(pnts[0]);
  int min_y = std::imag(pnts[0]);
  int max_y = std::imag(pnts[0]);
  for (const kika::cod &pnt : pnts) {
    min_x = std::min<int>(min_x, std::real(pnt));
    max_x = std::max<int>(max_x, std::real(pnt));
    min_y = std::min<int>(min_y, std::imag(pnt));
    max_y = std::max<int>(max_y, std::imag(pnt));
  }
  for (kika::cod &pnt : pnts) {
    pnt -= kika::cod(min_x - 1, min_y - 1);
  }
  max_x -= min_x - 1;
  max_y -= min_y - 1;
  return std::make_tuple(min_x - 1, min_y - 1, max_x + 2, max_y + 2, pnts);
} // R, C, pnts, guarantee border is free

std::vector<std::vector<kika::cod>> flood_all(std::vector<std::vector<int>> &col) {
  int col_cnt = 0;
  for (int i = 0; i < col.size(); ++i) {
    for (int j = 0; j < col[0].size(); ++j) {
      if (~col[i][j]) continue;
      flood(col.size(), col[0].size(), i, j, col_cnt, [](int x, int y) { return false; }, col);
      ++col_cnt;
    }
  }
  std::vector<std::vector<kika::cod>> ccs(col_cnt - 1);
  for (int i = 0; i < col.size(); ++i) {
    for (int j = 0; j < col[0].size(); ++j) {
      if (col[i][j] && col[i][j] < col_cnt) { // skip if background (col = 0)
        assert(0 < col[i][j] && col[i][j] <= ccs.size());
        ccs[col[i][j] - 1].emplace_back(i, j);
      }
    }
  }
  return ccs;
}

void render(cv::Mat &mat, const std::vector<kika::cod> &pnts) {
  for (auto pnt : pnts) {
    int x = real(pnt);
    int y = imag(pnt);
    mat.at<uchar>(x, y) = 255;
  }
}

void render(cv::Mat &mat, const std::vector<std::vector<int>> &col, int r = 0, int c = 0) {
  for (int i = 0; i < col.size(); ++i) {
    for (int j = 0; j < col[0].size(); ++j) {
      if (col[i][j]) {
        mat.at<uchar>(r + i, c + j) = col[i][j];
      }
    }
  }
}

struct Piece {
  std::vector<std::vector<int>> col;
  std::vector<kika::cod> convex_pnts;
  std::vector<kika::cod> corner_pnts;
  std::vector<kika::cod> ch_inside;
  std::vector<kika::cod> corner4;
  std::vector<int> edge_type;
  int offset_x;
  int offset_y;
  int perimeter;
  Piece(std::vector<kika::cod> P) {
    double max_x, max_y; std::tie(offset_x, offset_y, max_x, max_y, P) = normalize_pnts(P);
    std::vector<kika::cod> ch = kika::full_convex_hull(P);
    col.assign(max_x, std::vector<int>(max_y, -1));
    for (int i = 0; i < P.size(); ++i) {
      col[std::real(P[i])][std::imag(P[i])] = 200;
    }
    for (const kika::cod &p : ch) col[std::real(p)][std::imag(p)] = 200;
    std::vector<std::vector<kika::cod>> ccs = vec2::select(
      flood_all(col),
      [](const std::vector<kika::cod> &v) { return 10 < v.size(); }
    );

    std::vector<kika::cod> gs(ccs.size());
    std::vector<kika::cod> fs(ccs.size());
    std::vector<kika::cod> vs(ccs.size());
    std::vector<bool> qs(ccs.size());
    for (int i = 0; i < ccs.size(); ++i) {
      gs[i] = std::accumulate(ccs[i].begin(), ccs[i].end(), kika::cod()) /
              kika::cod(ccs[i].size(), 0);
      fs[i] = *std::max_element(
          ccs[i].begin(), ccs[i].end(), [&](kika::cod p, kika::cod q) {
            return kika::dist2(p, gs[i]) < kika::dist2(q, gs[i]);
          });
      vs[i] = fs[i] - gs[i];
      double sx = 0;
      double sy = 0;
      double norm_ang = kika::angle360(kika::cod(1, 0), vs[i]);
      kika::cod g_normed = kika::rotate(gs[i], -norm_ang);
      for (const kika::cod &p : ccs[i]) {
        kika::cod q = kika::rotate(p, -norm_ang);
        double dx = std::real(q) - std::real(g_normed);
        double dy = std::imag(q) - std::imag(g_normed);
        sx += dx * dx / ccs[i].size();
        sy += dy * dy / ccs[i].size();
      }
      qs[i] = sy * 2 < sx;
    }

    std::vector<std::tuple<double, int>> best(
        ccs.size(), {std::numeric_limits<double>::max(), -1});
    for (int i = 0; i < ccs.size(); ++i) {
      if (!qs[i]) continue;
      for (int j = 0; j < ccs.size(); ++j) {
        if (!qs[j]) continue;
        if (i == j) continue;
        double angle = kika::angle180(vs[i], vs[j]);
        if (acos(-1) * 3 / 4 < angle) {
          double dd = kika::dist2(gs[i], gs[j]);
          if (dd < std::get<0>(best[i])) {
            best[i] = std::make_tuple(dd, j);
          }
        }
      }
    }

    col.assign(max_x, std::vector<int>(max_y, -1));
    for (auto p : ch) col[std::real(p)][std::imag(p)] = 100;
    for (int i = 0; i < ccs.size(); ++i) {
      if (!qs[i]) continue;
      int j = std::get<1>(best[i]);
      assert(i == std::get<1>(best[j]));
      if (j < i) continue;
      std::vector<kika::cod> pnts;
      for (const kika::cod &p : ccs[i]) pnts.push_back(p);
      for (const kika::cod &p : ccs[j]) pnts.push_back(p);
      std::vector<kika::cod> ch = kika::full_convex_hull(pnts);
      for (auto p : ch) col[std::real(p)][std::imag(p)] = 200;
    }

    std::vector<std::vector<kika::cod>> col2pnts = flood_all(col);
    int best_i = std::max_element(col2pnts.begin(), col2pnts.end(),
                                  [](const std::vector<kika::cod> &u,
                                     const std::vector<kika::cod> &v) {
                                    return u.size() < v.size();
                                  }) -
                 col2pnts.begin();

    /* for (auto p : col2pnts[best_i]) col[std::real(p)][std::imag(p)] = 200; */

    kika::cod g = std::accumulate(col2pnts[best_i].begin(), col2pnts[best_i].end(), kika::cod());
    g /= kika::cod(col2pnts[best_i].size(), 0);

    ch_inside = kika::full_convex_hull(col2pnts[best_i]);
    perimeter = ch_inside.size();
    std::vector<std::tuple<double, double, double, double>> pnts;
    for (int i = 0; i < ch_inside.size(); ++i) {
      double angle = kika::angle360(kika::cod(1, 0), ch_inside[i] - g);
      double dd = kika::dist(ch_inside[i], g);
      pnts.emplace_back(angle, dd, std::real(ch_inside[i]), std::imag(ch_inside[i]));
    }
    std::sort(pnts.begin(), pnts.end());
    std::vector<kika::cod> plots(628);
    std::vector<kika::cod> plots_xy(628);
    for (int i = 0; i < 628; ++i) {
      auto it = std::upper_bound(pnts.begin(), pnts.end(), std::make_tuple(0.01 * i, 0));
      int b = it - pnts.begin();
      if (b == pnts.size()) b = 0;
      int a = (b - 1 + pnts.size()) % pnts.size();
      double p = i * 0.01 - std::get<0>(pnts[a]);
      double q = std::get<0>(pnts[b]) - i * 0.01;
      double dd =
          1e-3 < p + q
              ? (std::get<1>(pnts[a]) * q + std::get<1>(pnts[b]) * p) / (p + q)
              : (std::get<1>(pnts[a]) + std::get<1>(pnts[b])) / 2;
      plots[i] = kika::cod(0.01 * i, dd);
      plots_xy[i] = kika::cod(std::get<2>(pnts[a]), std::get<3>(pnts[a]));
    }

    for (int i = 0; i < plots.size(); ++i) {
      const int n = 15;
      std::vector<kika::cod> prev;
      std::vector<kika::cod> next;
      for (int j = 1; j < 1 + n; ++j) {
        int u = (i - j + plots.size()) % plots.size();
        int v = (i + j) % plots.size();
        for (int k = 0; k * k < n + 1 - j; ++k) {
          prev.push_back(plots[u]);
          next.push_back(plots[v]);
        }
      }
      kika::cod p_ab = kika::least_squares(prev);
      kika::cod n_ab = kika::least_squares(next);
      kika::cod p_v = kika::cod(1, std::real(p_ab));
      kika::cod n_v = kika::cod(1, std::real(n_ab));
      if (std::abs(std::imag(p_v)) == std::numeric_limits<double>::max()) {
        p_v = kika::cod(0, std::imag(p_v) < 0 ? -1 : 1);
      }
      if (std::abs(std::imag(n_v)) == std::numeric_limits<double>::max()) {
        n_v = kika::cod(0, std::imag(n_v) < 0 ? -1 : 1);
      }
      double angle = kika::angle360(p_v, n_v);
      if (acos(-1) < angle && angle < acos(-1) * 5 / 4) { // is corner
        corner_pnts.push_back(plots_xy[i]);
      }
    }

    { // corner4
      double maxd = 0;
      for (int i = 0; i < corner_pnts.size(); ++i) { // may improve by monotonicity
        for (int j = i + 1; j < corner_pnts.size(); ++j) {
          maxd = std::max(maxd, std::abs(corner_pnts[i] - corner_pnts[j]));
        }
      }
      const double T = maxd / 5; // threshold
      std::cout << T << std::endl;
      int p = 0;
      while (p < corner_pnts.size() && std::abs(corner_pnts[p] - corner_pnts[(p + 1) % corner_pnts.size()]) < T) ++p;
      assert(p < corner_pnts.size() && "Corner points are too close");
      for (int i = (p + 1) % corner_pnts.size(); corner4.size() < 4; ) {
        kika::cod g = corner_pnts[i];
        int j = (i + 1) % corner_pnts.size();
        for (; std::abs(corner_pnts[j] - corner_pnts[(j - 1 + corner_pnts.size()) % corner_pnts.size()]) < T; (j += 1) %= corner_pnts.size()) ;
        int m = i < j ? (i+j)/2 : (i+corner_pnts.size()+j)/2%corner_pnts.size();
        corner4.push_back(corner_pnts[m]);
        /* std::cout << "[" << i << ", " << j << "): " << std::abs(corner_pnts[j] - corner_pnts[(j - 1 + corner_pnts.size()) % corner_pnts.size()]) << std::endl; */
        i = j;
      }
      assert(corner4.front() != corner4.back());
    }

    { // edge_type
      edge_type.assign(4, 0);
      for (int i = 0; i < 4; ++i) {
        kika::cod u = corner4[i];
        kika::cod v = corner4[(i + 1) % 4];
        std::vector<double> w(2);
        for (const kika::cod &p : P) {
          double a = kika::cross(v - u, p - u);
          if (a < 0) w[0] = std::max(w[0], -a);
          else if (0 < a) w[1] = std::max(w[1], a);
        }
        if (std::min(w[0], w[1]) * 10 > std::max(w[0], w[1])) { // TODO: unsafe
          edge_type[i] = 2; // convex
        } else {
          std::set<std::pair<int, int>> bag;
          for (const kika::cod &p : P) {
            int x = std::real(p);
            int y = std::imag(p);
            bag.emplace(x, y);
          }
          int empty_count = [&]() {
            int res = 0;
            int lx = real(corner4[i]);
            int ly = imag(corner4[i]);
            int ux = real(corner4[(i + 1) % 4]);
            int uy = imag(corner4[(i + 1) % 4]);
            int dx = ux - lx;
            int dy = uy - ly;
            static const int _dx[] = {1, 0, -1, 0, 1, 1, -1, -1};
            static const int _dy[] = {0, 1, 0, -1, 1, -1, 1, -1};
            if (std::abs(dy) < std::abs(dx)) {
              double t = 1.0 * dy / dx;
              for (int j = std::min(0, dx); j <= std::max(0, dx); ++j) {
                int x = lx + j;
                int y = ly + t * j;
                bool flag = false;
                for (int k = 0; k < 8; ++k) {
                  if (bag.count(std::make_pair(x + _dx[k], y + _dy[k]))) {
                    flag = true;
                    break;
                  }
                }
                res += !flag;
              }
            } else {
              double t = 1.0 * dx / dy;
              for (int j = std::min(0, dy); j <= std::max(0, dy); ++j) {
                int x = lx + t * j;
                int y = ly + j;
                bool flag = false;
                for (int k = 0; k < 8; ++k) {
                  if (bag.count(std::make_pair(x + _dx[k], y + _dy[k]))) {
                    flag = true;
                    break;
                  }
                }
                res += !flag;
              }
            }
            return res;
          } ();
          edge_type[i] = 3 < empty_count;
        }
      }
    }

    col.assign(max_x, std::vector<int>(max_y, 0));
    for (auto p : P) {
      assert(-1 < std::real(p) && std::real(p) < col.size());
      assert(-1 < std::imag(p) && std::imag(p) < col[std::real(p)].size());
      col[std::real(p)][std::imag(p)] = 20;
    }
    /* for (auto p : ch_inside) col[std::real(p)][std::imag(p)] = 100; */
    for (auto p : corner4) col[std::real(p)][std::imag(p)] = 200;
    /* for (int i = 0; i < 4; ++i) { */
    /*   if (edge_type[i] == 0) col[std::real(corner4[i])][std::imag(corner4[i])] = 200; */
    /* } */
  }
  void match(const cv::Mat &img, const Piece &oth) {
    for (int i = 0; i < 4; ++i) {
      kika::cod da = corner4[(i + 1) % 4] - corner4[i];
      for (int j = 0; j < 4; ++j) {
        kika::cod db = oth.corner4[(j + 1) % 4] - oth.corner4[j];
        if (0.5 < std::abs(std::abs(da) - std::abs(db))) continue;
        /* std::cout << i << " " << j << " " << da << " " << db << " " << std::abs(da) << " " << std::abs(db) << std::endl; */
        return;
      }
    }
    int side = std::min(perimeter, oth.perimeter) / 4;
    /* Piece piece4(pieces[4]); // (121, 17), (118, 155) */ 
    /* Piece piece1(pieces[1]); // (13, 13), (9, 151) */
    int R = col.size() + oth.col.size();
    int C = col[0].size() + oth.col[0].size();
    cv::Mat img_d(R, C, CV_8UC3, cv::Scalar(0));
    for (int i = 0; i < col.size(); ++i) {
      for (int j = 0; j < col[0].size(); ++j) {
        if (col[i][j] == 0) continue;
        for (int k = 0; k < 3; ++k) {
          img_d.at<cv::Vec3b>(i, j)[k] = img.at<cv::Vec3b>(offset_x + i, offset_y + j)[k];
        }
      }
    }
    int rel_offset_x = 121 - 13;
    int rel_offset_y = 17 - 13;
    for (int i = 0; i < oth.col.size(); ++i) {
      for (int j = 0; j < oth.col[0].size(); ++j) {
        if (oth.col[i][j] == 0) continue;
        for (int k = 0; k < 3; ++k) {
          img_d.at<cv::Vec3b>(rel_offset_x + i, rel_offset_y + j)[k] =
              img.at<cv::Vec3b>(oth.offset_x + i, oth.offset_y + j)[k];
        }
      }
    }
    cv::imwrite("pieces.png", img_d);
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); 
    cv::imshow("Display window", img_d);  
    cv::waitKey(0);
  }
  void render(const cv::Mat &img) {
    cv::Mat img_d(col.size(), col[0].size(), CV_8UC3, cv::Scalar(0));
    for (int i = 0; i < col.size(); ++i) {
      for (int j = 0; j < col[0].size(); ++j) {
        if (col[i][j] == 0) continue;
        for (int k = 0; k < 3; ++k) {
          img_d.at<cv::Vec3b>(i, j)[k] = img.at<cv::Vec3b>(offset_x + i, offset_y + j)[k];
        }
      }
    }
    cv::imwrite("pieces.png", img_d);
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); 
    cv::imshow("Display window", img_d);  
    cv::waitKey(0);
  }
};

struct Board {
  std::vector<Piece> pieces;
  std::vector<std::vector<std::tuple<int, int, double, kika::cod, kika::cod, kika::cod, kika::cod, kika::cod>>> sol;
  // index, edge_id, rot_angle, rot_uvec, rot_lvec, ul-xy, ur-xy, bl-xy
  Board(std::vector<Piece> _pieces) : pieces(_pieces) {
    /* srand(time(0)); random_shuffle(pieces.begin(), pieces.end()); */
    for (auto p : pieces) {
      for (int d = 0; d < 4; ++d) std::cout << p.corner4[d] << " ";
      std::cout << p.col.size() << std::endl;
    }
    std::vector<int> used(pieces.size());
    for (int row = 0, done = 0; !done; ++row) {
      sol.resize(row + 1);
      assert(row < sol.size());
      for (int col = 0; ; ++col) {
        std::cout << row << ", " << col << ": " << std::endl;
        sol[row].resize(col + 1);
        assert(col < sol[row].size());
        std::vector<int> et(2); {
          if (row) {
            int pid = std::get<0>(sol[row - 1][col]);
            int eid = (std::get<1>(sol[row - 1][col]) + 2) % 4;
            assert(pid < pieces.size());
            assert(eid < pieces[pid].edge_type.size());
            et[0] = pieces[pid].edge_type[eid];
          }
          if (col) {
            int pid = std::get<0>(sol[row][col - 1]);
            int eid = (std::get<1>(sol[row][col - 1]) + 3) % 4;
            assert(pid < pieces.size());
            assert(eid < pieces[pid].edge_type.size());
            et[1] = pieces[pid].edge_type[eid];
          }
        }
        kika::cod u_vec = row ? std::get<3>(sol[row - 1][col]) : kika::cod(0, 1);
        kika::cod l_vec = col ? std::get<4>(sol[row][col - 1]) : kika::cod(-1, 0);
        std::cout << std::abs(u_vec) << " " << std::abs(l_vec) << std::endl;
        double ang = kika::angle360(u_vec, l_vec);
        kika::cod st = row ? std::get<7>(sol[row - 1][col]) : col ? std::get<6>(sol[row][col - 1]) : kika::cod();
        std::cout << st << std::endl;
        std::cout << ang << " " << et[0] << " " << et[1] << std::endl;
        auto find_and_put = [&]() {
          for (int pi = 0; pi < pieces.size(); ++pi) {
            if (used[pi]) continue;
            /* if (row == 3 && col == 0 && pi == 6) continue; */
            assert(pieces[pi].edge_type.size() == 4);
            assert(pieces[pi].corner4.size() == 4);
            for (int d = 0; d < 4; ++d) {
              if ((et[0] + pieces[pi].edge_type[d]) % 3 == 0) {
                if ((et[1] + pieces[pi].edge_type[(d + 1) % 4]) % 3 == 0) {
                  kika::cod pu_vec = pieces[pi].corner4[(d + 1) % 4] - pieces[pi].corner4[(d + 0) % 4];
                  kika::cod pl_vec = pieces[pi].corner4[(d + 2) % 4] - pieces[pi].corner4[(d + 1) % 4];
                  double p_ang = kika::angle360(pu_vec, pl_vec);
                  std::cout << std::abs(pu_vec) << " " << std::abs(pl_vec) << " " << std::abs(p_ang) << std::endl;
                  double d_ang = std::min(std::abs(p_ang - ang), std::abs(std::abs(p_ang - ang) - 2 * acos(-1)));
                  double d_ulen = row ? std::abs(std::abs(pu_vec) - std::abs(u_vec)) : 0;
                  double d_llen = col ? std::abs(std::abs(pl_vec) - std::abs(l_vec)) : 0;
                  std::cout << "*" << pi << " " << d_ang << " " << d_ulen << " " << d_llen << std::endl;
                  if (row == 4 && col == 1) {
                  } else {
                  if (0.06 < d_ang || 6 < d_ulen || 6 < d_llen) continue;}
                  /* if (1 < d_ulen + d_llen + d_ang * 4) continue; */

                  double rot_ang = kika::angle360(pu_vec, u_vec) + std::acos(-1);
                  kika::cod nuvec = pieces[pi].corner4[(d + 3) % 4] - pieces[pi].corner4[(d + 2) % 4];
                  kika::cod nlvec = pieces[pi].corner4[(d + 4) % 4] - pieces[pi].corner4[(d + 3) % 4];
                  kika::cod rot_nuvec = kika::rotate(nuvec, rot_ang);
                  kika::cod rot_nlvec = kika::rotate(nlvec, rot_ang);
                  kika::cod rvec = pieces[pi].corner4[(d + 0) % 4] - pieces[pi].corner4[(d + 1) % 4];
                  kika::cod dvec = pieces[pi].corner4[(d + 2) % 4] - pieces[pi].corner4[(d + 1) % 4];
                  kika::cod rot_rvec = kika::rotate(rvec, rot_ang);
                  kika::cod rot_dvec = kika::rotate(dvec, rot_ang);
                  kika::cod offset_rot = kika::rotate(pieces[pi].corner4[(d + 1) % 4], rot_ang);
                  kika::cod ur = st + rot_rvec;
                  kika::cod bl = st + rot_dvec;

                  sol[row][col] = std::make_tuple(pi, d, rot_ang, rot_nuvec, rot_nlvec, st - offset_rot, ur, bl);
                  std::cout << pi << std::endl;
                  std::cout << rvec << " " << rot_rvec << " " << offset_rot << " " << ur << " " << bl << " " << st << std::endl;

                  used[pi] = 1; // if really match
                  return (pieces[pi].edge_type[(d + 3) % 4] == 0) |
                         (pieces[pi].edge_type[(d + 2) % 4] == 0) << 1;
                }
              }
            }
          }
          return -1;
        };
        int ret = find_and_put();
        std::cout << ret << std::endl;
        if (ret == -1) { sol[row].pop_back(); return; }
        if (ret == 3) done = true;
        else if (ret == 1) break;
      }
    }
  }
  void render(const cv::Mat &img) {
    const int offset = 10;
    int R = 800;//col.size() + oth.col.size();
    int C = 800;//col[0].size() + oth.col[0].size();
    cv::Mat img_d(R, C, CV_8UC3, cv::Scalar(0));
    for (int row = 0; row < sol.size(); ++row) {
      for (const auto &p : sol[row]) {
        int pid = std::get<0>(p);
        double rot_ang = std::get<2>(p);
        kika::cod ul = std::get<5>(p);
        for (int i = 0; i < pieces[pid].col.size(); ++i) {
          for (int j = 0; j < pieces[pid].col[i].size(); ++j) {
            if (pieces[pid].col[i][j] == 0) continue;
            for (int k = 0; k < 3; ++k) {
              /* std::cout << i << " " << j << " " << rot_ang << std::endl; */
              kika::cod nij = ul + kika::rotate(kika::cod(i, j), rot_ang);
              int ni = offset + std::real(nij);
              int nj = offset + std::imag(nij);
              /* std::cout << ni << " " << nj << std::endl; */
              assert(-1 < ni && -1 < nj && ni < R && nj < C);
              img_d.at<cv::Vec3b>(ni, nj)[k] = img.at<cv::Vec3b>(pieces[pid].offset_x + i, pieces[pid].offset_y + j)[k];
            }
          }
        }
      }
    }
    cv::imwrite("pieces.png", img_d);
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); 
    cv::imshow("Display window", img_d);  
    cv::waitKey(0);
  }
};

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: ./solve path_to_image" << std::endl;
    return 0;
  }

  cv::Mat img = cv::imread(argv[1]);

  std::vector<std::vector<kika::cod>> piece_pnts = extractPieces(img);
  for (auto v : piece_pnts) {
    std::cout << v.size() << std::endl;
  }

  std::vector<Piece> pieces;
  for (auto pnts : piece_pnts) pieces.push_back(Piece(pnts));
    for (auto p : pieces) {
      for (int d = 0; d < 4; ++d) std::cout << p.corner4[d] << " ";
      std::cout << p.col.size() << std::endl;
    }

  /* Piece piece4(pieces[4]); // (121, 17), (118, 155) */ 
  /* /1* piece4.render(img); *1/ */
  /* piece4.match(img, piece1); */

  /* cv::Mat img_d(img.rows, img.cols, CV_8U, cv::Scalar(0)); */

  /* int prev_max_r = 0; */
  /* int next_max_r = 0; */
  /* int cur_c = 0; */
  /* for (int i = 0; i < pieces.size(); ++i) { */
  /*   Piece piece(pieces[i]); */
  /*   int r = piece.col.size(); */
  /*   int rr = 10 + r + 10; */
  /*   int c = piece.col[0].size(); */
  /*   int cc = 10 + c + 10; */
  /*   next_max_r = std::max(next_max_r, prev_max_r + rr); */
  /*   if (img.cols <= cur_c + cc) { */
  /*     cur_c = 0; */
  /*     prev_max_r = next_max_r; */
  /*   } */
  /*   render(img_d, piece.col, prev_max_r + 10, cur_c + 10); */
  /*   cur_c += cc; */
  /* } */

  /* cv::imwrite("pieces.png", img_d); */
  /* cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); */ 
  /* cv::imshow("Display window", img_d); */  
  /* cv::waitKey(0); */

  Board board(pieces);
  board.render(img);

  return 0;
}
