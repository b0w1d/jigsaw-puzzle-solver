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
  cv::Mat img_inv(img.rows, img.cols, CV_8UC3);
  for (int i = 0; i < img.rows; ++i) {
    for (int j = 0; j < img.cols; ++j) {
      for (int k = 0; k < 3; ++k) {
        img_inv.at<cv::Vec3b>(i, j)[k] = 255 - img.at<cv::Vec3b>(i, j)[k];
      }
    }
  } // invert colors

  cv::Mat img_gray;
  cv::cvtColor(img_inv, img_gray, cv::COLOR_BGR2GRAY);

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
  for (int i = 0; i < res.rows; ++i) {
    for (int j = 0; j < res.cols; ++j) {
      if (~col[i][j]) continue;
      ++col_cnt;
      flood(res.rows, res.cols, i, j, col_cnt,
            [](int x, int y) { return false; }, col);
    }
  }
  std::vector<std::vector<kika::cod>> col2pos(col_cnt + 1);
  for (int i = 0; i < res.rows; ++i) {
    for (int j = 0; j < res.cols; ++j) {
      if (col[i][j]) {
        col2pos[col[i][j]].push_back(kika::cod(i, j));
      }
    }
  }
  return vec2::slice(col2pos, 1);
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
  Piece(std::vector<kika::cod> pnts) {
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
    max_x -= min_x - 2;
    min_x = 1;
    max_y -= min_y - 2;
    min_y = 1;

    std::vector<kika::cod> ch = kika::full_convex_hull(pnts);

    col.assign(max_x + 1, std::vector<int>(max_y + 1, -1));
    for (const kika::cod &p : pnts) col[std::real(p)][std::imag(p)] = 200;
    for (const kika::cod &p : ch) col[std::real(p)][std::imag(p)] = 200;

    int col_cnt = 0;
    for (int i = 0; i < max_x + 1; ++i) {
      for (int j = 0; j < max_y + 1; ++j) {
        if (~col[i][j]) continue;
        flood(max_x + 1, max_y + 1, i, j, col_cnt, [](int x, int y) { return false; }, col);
        ++col_cnt;
      }
    }
    std::vector<std::vector<kika::cod>> ccs(col_cnt - 1);
    for (int i = 0; i < max_x + 1; ++i) {
      for (int j = 0; j < max_y + 1; ++j) {
        if (col[i][j] && col[i][j] < 200) { // skip if background | convex hull
          assert(0 < col[i][j] && col[i][j] <= ccs.size());
          ccs[col[i][j] - 1].emplace_back(i, j);
        }
      }
    }

    ccs = vec2::select(ccs, [](std::vector<kika::cod> v) { return 10 < v.size(); });

    std::vector<kika::cod> gs(ccs.size());
    std::vector<kika::cod> fs(ccs.size());
    std::vector<kika::cod> vs(ccs.size());
    std::vector<kika::cod> rs(ccs.size());
    std::vector<bool> qs(ccs.size());
    for (int i = 0; i < ccs.size(); ++i) {
      gs[i] = std::accumulate(ccs[i].begin(), ccs[i].end(), kika::cod()) /
              kika::cod(ccs[i].size(), 0);
      assert(ccs[i].size());
      fs[i] = *std::max_element(
          ccs[i].begin(), ccs[i].end(), [&](kika::cod p, kika::cod q) {
            return kika::dist2(p, gs[i]) < kika::dist2(q, gs[i]);
          });
      vs[i] = fs[i] - gs[i];
      double sx = 0;
      double sy = 0;
      for (const kika::cod &p : ccs[i]) {
        double dx = std::real(p) - std::real(gs[i]);
        double dy = std::imag(p) - std::imag(gs[i]);
        sx += dx * dx / ccs[i].size();
        sy += dy * dy / ccs[i].size();
      }
      rs[i] = kika::cod(sx, sy);
      qs[i] = 0.2 < std::abs(sx - sy) / (sx + sy);
      /* if (0.2 < std::abs(sx - sy) / (sx + sy)) { // convex */
      /*   col[std::real(furthest)][std::imag(furthest)] = 255; */
      /* } */
    }

    std::vector<std::tuple<double, int>> best(
        ccs.size(), {std::numeric_limits<double>::max(), -1});
    for (int i = 0; i < ccs.size(); ++i) {
      if (!qs[i]) continue;
      for (int j = 0; j < ccs.size(); ++j) {
        if (!qs[j]) continue;
        if (i == j) continue;
        double angle = kika::angle(vs[i], vs[j]);
        if (acos(-1) * 3 / 4 < angle) {
          double dd = kika::dist2(gs[i], gs[j]);
          if (dd < std::get<0>(best[i])) {
            best[i] = std::make_tuple(dd, j);
          }
        }
      }
    }

    col.assign(max_x + 1, std::vector<int>(max_y + 1, -1));
    for (auto p : ch) col[std::real(p)][std::imag(p)] = 200;
    flood(max_x + 1, max_y + 1, 0, 0, 0, [](int a, int b) { return false; }, col);
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
    col_cnt = 1;
    for (int i = 0; i < max_x + 1; ++i) {
      for (int j = 0; j < max_y + 1; ++j) {
        if (~col[i][j]) continue;
        flood(max_x + 1, max_y + 1, i, j, col_cnt, [&](int a, int b) { return false; }, col);
        ++col_cnt;
      }
    }
    std::vector<std::vector<kika::cod>> col2pnts(std::max(256, col_cnt));
    for (int i = 0; i < max_x + 1; ++i) {
      for (int j = 0; j < max_y + 1; ++j) {
        col2pnts[col[i][j]].emplace_back(i, j);
      }
    }
    int max_size = 0;
    int best_i = -1;
    for (int i = 1; i < col2pnts.size(); ++i) {
      if (max_size < col2pnts[i].size()) {
        max_size = col2pnts[i].size();
        best_i = i;
      }
    }

    std::vector<kika::cod> ch_inside = kika::full_convex_hull(col2pnts[best_i]);
    for (int i = 0; i < ch_inside.size(); ++i) {
      const int n = 10;
      std::vector<kika::cod> prev(n);
      std::vector<kika::cod> next(n);
      for (int j = 0; j < n; ++j) {
        prev[j] = ch_inside[(i - j + ch_inside.size()) % ch_inside.size()];
        next[j] = ch_inside[(i + j) % ch_inside.size()];
      }
      cod p_ab = kika::least_squares(prev);
      cod n_ab = kika::least_squares(next);
    }
  }
};

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: ./solve path_to_image" << std::endl;
    return 0;
  }

  cv::Mat img = cv::imread(argv[1]);

  std::vector<std::vector<kika::cod>> pieces = extractPieces(img);

  cv::Mat img_d(img.rows, img.cols, CV_8U, cv::Scalar(0));

  int prev_max_r = 0;
  int next_max_r = 0;
  int cur_c = 0;
  for (int i = 0; i < pieces.size(); ++i) {
    Piece piece(pieces[i]);
    int r = piece.col.size();
    int rr = 10 + r + 10;
    int c = piece.col[0].size();
    int cc = 10 + c + 10;
    next_max_r = std::max(next_max_r, prev_max_r + rr);
    if (img.cols <= cur_c + cc) {
      cur_c = 0;
      prev_max_r = next_max_r;
    }
    render(img_d, piece.col, prev_max_r + 10, cur_c + 10);
    cur_c += cc;
  }

  /* for (int i = 1; i < pieces.size(); ++i) { */
  /*   std::cout << i << std::endl; */
  /*   Piece piece(pieces[i]); */
  /* } */

  cv::imwrite("pieces.png", img_d);
  cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); 
  cv::imshow("Display window", img_d);  
  cv::waitKey(0);

  return 0;
}
