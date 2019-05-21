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

void render(cv::Mat &mat, const std::vector<std::vector<int>> &col) {
  for (int i = 0; i < col.size(); ++i) {
    for (int j = 0; j < col[0].size(); ++j) {
      if (col[i][j]) {
        mat.at<uchar>(i, j) = col[i][j];
      }
    }
  }
}

std::vector<std::vector<int>> detectPieceFeatures(int R, int C, std::vector<kika::cod> pnts) {
  std::vector<std::vector<int>> res(R, std::vector<int>(C, -1));
  for (const kika::cod &pnt : pnts) {
    int x = std::real(pnt);
    int y = std::imag(pnt);
    res[x][y] = 200;
  }
  std::vector<kika::cod> ch = kika::convex_hull(pnts);
  ch.push_back(ch[0]);
  for (int i = 0; i + 1 < ch.size(); ++i) {
    int lx = real(ch[i]);
    int ly = imag(ch[i]);
    int ux = real(ch[i + 1]);
    int uy = imag(ch[i + 1]);
    int dx = ux - lx;
    int dy = uy - ly;
    if (std::abs(dy) < std::abs(dx)) {
      double t = 1.0 * dy / dx;
      for (int j = std::min(0, dx); j < std::max(0, dx); ++j) {
        assert(0 <= lx + j && lx + j < R);
        assert(0 <= ly + t * j && ly + t * j < C);
        res[lx + j][ly + t * j] = 200;
      }
    } else {
      double t = 1.0 * dx / dy;
      for (int j = std::min(0, dy); j < std::max(0, dy); ++j) {
        assert(0 <= ly + j && ly + j < C);
        assert(0 <= lx + t * j && lx + t * j < R);
        res[lx + t * j][ly + j] = 200;
      }
    }
  }
  int col_cnt = 0;
  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < C; ++j) {
      if (~res[i][j]) continue;
      flood(R, C, i, j, col_cnt * 20, [](int x, int y) { return false; }, res);
      ++col_cnt;
    }
  }
  std::vector<std::vector<kika::cod>> mat(col_cnt - 1);
  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < C; ++j) {
      if (0 < res[i][j] && res[i][j] < col_cnt * 20) {
        mat[res[i][j] / 20 - 1].push_back(kika::cod(i, j));
      }
    }
  }
  for (int c = 0; c < mat.size(); ++c) {
    if (mat[c].size() < 10) continue;
    double mx = vec2::reduce(
      vec2::map(
        mat[c],
        [](kika::cod p) { return std::real(p); }
      ),
      std::plus<double>()
    ) / mat[c].size();
    double my = vec2::reduce(
      vec2::map(
        mat[c],
        [](kika::cod p) { return std::imag(p); }
      ),
      std::plus<double>()
    ) / mat[c].size();
    double sx = vec2::reduce(
      vec2::map(
        vec2::map(
          mat[c],
          [&](kika::cod p) { return (std::real(p) - mx); }
        ),
        [](double v) { return v * v; }
      ),
      std::plus<double>()
    ) / mat[c].size();
    double sy = vec2::reduce(
      vec2::map(
        vec2::map(
          mat[c],
          [&](kika::cod p) { return (std::imag(p) - my); }
        ),
        [](double v) { return v * v; }
      ),
      std::plus<double>()
    ) / mat[c].size();
    for (const auto &p : mat[c]) {
      int x = std::real(p);
      int y = std::imag(p);
      if (std::abs(sx - sy) / (sx + sy) < 0.2) { // concave
        res[x][y] = 50;
      } else { // convex
        res[x][y] = 100;
      }
    }
  }
  return res;
}

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: ./solve path_to_image" << std::endl;
    return 0;
  }

  cv::Mat img = cv::imread(argv[1]);

  std::vector<std::vector<kika::cod>> pieces = extractPieces(img);

  cv::Mat img_d(img.rows, img.cols, CV_8U, cv::Scalar(0));
  for (const auto &piece : pieces) {
    std::vector<std::vector<int>> col = detectPieceFeatures(img.rows, img.cols, piece);
    render(img_d, col);
  }

  cv::imwrite("pieces.png", img_d);
  cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); 
  cv::imshow("Display window", img_d);  
  cv::waitKey(0);

  return 0;
}
