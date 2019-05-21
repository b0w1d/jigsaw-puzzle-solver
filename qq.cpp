#include <iostream>
#include <vector>
#include <map>
#include <queue>
#include <algorithm>
#include <functional>
#include <type_traits>
#include <initializer_list>
#include <cmath>
#include <cassert>
#include "opencv2/opencv.hpp"
#include "src/array.hpp"
#include "src/kika.hpp"
#include "src/matrix_op.hpp"

cv::Mat removeBg(const cv::Mat &img) {
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

  std::vector<std::vector<int>> mat(img_gray.rows, std::vector<int>(img_gray.cols));
  for (int i = 0; i < img_gray.rows; ++i) {
    for (int j = 0; j < img_gray.cols; ++j) {
      mat[i][j] = img_gray.at<uchar>(i, j);
    }
  }
  std::vector<std::vector<int>> sobel_filter_x = {
    {-1, 0, +1},
    {-2, 0, +2},
    {-1, 0, +1}
  };
  std::vector<std::vector<int>> sobel_filter_y = matrix_op::transpose(sobel_filter_x);
  std::vector<std::vector<int>> gx = matrix_op::convolve(sobel_filter_x, mat);
  std::vector<std::vector<int>> gy = matrix_op::convolve(sobel_filter_y, mat);
  cv::Mat res(img_gray.rows, img_gray.cols, CV_8U);
  for (int i = 1; i + 1 < img_gray.rows; ++i) {
    for (int j = 1; j + 1 < img_gray.cols; ++j) {
      int magnitude = std::sqrt(gx[i][j]*gx[i][j] + gy[i][j]*gy[i][j]);
      int threshold = 70;
      res.at<uchar>(i, j) = (threshold < magnitude) * magnitude;
    }
  }
  std::vector<std::vector<int>> col(res.rows, std::vector<int>(res.cols, -1));
  auto flood = [&](int sx, int sy, int c, std::function<bool(int, int)> pred) {
    std::queue<int> que;
    que.push(sx * res.cols + sy);
    col[sx][sy] = c;
    res.at<uchar>(sx, sy) = c * 10;
    while (que.size()) {
      int x = que.front() / res.cols;
      int y = que.front() % res.cols;
      que.pop();
      for (int d = 0; d < 4; ++d) {
        static const int dx[] = {0, 1, 0, -1};
        static const int dy[] = {1, 0, -1, 0};
        int nx = x + dx[d];
        int ny = y + dy[d];
        if (nx < 0 || nx == res.rows || ny < 0 || ny == res.cols) continue;
        if (~col[nx][ny]) continue;
        if (pred(nx, ny)) continue;
        que.push(nx * res.cols + ny);
        col[nx][ny] = c;
        res.at<uchar>(nx, ny) = 0; // c * 10;
      }
    }
  };
  int col_cnt = 0; // assumes between every piece there is some space, as well as the corner
  flood(0, 0, col_cnt, [&](int x, int y) { return 0 < res.at<uchar>(x, y); });
  for (int i = 0; i < res.rows; ++i) {
    for (int j = 0; j < res.cols; ++j) {
      if (~col[i][j]) continue;
      ++col_cnt;
      flood(i, j, col_cnt, [](int x, int y) { return false; });
    }
  }
  std::vector<std::vector<kika::cod>> col2pos(col_cnt + 1);
  for (int i = 0; i < res.rows; ++i) {
    for (int j = 0; j < res.cols; ++j) {
      if (col[i][j]) {
        col2pos[col[i][j]].emplace_back(i, j);
      }
    }
  }
  for (int c = 1; c < col_cnt + 1; ++c) {
    for (kika::cod p : col2pos[c]) {
      int x = real(p);
      int y = imag(p);
      res.at<uchar>(x, y) = c * 10;
    }
    for (auto p : col2pos[c]) {
      res.at<uchar>(real(p), imag(p)) = 100;
    }
    std::vector<kika::cod> ch = kika::convex_hull(col2pos[c]);
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
          res.at<uchar>(lx + j, ly + t * j) = 200;
        }
      } else {
        double t = 1.0 * dx / dy;
        for (int j = std::min(0, dy); j < std::max(0, dy); ++j) {
          res.at<uchar>(lx + t * j, ly + j) = 200;
        }
      }
    }
    for (auto p : ch) {
      res.at<uchar>(real(p), imag(p)) = 255;
    }
  }
  return res;
  cv::Mat res_masked(res.rows, res.cols, CV_8U);
  for (int c = 1; c < col_cnt + 1; ++c) {
    int mx = Array<kika::cod>(col2pos[c])
      .map([](kika::cod c) { return real(c); })
      .reduce([](Array<double> s, double x) {
        return Array<double>({std::min(s[0], x), std::max(s[1], x)});
      }, Array<double>({std::numeric_limits<double>::max(), 0}))
      .reduce([](double x, double y) { return (x + y) / 2; });
    int my = Array<kika::cod>(col2pos[c])
      .map([](kika::cod c) { return imag(c); })
      .reduce([](Array<double> s, double x) {
        return Array<double>({std::min(s[0], x), std::max(s[1], x)});
      }, Array<double>({std::numeric_limits<double>::max(), 0}))
      .reduce([](double x, double y) { return (x + y) / 2; });
    for (auto p : col2pos[c]) {
      int i = real(p);
      int j = imag(p);
      kika::cod xy = kika::rotate(kika::cod(i - mx, j - my), std::acos(-1) / 8);
      int x = i;//std::real(xy) + 0.5 + mx;
      int y = j;//std::imag(xy) + 0.5 + my;
      res_masked.at<uchar>(x, y) = res.at<uchar>(i, j);
    }
  }
  return res_masked;
}

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: ./solve path_to_image" << std::endl;
    return 0;
  }

  cv::Mat img = cv::imread(argv[1]);
  cv::Mat img_pieces = removeBg(img);

  cv::imwrite("pieces.png", img_pieces);

  cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); 
  cv::imshow("Display window", img_pieces);  
  cv::waitKey(0);

  return 0;
}
