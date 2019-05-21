#include <iostream>
#include <vector>
#include <map>
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
#include "src/array.hpp"
#include "src/kika.hpp"

void flood(
  int R, int C, // #row, #col
  int sx, int sy, int c, // start x, y, color to fill
  std::function<bool(int, int)> invalid, // (x, y).invalid?
  Array<Array<int>> &col // to color on, initialize with 0
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

Array<Array<kika::cod>> extractPieces(const cv::Mat &img) {
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

  Array<Array<int>> mat(img_gray.rows, Array<int>(img_gray.cols));
  for (int i = 0; i < img_gray.rows; ++i) {
    for (int j = 0; j < img_gray.cols; ++j) {
      mat[i][j] = img_gray.at<uchar>(i, j);
    }
  }
  Array<Array<int>> sobel_filter_x = {
    {-1, 0, +1},
    {-2, 0, +2},
    {-1, 0, +1}
  };
  Array<Array<int>> sobel_filter_y = sobel_filter_x.transpose();
  Array<Array<int>> gx = mat.convolve(sobel_filter_x);
  Array<Array<int>> gy = mat.convolve(sobel_filter_y);
  cv::Mat res(img.rows, img.cols, CV_8U);
  for (int i = 1; i + 1 < img.rows; ++i) {
    for (int j = 1; j + 1 < img.cols; ++j) {
      int magnitude = std::sqrt(gx[i][j]*gx[i][j] + gy[i][j]*gy[i][j]);
      int threshold = 70;
      res.at<uchar>(i, j) = (threshold < magnitude) * magnitude;
    }
  }
  int col_cnt = 0; // assumes between every piece there is some space, as well as the corner
  Array<Array<int>> col(res.rows, Array<int>(res.cols, -1));
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
  Array<Array<kika::cod>> col2pos(col_cnt + 1);
  for (int i = 0; i < res.rows; ++i) {
    for (int j = 0; j < res.cols; ++j) {
      if (col[i][j]) {
        col2pos[col[i][j]].push_back(kika::cod(i, j));
      }
    }
  }
  return col2pos.slice(1);
}

void render(cv::Mat &mat, const Array<kika::cod> &pnts) {
  for (auto pnt : pnts) {
    int x = real(pnt);
    int y = imag(pnt);
    mat.at<uchar>(x, y) = 255;
  }
}

void render(cv::Mat &mat, const Array<Array<int>> &col) {
  for (int i = 0; i < col.size(); ++i) {
    for (int j = 0; j < col[0].size(); ++j) {
      mat.at<uchar>(i, j) = col[i][j];
    }
  }
}

Array<Array<int>> detectPieceFeatures(int R, int C, Array<kika::cod> pnts) {
  Array<Array<int>> res(R, Array<int>(C));
  for (const kika::cod &pnt : pnts) {
    int x = real(pnt);
    int y = imag(pnt);
    res[x][y] = 100;
  }
  Array<kika::cod> ch = kika::convex_hull(pnts);
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
        res[lx + j][ly + t * j] = 200;
      }
    } else {
      double t = 1.0 * dx / dy;
      for (int j = std::min(0, dy); j < std::max(0, dy); ++j) {
        res[lx + t * j][ly + j] = 200;
      }
    }
  }
  for (auto p : ch) {
    res[std::real(p)][std::imag(p)] = 255;
  }
  return res;
  /* cv::Mat res_masked(res.rows, res.cols, CV_8U); */
  /* for (int c = 1; c < col_cnt + 1; ++c) { */
  /*   int mx = Array<kika::cod>(col2pos[c]) */
  /*     .map([](kika::cod c) { return std::real(c); }) */
  /*     .reduce([](Array<double> s, double x) { */
  /*       return Array<double>({std::min(s[0], x), std::max(s[1], x)}); */
  /*     }, Array<double>({std::numeric_limits<double>::max(), 0})) */
  /*     .reduce([](double x, double y) { return (x + y) / 2; }); */
  /*   int my = Array<kika::cod>(col2pos[c]) */
  /*     .map([](kika::cod c) { return std::imag(c); }) */
  /*     .reduce([](Array<double> s, double x) { */
  /*       return Array<double>({std::min(s[0], x), std::max(s[1], x)}); */
  /*     }, Array<double>({std::numeric_limits<double>::max(), 0})) */
  /*     .reduce([](double x, double y) { return (x + y) / 2; }); */
  /*   for (auto p : col2pos[c]) { */
  /*     int i = real(p); */
  /*     int j = imag(p); */
  /*     kika::cod xy = kika::rotate(kika::cod(i - mx, j - my), std::acos(-1) / 8); */
  /*     int x = i;//std::real(xy) + 0.5 + mx; */
  /*     int y = j;//std::imag(xy) + 0.5 + my; */
  /*     res_masked.at<uchar>(x, y) = res.at<uchar>(i, j); */
  /*   } */
  /* } */
  /* return res_masked; */
}

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: ./solve path_to_image" << std::endl;
    return 0;
  }

  cv::Mat img = cv::imread(argv[1]);

  Array<Array<kika::cod>> pieces = extractPieces(img);

  Array<Array<int>> col = detectPieceFeatures(img.rows, img.cols, pieces[0]);
  cv::Mat img_p0(img.rows, img.cols, CV_8U, cv::Scalar(0, 0, 0));
  render(img_p0, col);

  cv::imwrite("pieces.png", img_p0);
  cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); 
  cv::imshow("Display window", img_p0);  
  cv::waitKey(0);

  return 0;
}
