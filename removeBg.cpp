#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include "opencv2/opencv.hpp"

template<typename T>
std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>> &a) {
  std::vector<std::vector<T>> r(a[0].size(), std::vector<T>(a.size()));
  for (int i = 0; i < a.size(); ++i) {
    for (int j = 0; j < a[0].size(); ++j) {
      r[j][i] = a[i][j];
    }
  }
  return r;
}

template<typename T>
std::vector<std::vector<T>> convolve(
  const std::vector<std::vector<T>> &f,
  const std::vector<std::vector<T>> &a
) {
  assert(f.size() == f[0].size());
  assert(f.size() & 1);
  std::vector<std::vector<T>> r(a.size(), std::vector<T>(a[0].size()));
  for (int i = 0; i < a.size(); ++i) {
    for (int j = 0; j < a[0].size(); ++j) {
      int m = f.size() / 2;
      for (int x = -m; x < m + 1; ++x) {
        for (int y = -m; y < m + 1; ++y) {
          int s = i - x;
          int t = j - y;
          if (0 <= s && s < a.size() && 0 <= t && t < a[0].size()) {
            r[i][j] += f[m + x][m + y] * a[s][t];
          }
        }
      }
    }
  }
  return r;
}

cv::Mat removeBg(const cv::Mat &img_gray) {
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
  std::vector<std::vector<int>> sobel_filter_y = transpose(sobel_filter_x);
  std::vector<std::vector<int>> gx = convolve(sobel_filter_x, mat);
  std::vector<std::vector<int>> gy = convolve(sobel_filter_y, mat);
  cv::Mat res(img_gray.rows, img_gray.cols, CV_8U);
  for (int i = 1; i + 1 < img_gray.rows; ++i) {
    for (int j = 1; j + 1 < img_gray.cols; ++j) {
      int magnitude = std::sqrt(gx[i][j]*gx[i][j] + gy[i][j]*gy[i][j]);
      int threshold = 70;
      res.at<uchar>(i, j) = (threshold < magnitude) * magnitude;
    }
  }
  return res;
}

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: ./solve path_to_image" << std::endl;
    return 0;
  }

  cv::Mat img_gray = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

  cv::Mat pieces = removeBg(img_gray);

  cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); 
  cv::imshow("Display window", pieces);  
  cv::waitKey(0);

  return 0;
}
