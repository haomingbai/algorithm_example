/**
 * @file convex_hull.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-11-16
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <algorithm>
#include <complex>
#include <iterator>
#include <utility>
#include <vector>

template <typename T>
using Point2D = std::complex<T>;

template <typename T>
using Vector2D = std::complex<T>;

template <typename T>
auto CrossProductValue(const Vector2D<T>& v1, const Vector2D<T>& v2) {
  return v1.real() * v2.imag() - v1.imag() * v2.real();
}

// --- 使用标准 Andrew 单调链构造凸包（CCW，不含重复端点） ---
template <typename I, typename P = std::iter_value_t<I>,
          typename T = decltype(std::declval<P>().real())>
std::vector<Point2D<T>> ConvexHull(I begin, I end) {
  std::vector<Point2D<T>> pts(begin, end);
  if (pts.size() <= 1) return pts;

  std::sort(pts.begin(), pts.end(),
            [](const Point2D<T>& a, const Point2D<T>& b) {
              if (a.real() != b.real()) return a.real() < b.real();
              return a.imag() < b.imag();
            });

  pts.erase(std::unique(pts.begin(), pts.end(),
                        [](const Point2D<T>& a, const Point2D<T>& b) {
                          return a.real() == b.real() && a.imag() == b.imag();
                        }),
            pts.end());

  if (pts.size() <= 1) return pts;

  std::vector<Point2D<T>> lower, upper;
  for (const auto& p : pts) {
    while (lower.size() >= 2) {
      auto v1 = lower[lower.size() - 1] - lower[lower.size() - 2];
      auto v2 = p - lower[lower.size() - 1];
      auto cr = CrossProductValue(v1, v2);
      if (cr <= T(0))
        lower.pop_back();
      else
        break;
    }
    lower.push_back(p);
  }
  for (auto it = pts.rbegin(); it != pts.rend(); ++it) {
    const auto& p = *it;
    while (upper.size() >= 2) {
      auto v1 = upper[upper.size() - 1] - upper[upper.size() - 2];
      auto v2 = p - upper[upper.size() - 1];
      auto cr = CrossProductValue(v1, v2);
      if (cr <= T(0))
        upper.pop_back();
      else
        break;
    }
    upper.push_back(p);
  }

  // 合并 lower + upper，去掉重复的端点
  lower.pop_back();
  upper.pop_back();
  lower.insert(lower.end(), upper.begin(), upper.end());
  return lower;  // CCW order, no duplicate endpoints
}
