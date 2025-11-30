/**
 * @file convex_diameter_square.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-11-19
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <span>
#include <vector>

template <typename T>
using Point2D = std::complex<T>;

template <typename T>
using Vector2D = std::complex<T>;

template <typename T>
using Convex2D = std::span<Point2D<T>>;

template <typename T>
T CrossProductValue(const Vector2D<T>& v1, const Vector2D<T>& v2) {
  return v1.real() * v2.imag() - v1.imag() * v2.real();
}

// --- 经典旋转卡壳，计算凸包直径的平方（template 版本，使用整数运算） ---
template <typename T>
T ConvexDiameterSquare(const Convex2D<T>& convex) {
  const size_t n = convex.size();
  if (n < 2) return T(0);
  if (n == 2) return std::norm(convex[0] - convex[1]);

  size_t j = 1;
  T best = 0;
  for (size_t i = 0; i < n; ++i) {
    size_t ni = (i + 1) % n;
    // 向前推进 j，直到 abs(cross(edge_i, p_{j+1})) <= abs(cross(edge_i, p_j))
    while (true) {
      size_t nj = (j + 1) % n;
      T cross_curr =
          CrossProductValue(convex[ni] - convex[i], convex[j] - convex[i]);
      T cross_next =
          CrossProductValue(convex[ni] - convex[i], convex[nj] - convex[i]);
      // 取绝对值（适用于整数或浮点）
      if (cross_curr < T(0)) cross_curr = -cross_curr;
      if (cross_next < T(0)) cross_next = -cross_next;
      if (cross_next > cross_curr) {
        j = nj;
      } else {
        break;
      }
    }
    // 更新候选直径（i, j) 与 (ni, j)
    T d = std::norm(convex[i] - convex[j]);
    if (d > best) best = d;
    d = std::norm(convex[ni] - convex[j]);
    if (d > best) best = d;
  }
  return best;
}

int main() {
  size_t N;
  if (!(std::cin >> N)) return 0;
  std::vector<Point2D<long long>> pts;
  pts.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    long long x, y;
    std::cin >> x >> y;
    pts.emplace_back(Point2D<long long>(x, y));
  }

  // 计算凸包（CCW）
  pts = ConvexHull(pts.begin(), pts.end());
  // 计算直径平方
  auto d2 = ConvexDiameterSquare(Convex2D<long long>(pts));
  std::cout << d2 << '\n';
  return 0;
}
