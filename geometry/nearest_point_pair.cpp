/**
 * @file nearest_point_pair.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-09-16
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <algorithm>
#include <complex>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

template <typename T>
using Point2D = std::complex<T>;

template <typename T>
T FindNearestDistanceSquare(std::span<Point2D<T>> point_list) {
  const size_t n = point_list.size();
  if (n < 2) return std::numeric_limits<T>::max();
  if (n == 2) return std::norm(point_list.front(), point_list.back());

  // 先对点进行排序.
  std::ranges::sort(point_list, [](const auto& a, const auto& b) {
    return a.real() < b.real();
  });

  // 递归 lambda，半开区间 [l, r)
  auto rec = [&point_list](auto self, size_t l, size_t r) -> T {
    size_t len = r - l;
    if (len < 2) {
      return std::numeric_limits<T>::max();
    }
    if (len == 2) {
      // 两点时按原样比较（不改变顺序）
      return std::norm(point_list[l], point_list[l + 1]);
    }

    size_t m = l + (len / 2);
    // 保存分割线 x 坐标。**注意**：必须在递归之前或基于当前 x 排序的假设确定
    // mid_x。 这里我们使用分割点为右半部分首元素的
    // x（常见做法，避免浮点平均）。
    T mid_x = static_cast<T>(point_list[m].real());

    // 递归求左右最短
    // 这里的dl, dr, d具有一定误导性, 其实指的是d^2
    T dl = self(self, l, m);
    T dr = self(self, m, r);
    T d = std::min(dl, dr);

    // 找到横向候选区间：从中间向两边线性扩展直到 dx^2 >= d
    // （比起二分查找，这里更简单、分支更少，且通常很快——因为带宽一般较小）
    size_t left_edge = m;  // inclusive
    if (m > l) {
      for (ptrdiff_t i = static_cast<ptrdiff_t>(m) - 1;
           i >= static_cast<ptrdiff_t>(l); --i) {
        T dx = static_cast<T>(point_list[i].real()) - mid_x;
        if (dx * dx < d)
          left_edge = static_cast<size_t>(i);
        else
          break;
      }
    }

    size_t right_edge = m;  // exclusive
    for (size_t j = m; j < r; ++j) {
      T dx = static_cast<T>(point_list[j].real()) - mid_x;
      if (dx * dx < d)
        right_edge = j + 1;  // j included
      else
        break;
    }

    // 如果没有跨中线的候选点，直接返回
    if (left_edge >= right_edge) return d;

    // 把候选点放到临时数组并按 y 排序（这样在 strip 内可以早停）
    std::vector<Point2D<T>*> strip;
    strip.reserve(right_edge - left_edge);
    for (size_t idx = left_edge; idx < right_edge; ++idx)
      strip.push_back(&point_list[idx]);

    // 这里先按照y排序, 再按照x排序.
    // 这是一个非常好的优化点.
    // 只要a, b之间y的差值足够大, 依然可以break剪枝.
    std::sort(strip.begin(), strip.end(),
              [](const Point2D<T>* a, const Point2D<T>* b) {
                if (a->imag() != b->imag()) return a->imag() < b->imag();
                return a->real() < b->real();
              });

    // 经典 strip 比较：对每个点只检查 y 差绝对值小于 sqrt(d) 的后续点
    // 这里比较使用 dy^2 >= d 的早停条件（避免 sqrt）
    for (size_t i = 0; i < strip.size(); ++i) {
      for (size_t j = i + 1; j < strip.size(); ++j) {
        T dy =
            static_cast<T>(strip[j]->imag()) - static_cast<T>(strip[i]->imag());
        if (dy * dy >= d) break;  // y 差已足够大，后面不用再看
        // 计算完整平方距离
        T cur = std::norm(*strip[i], *strip[j]);
        d = std::min(cur, d);
      }
    }

    return d;
  };

  return rec(rec, 0, n);
}
