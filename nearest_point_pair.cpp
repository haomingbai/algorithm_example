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
#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <vector>

#include "./concepts.cpp"

template <Multiplyable T>
struct Point2D {
  double x, y;  // 保持你原来的设计不变（模板参数存在但成员为 double）
};

// 保持签名但改为用 T 做整数运算（将坐标 cast 为 T）
// 假设你已经声明输入为整数且平方不会溢出（按你要求）
template <Multiplyable T>
T DistanceSquareWith(const Point2D<T>& a, const Point2D<T>& b) {
  T dx = static_cast<T>(a.x) - static_cast<T>(b.x);
  T dy = static_cast<T>(a.y) - static_cast<T>(b.y);
  return dx * dx + dy * dy;
}

// 保持外部签名不变：输入是按 x 排序好的点（你的 main 已经做了 sort）
template <Multiplyable T>
T FindNearestDistanceSquare(std::span<Point2D<T>> point_list) {
  const size_t n = point_list.size();
  if (n < 2) return std::numeric_limits<T>::max();
  if (n == 2) return DistanceSquareWith(point_list.front(), point_list.back());

  // 为递归使用，直接操作底层数据指针，避免频繁拷贝 span 对象
  Point2D<T>* base = point_list.data();

  // 递归 lambda，半开区间 [l, r)
  std::function<T(size_t, size_t)> rec;
  rec = [&](size_t l, size_t r) -> T {
    size_t len = r - l;
    if (len < 2) {
      return std::numeric_limits<T>::max();
    }
    if (len == 2) {
      // 两点时按原样比较（不改变顺序）
      return DistanceSquareWith(base[l], base[l + 1]);
    }

    size_t m = l + (len >> 1);
    // 保存分割线 x 坐标。**注意**：必须在递归之前或基于当前 x 排序的假设确定
    // mid_x。 这里我们使用分割点为右半部分首元素的
    // x（常见做法，避免浮点平均）。
    T mid_x = static_cast<T>(base[m].x);

    // 递归求左右最短
    T dl = rec(l, m);
    T dr = rec(m, r);
    T d = dl < dr ? dl : dr;

    // 找到横向候选区间：从中间向两边线性扩展直到 dx^2 >= d
    // （比起二分查找，这里更简单、分支更少，且通常很快——因为带宽一般较小）
    size_t left_edge = m;  // inclusive
    if (m > l) {
      for (ptrdiff_t i = static_cast<ptrdiff_t>(m) - 1;
           i >= static_cast<ptrdiff_t>(l); --i) {
        T dx = static_cast<T>(base[i].x) - mid_x;
        if (dx * dx < d)
          left_edge = static_cast<size_t>(i);
        else
          break;
      }
    }

    size_t right_edge = m;  // exclusive
    for (size_t j = m; j < r; ++j) {
      T dx = static_cast<T>(base[j].x) - mid_x;
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
      strip.push_back(&base[idx]);

    std::sort(strip.begin(), strip.end(),
              [](const Point2D<T>* a, const Point2D<T>* b) {
                if (a->y != b->y) return a->y < b->y;
                return a->x < b->x;
              });

    // 经典 strip 比较：对每个点只检查 y 差绝对值小于 sqrt(d) 的后续点
    // 这里比较使用 dy^2 >= d 的早停条件（避免 sqrt）
    for (size_t i = 0; i < strip.size(); ++i) {
      for (size_t j = i + 1; j < strip.size(); ++j) {
        T dy = static_cast<T>(strip[j]->y) - static_cast<T>(strip[i]->y);
        if (dy * dy >= d) break;  // y 差已足够大，后面不用再看
        // 计算完整平方距离
        T cur = DistanceSquareWith(*strip[i], *strip[j]);
        if (cur < d) d = cur;
      }
    }

    return d;
  };

  return rec(0, n);
}
