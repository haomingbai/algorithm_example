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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

struct Point2D {
  double x, y;
};

double DistanceWith(const Point2D &a, const Point2D &b) {
  return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

double FindNearestDistance(std::span<Point2D> point_list) {
  if (point_list.size() == 2) {
    return DistanceWith(point_list.front(), point_list.back());
  } else if (point_list.size() < 2) {
    return MAXFLOAT;
  }

  auto left_span = point_list.subspan(0, point_list.size() / 2);
  auto right_span = point_list.subspan(point_list.size() / 2);

  double distance_left = FindNearestDistance(left_span);
  double distance_right = FindNearestDistance(right_span);

  auto dist_to_cmp = std::min(distance_left, distance_right);

  auto mid_x = (point_list[point_list.size() / 2 - 1].x +
                point_list[point_list.size() / 2].x) /
               2;
  size_t left_edge = SIZE_MAX, right_edge = SIZE_MAX;
  size_t left_mid_edge = left_span.size() - 1;
  size_t right_mid_edge = point_list.size() / 2;

  {
    for (ptrdiff_t i = left_span.size() - 1; i >= 0; i--) {
      if (std::abs(left_span[i].x - mid_x) < dist_to_cmp) {
        left_edge = i;
      } else {
        break;
      }
    }

    for (long i = 0; i < right_span.size(); i++) {
      if (std::abs(right_span[i].x - mid_x) < dist_to_cmp) {
        right_edge = i + right_mid_edge;
      } else {
        break;
      }
    }
  }

  if (left_edge == SIZE_MAX || right_edge == SIZE_MAX) {
    return dist_to_cmp;
  }

  for (auto i = left_edge; i <= left_mid_edge; i++) {
    for (auto j = right_mid_edge; j <= right_edge; j++) {
      dist_to_cmp =
          std::min(dist_to_cmp, DistanceWith(point_list[i], point_list[j]));
    }
  }

  return dist_to_cmp;
}
