/**
 * @file quick_sort.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2026-05-27
 *
 * Copyright © 2026 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <functional>
#include <iterator>
#include <utility>

template <typename It, typename Cmp = std::less<
                           std::remove_cvref_t<decltype(*std::declval<It>())>>>
void QuickSort(It begin, It end, Cmp cmp = Cmp{}) {
  if (begin == end) {
    return;
  }
  It left = begin, right = std::prev(end);
  if (left == right) {
    return;
  }
  left = std::next(left);
  const auto &pivot_value = *begin;
  while (left != right) {
    // First find the rightest position on the right
    // where the value at this position is less than the pivot,
    // which means that it should be moved to left.
    while (right != left && !cmp(*right, pivot_value)) {
      // not *right < pivot_value
      // -> *right >= pivot_value
      right = std::prev(right);
    }
    // *right < pivot_value

    // Then find the leftest position on the left
    // where the value at this position is greater than the pivot,
    // which means that is should be moved to right.
    while (left != right && !cmp(pivot_value, *left)) {
      // not pivot_value < *left
      // -> *left <= pivot_value
      left = std::next(left);
    }
    // *left > pivot_value
    if (left != right) {
      std::swap(*left, *right);
    }
  }
  It pivot_it = left;
  if (cmp(pivot_value, *pivot_it)) {
    pivot_it = std::prev(pivot_it);
  }
  std::swap(*begin, *pivot_it);
  QuickSort(begin, pivot_it, cmp);
  QuickSort(std::next(pivot_it), end, cmp);
}

#include <iostream>
#include <vector>

int main() {
  std::vector<int> v{1, 2, 3, 4, 5};
  QuickSort(v.begin(), v.end(), std::greater<int>{});
  for (auto &&e : v) {
    std::cout << e << ' ';
  }
  std::cout << '\n';
}
