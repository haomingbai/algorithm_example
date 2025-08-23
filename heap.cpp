/**
 * @file heap.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-08-22
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <cassert>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

template <typename DT, typename Compare = std::less<DT>>
struct Heap {
  std::vector<DT> data_;
  Compare comp;

  Heap() : data_(1), comp() {}

  void push(const DT &elem) {
    data_.push_back(elem);

    for (auto curr_idx = data_.size() - 1; curr_idx > 1;) {
      auto parent_idx = curr_idx / 2;

      if (comp(data_[parent_idx], data_[curr_idx])) {
        std::swap(data_[parent_idx], data_[curr_idx]);
        curr_idx = parent_idx;
      } else {
        break;
      }
    }
  }

  void pop() {
    std::swap(data_[1], data_.back());
    data_.pop_back();

    for (size_t curr_idx = 1; curr_idx < data_.size();) {
      auto left_idx = curr_idx * 2;
      auto right_idx = curr_idx * 2 + 1;
      size_t child_idx = 0;

      if (left_idx < data_.size()) {
        child_idx = left_idx;
      }
      if (right_idx < data_.size() && comp(data_[left_idx], data_[right_idx])) {
        child_idx = right_idx;
      }

      if (child_idx && comp(data_[curr_idx], data_[child_idx])) {
        std::swap(data_[curr_idx], data_[child_idx]);
        curr_idx = child_idx;
      } else {
        break;
      }
    }
  }

  const DT &top() { return data_[1]; }

  bool empty() { return data_.size() <= 1; }

  size_t size() { return data_.size() - 1; }
};
