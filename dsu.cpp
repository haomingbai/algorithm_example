/**
 * @file DSU.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-06-26
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

class DSU {
  std::vector<std::size_t> parent_, size_;

 public:
  DSU(std::size_t size) : parent_(size), size_(size, 1) {
    for (auto i = 0ul; i < parent_.size(); i++) {
      parent_[i] = i;
    }
  }

  std::size_t findRoot(std::size_t idx) {
    // 递归终止条件, 寻找到根或者当前位置高度为1(根为0).
    if (parent_[idx] == idx || parent_[parent_[idx]] == parent_[idx]) {
      return parent_[idx];
    }

    parent_[idx] = findRoot(parent_[idx]);
    return parent_[idx];
  }

  void unite(std::size_t idx1, std::size_t idx2) {
    // 先找到根节点, 因为只有根节点维护了大小数据.
    idx1 = findRoot(idx1);
    idx2 = findRoot(idx2);

    // 如果二者根相同, 那么二者已经在同一集合.
    if (idx1 == idx2) {
      return;
    }

    // 因为要把2挂靠到1上, 所以下标1必须是较大的.
    if (size_[idx1] < size_[idx2]) {
      std::swap(idx1, idx2);
    }

    // 将2挂靠到1
    parent_[idx2] = idx1;
    // 此时1的树根是二者的共同树根, 所以只有1需要维护树大小.
    // 1的树大小是原先二者树大小之和, 因为2树和1树合并了.
    size_[idx1] += size_[idx2];
    return;
  }

  bool inSameSet(std::size_t idx1, std::size_t idx2) {
    return findRoot(idx1) == findRoot(idx2);
  }
};
