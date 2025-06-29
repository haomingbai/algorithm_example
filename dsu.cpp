/**
 * @file dsu.cpp
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
#include <vector>

class dsu {
  std::vector<std::size_t> parent;

 public:
  dsu(std::size_t size) : parent(size) {
    for (auto i = 0uz; i < parent.size(); i++) {
      parent[i] = i;
    }
  }

  std::size_t findRoot(std::size_t idx) {
    if (parent[idx] == idx) {
      return idx;
    }

    std::size_t res = idx;
    while (res != parent[res]) {
      res = parent[res];
    }

    std::size_t curr = idx;
    while (parent[curr] != res) {
      auto to_modify = curr;
      curr = parent[to_modify];
      parent[to_modify] = res;
    }

    return res;
  }

  void unite(std::size_t idx1, std::size_t idx2) {
    parent[findRoot(idx2)] = findRoot(idx1);
  }

  bool inSameTree(std::size_t idx1, std::size_t idx2) {
    return findRoot(idx1) == findRoot(idx2);
  }
};
