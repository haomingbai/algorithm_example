/**
 * @file hash_methods.cpp
 * @brief 
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-07-04
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <cstddef>
#include <map>
#include <vector>

size_t idx_table[(size_t)2e5 + 1];
std::map<size_t, size_t> value_table;

void createHash(std::vector<size_t> &arr) {
  for (auto &it : arr) {
    value_table[it] = 0;
  }

  size_t cnt = 1;
  for (auto &it : value_table) {
    idx_table[cnt] = it.first;
    it.second = cnt;
    cnt++;
  }
}

size_t descretization(size_t ori) { return value_table[ori]; }

size_t deDescretization(size_t idx) { return idx_table[idx]; }

