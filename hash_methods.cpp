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

#include "concepts.cpp"

/**
 * 离散化（Discretization）: 将大范围稀疏数据映射为紧凑连续下标
 *
 * 适用场景:
 *   - 原始数据范围大 (如 [-1e9, 1e9])
 *   - 实际数据点少 (如 n=1000)
 *
 * 步骤:
 *   1. 收集所有待离散化值
 *   2. 排序 + 去重
 *   3. 二分查找建立映射: 原值 → 新下标
 *
 * 示意图:
 *  原始数据: [100, -5, 7, 1e9, -5, 100]  // 范围大且重复
 *            ↓ 收集去重
 *  去重集合: [-5, 7, 100, 1000000000]
 *            ↓ 分配下标
 *  映射关系:
 *      -5  → 0
 *       7  → 1
 *     100  → 2
 *  1e9  → 3
 *
 * 效果:
 *   原始数组 [100, -5, 7, 1e9]
 *   → 离散化 [2, 0, 1, 3]  // 紧凑下标(0-indexed)
 *
 * 优势:
 *   - 将稀疏数据转为密集索引
 *   - 可用数组/线段树等连续结构处理
 *   - 降低空间复杂度
 */

class Descretization {
  std::vector<size_t> hash_val;
  std::map<size_t, size_t> val_hash;

 public:
  template <typename Container>
    requires(RandomAccessContainer<Container, size_t>)
  Descretization(const Container &arr) {
    for (auto &it : arr) {
      val_hash[it] = 0;
    }

    hash_val.reserve(val_hash.size() + 1);
    hash_val.push_back(0);

    size_t cnt = 1;
    for (auto &it : val_hash) {
      it.second = cnt;
      hash_val.push_back(it.first);
      cnt++;
    }
  }

  size_t hash(size_t val) { return val_hash[val]; }

  size_t dehash(size_t hash) { return hash_val[hash]; }

  size_t getMaxHash() {
    return val_hash.rbegin()->second;
  }

  size_t getMinHash() {
    return val_hash.begin()->second;
  }
};
