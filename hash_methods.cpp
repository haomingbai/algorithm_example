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

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
  std::vector<size_t> hash_val_;

 public:
  template <typename Container>
    requires(RandomAccessContainer<Container, size_t>)
  Descretization(const Container &arr) : hash_val_(arr.size()) {
    for (size_t i = 0; i < arr.size(); i++) {
      hash_val_[i] = arr[i];
    }

    // 排序, 用下标作为哈希.
    std::sort(hash_val_.begin(), hash_val_.end());
    // 去重.
    hash_val_.erase(std::unique(hash_val_.begin(), hash_val_.end()),
                    hash_val_.end());
  }

  // 获得hash
  size_t hash(size_t val) {
    // 用lower_bound查找第一个大于或等于某个元素的位置
    auto pos = std::lower_bound(hash_val_.begin(), hash_val_.end(), val);

    // 没找到返回...
    if (*pos > val) {
      return SIZE_MAX;
    }

    // 找到等于了...
    return pos - hash_val_.begin();
  }

  // 从hash获取原元素值.
  size_t dehash(size_t hash) {
    if (hash >= hash_val_.size()) {
      return SIZE_MAX;
    }

    return hash_val_[hash];
  }

  // 获取当前离散化的最大和最小散列值.
  size_t getMaxHash() {
    return hash_val_.size() > 0 ? hash_val_.size() - 1 : 0;
  }
  size_t getMinHash() { return 0; }
};
