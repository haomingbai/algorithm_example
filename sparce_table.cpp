/**
 * @file st_table.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-07-06
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <ostream>
#include <vector>

#include "concepts.cpp"

/*
 * Sparse Table (ST表) 简介：
 * ST表是一种静态区间查询的数据结构，适用于可交换、可复合的区间操作（如最值、gcd等）。
 * 它通过预处理 f[i][k] 表示以 i 为起点、长度为 2^k 的区间结果，
 * 预处理时间 O(N log N)，查询时间 O(1)，空间 O(N log N)。
 *
 * 示例结构（N=8, 最大 k=3）：
 *            k=0   k=1   k=2   k=3
 *        -----------------------------
 * f[i][k]:  a[0] | min(a[0],a[1]) | ... | ...
 * i = 0    a[0]   a[0,1]           a[0,3]   a[0,7]
 * i = 1    a[1]   a[1,2]           a[1,3]   -
 * i = 2    a[2]   a[2,3]           -        -
 * ...
 *
 * 查询区间 [L,R] 时令 len = R-L+1, k = floor(log2(len))，
 * 则答案 = combine(f[L][k], f[R-2^k+1][k])。
 */

template <typename T>
class SparseTable {
 protected:
  using VT = std::vector<T>;
  using VVT = std::vector<VT>;
  using FuncType = std::function<T(T, T)>;

  /// 数据存放在一个数组里,
  /// 二维数组的第i行代表从下标i开始的长度为2^j子数组的运算的累加.
  VVT data;

  /// 运算的内容, 默认是max运算.
  FuncType func;

  /// 原数组的大小
  size_t size;

  /// 取对数之后的大小, 这个数太关键了, 所以单列.
  size_t log_size;

 public:
  /// 只解释这一个构造, 另一个就换了一个运算符就不解释了.
  template <typename Container>
    requires(RandomAccessContainer<Container, T>)
  SparseTable(const Container &arr)
      :  /// 原数组有多少个元素就应该有多少行,
         /// 毕竟一个区间可能会从任何一个元素的位置开始.
        data(arr.size()),
        /// 默认取得求最大运算, 一定要是满足结合率和可重复贡献这两个条件的.
        func([](T a, T b) { return std::max(a, b); }),
        /// 原数组的大小, 单独维护.
        size(arr.size()) {
    /// 数组的大小取得对数之后的大小, 这里标志着每一行最多会有多少元素.
    log_size = std::ceil(std::log2(size));

    /// 因为运算满足func(x, x) = x, 因此data[i][0]表示[i,i]范围内的运算累加,
    /// 结果就是x.
    for (size_t i = 0; i < size; i++) {
      size_t curr_arr_size = log_size + 1;
      data[i].resize(curr_arr_size);
      data[i][0] = arr[i];
    }

    /// 按照列进行遍历, 因为已经单独处理了第0列了, 所以我们从第1列开始.
    for (size_t i = 1; i <= log_size; i++) {
      /// 这里主要是找到列的最大长度.
      /// 注意最大长度不能直接算, 我们的对数运算并不是最精确,
      /// 因此有可能多减一点下标就小于0, 然后无符号整数就循环了.
      size_t unused_area = (1 << (i));  /// (col_max_idx - 1) + 2 ^ i < size
      for (size_t j = 0; unused_area + j <= size; j++) {
        data[j][i] =
            func(data[j][i - 1]  /// 当前行的上一个元素
                 ,
                 data[j + (1 << (i - 1))]
                     [i - 1]  /// 和上一个元素管辖区间不相交的,
                              /// 辖域长度和上一个元素的管辖区间相同的元素.
                              /// 就例如辖域为[0, 1]和[2, 3]的两个元素.
            );
      }
    }
  }

  template <typename Container>
    requires(RandomAccessContainer<Container, T>)
  SparseTable(const Container &arr, FuncType func)
      : data(arr.size()), func(func), size(arr.size()) {
    log_size = std::ceil(std::log2(size));

    for (size_t i = 0; i < size; i++) {
      size_t curr_arr_size = log_size + 1;
      data[i].resize(curr_arr_size);
      data[i][0] = arr[i];
    }

    for (size_t i = 1; i <= log_size; i++) {
      size_t unused_area = (1 << (i));  /// (col_max_idx - 1) + 2 ^ i < size
      for (size_t j = 0; unused_area + j <= size; j++) {
        data[j][i] =
            func(data[j][i - 1]  /// 当前行的上一个元素
                 ,
                 data[j + (1 << (i - 1))]
                     [i - 1]  /// 和上一个元素管辖区间不相交的,
                              /// 辖域长度和上一个元素的管辖区间相同的元素.
                              /// 就例如辖域为[0, 1]和[2, 3]的两个元素.
            );
      }
    }
  }

  T query(size_t left, size_t right) {
    auto range_length = right - left + 1;
    size_t max_exist_range_log = std::floor(std::log2(range_length));

    /// 查找就是前后两段最长的且长度为2的幂次的.
    /// [        ]
    ///       [        ]
    /// 就类似上方的效果.
    auto res =
        func(data[left][max_exist_range_log],
             data[right - (1 << max_exist_range_log) + 1][max_exist_range_log]);
    return res;
  }
};
