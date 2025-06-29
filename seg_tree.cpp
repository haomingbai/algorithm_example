/**
 * @file seg_tree.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-06-29
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

// 可以支持构造和查询的最朴素的线段树.
template <typename T>
class SimpleSegTree {
  std::vector<T> data;  // 数据数组,  大小之后会讲
  std::size_t size;     // 线段树存放的数组的大小

  std::function<T(T, T)> op;  // 线段树所代表的运算操作,  常见的有加法，乘法这类
  T one;                      // 幺元

  // 私有的建树操作
  auto build(size_t left, size_t right, size_t curr_root,
             const std::vector<T> &vec) -> void {
    // `mid` 表示中间位置的下标, 这里 `left` 和 `right`
    // 代表当前建树的区间的左端点和右端点 当前准备build的部分为原数组的[left,
    // right]的部分. `mid` 变量将这个区间一分为二, 分为[left, mid]和[mid + 1,
    // right].
    // right - left >= 1, 即区间长度至少为 2 的时候, 这个划分一直有效, 当right -
    // left = 0, 这个时候我们遇到了树的叶子节点, 我们接下来单独讨论
    auto mid = (left + right) / 2;

    // 这里我们讨论的是树的叶子节点. `curr_root` 参数这里代表的是[left, right]
    // 区间内, `vec` 内部元素的和 (也就是 vec[left] + vec[left + 1] + ... +
    // vec[right]) 应当存放的位置. 也就是说, vec[left] + vec[left + 1] + ... +
    // vec[right] = data[curr_root].
    // 当区间的长度就是 1 的时候, 那这个区间也没有办法被拆分为左右两个了,
    // 和就直接放在相应位置了
    if (left == right) {
      data[curr_root] = vec[mid];
      return;
    }

    // 这里我们获得的是线段树的两个孩子的分支, 如果以 `curr_root` 作为根,
    // 那么将这个区间拆分为 [left, mid] 和 [mid + 1, right] 这两个子区间,
    // 那么按照线段树的规则, 这两个子区间内的原数组元素的和应当放在两个
    // 当前节点的两个孩子分支内待查, 而这两个分支的位置则分别为:
    // left_idx = curr_root * 2
    // right_idx = curr_root * 2 + 1
    // 也就是说, 这个树的枝叶的位置和 data 数组下标的对应应该是长这样的:
    // 1---2---4---8    # vec[1]
    //   |   |   |
    //   |   |   --9    # vec[2]
    //   |   --5---10   # vec[3]
    //   |       |
    //   |       --11   # vec[4]
    //   --3---6---12   # vec[5]
    //       |   |
    //       |   --13   # vec[6]
    //       |
    //       --7---14   # vec[7]
    //           |
    //           --15   # vec[8]
    // 很明显可以看到, 如果是二的幂次大小的原数组, 那么对应的数据数组的大小
    // 应该为 2n - 1, 至于具体的情况以后会讲
    auto left_branch_idx = curr_root * 2, right_branch_idx = curr_root * 2 + 1;

    // 这里先构造两个子区间的线段树, 将两个子区间的元素的操作的累积(默认是和)
    // 给先计算出来
    build(left, mid, left_branch_idx, vec);
    build(mid + 1, right, right_branch_idx, vec);

    // 最后区间内元素的操作的累积就是两个子区间的累积的累积
    // 就是 a[1] + a[2] + a[3] = (a[1] + a[2]) + (a[3])
    // 的道理.
    data[curr_root] = op(data[left_branch_idx], data[right_branch_idx]);

    // 礼貌起见, 我们写上 return 语句, 当然也可以不写
    return;
  }

  // 私有的查询语句
  // 参数的意义如下:
  // `left` 和 `right` 代表了需要的查询的整个区间
  // `curr_left` 和 `curr_right` 代表本次调用中需要完成的查询区间
  // 也就是说, 这次查询, 需要完成 [left, right] 和
  // [curr_left, curr_right] 的交集部分的操作的累积的查询
  // (就比如说它们的和, 以后相关部分都以求和举例)
  // 以求和为例, 就是要找到原数组中 [left, right] 和
  // [curr_left, curr_right] 的交集部分的元素的和.
  // 这个函数怎么用好一会再说.
  // 这里 `curr_idx` 代表原数组在区间 [curr_left, curr_right] 的和
  // 在 data[curr_idx].
  T query(size_t left, size_t right, size_t curr_left, size_t curr_right,
          size_t curr_idx) {
    // 如果我们遇到的场景是这样的:
    // left --- curr_left --- curr_right --- right
    // 也就是说, 我们当前的两个区间的交集部分的和就是我们
    // 预先求出的原数组在 [curr_left, curr_right] 的和
    if (left <= curr_left && right >= curr_right) {
      return data[curr_idx];
    }

    // curr_left --- curr_right --- left --- right
    // left --- right --- curr_left --- curr_right
    // 没有交集, 记得返回的是幺元
    if (right < curr_left || left > curr_right) {
      return one;
    }

    // 有交集, 但是当前可以处理的区间并没有被待处理的区间完全包裹
    // 这时我们将可以处理的区间一分为二, 然后将我们的查询分摊到
    // 下辖的两个子区间内就可以了.
    auto mid = (curr_left + curr_right) / 2;
    auto left_branch_idx = curr_idx * 2, right_branch_idx = curr_idx * 2 + 1;

    return op(query(left, right, curr_left, mid, left_branch_idx),
              query(left, right, mid + 1, curr_right, right_branch_idx));
  }

 public:
  // The data begins from 1
  SimpleSegTree(const std::vector<T> &vec)
      :  // 我们存放数据, 在原数组的长度为 2^n 时, 需要 2 ^ (n + 1) - 1 的大小
         // 那么, 我们可以将不满 2^n 的长度的数组规约到最近的 2^n
        data(1 << static_cast<size_t>(std::ceil(std::log2(vec.size())) + 1)),
        size(vec.size() - 1),
        op(std::plus<T>()),
        one(0) {
    // 建树的区间为 [1, n - 1], 线段树的根存放在下标为 1 的位置
    build(1, vec.size() - 1, 1, vec);
  }

  SimpleSegTree(const std::vector<T> &vec, std::function<T(T, T)> op,
                const T &one)
      : data(1 << static_cast<size_t>(std::ceil(std::log2(vec.size())) + 1)),
        size(vec.size() - 1),
        op(op),
        one(one) {
    build(1, vec.size() - 1, 1, vec);
  }

  T query(size_t left, size_t right) {
    // 查询 [left, right] 和整个原数组的交集, 整个数组的和
    // 存放在 data[1] 的位置.
    return this->query(left, right, 1, size, 1);
  }
};
