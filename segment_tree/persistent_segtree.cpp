/**
 * @file persistent_segtree.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-07-18
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <utility>
#include <vector>

#include "../concepts.cpp"

/**
 * 主席树（可持久化线段树）原理说明
 *
 * 前置要求：
 *   1. 理解线段树(ADT)的基本概念：区间二分、节点存储区间统计值
 *   2. 了解动态开点技术：用指针/数组下标代替固定数组结构
 *
 * 核心思想：通过"部分复用"实现历史版本保留
 *
 * 传统线段树 vs 主席树：
 *
 *  传统线段树（数组存储）：
 *      [1,4]                      内存结构（初始版本）：
 *      /    \                      [0]: [1,4] = 5
 *   [1,2]  [3,4]                   [1]: [1,2] = 2
 *   /   \   /   \                  [2]: [3,4] = 3
 * [1] [2] [3] [4]                 [3]: [1] = 1
 *                                  [4]: [2] = 1
 *                                  [5]: [3] = 1
 *                                  [6]: [4] = 2
 *
 *  修改位置2的值后（直接覆盖）：
 *      [1,4] = 6                  旧版本被破坏！
 *      /    \
 *   [1,2]=3  [3,4]=3
 *
 * 主席树（动态开点 + 版本链）：
 *
 *  初始版本v0：
 *      (0) [1,4] = 5              节点结构：{ left, right, sum }
 *          /    \
 *    (1)[1,2]=2 (2)[3,4]=3
 *
 *  修改位置2（创建新版本v1）：
 *      (3) [1,4] = 6  <-- v1根节点  关键原则：只克隆访问路径上的节点
 *          /    \                  非修改路径复用旧节点
 *       (4)[1,2]=3  \              v0节点(2)被v1复用
 *        /   \       \
 *    (5)[1] (6)[2]   (2)[3,4]=3  // 注意这里复用v0的节点(2)
 *
 *  版本关系示意图：
 *      v0: 0--1--[叶子]  0--2--[叶子]
 *      v1: 3--4--5      3--2 (复用v0的2号节点)
 *
 * 动态开点技术：
 *   - 节点用结构体表示：struct Node { int l, r, sum; }
 *   - 不再使用堆式存储（2*p, 2*p+1），用l,r记录子节点下标
 *   - 初始只有根节点，按需动态创建子节点
 *
 * 可持久化原理（修改流程）：
 *   1. 从目标版本的根节点开始遍历
 *   2. 对访问路径上的每个节点：
 *        a. 创建新节点，复制原节点数据
 *        b. 修改涉及的分支指向新创建的子节点
 *        c. 未修改的分支指向旧节点（直接复用）
 *   3. 新版本根节点指向修改后的路径
 *
 * 空间复杂度：O(q log n)  q=操作数
 * 时间复杂度：每次修改 O(log n)
 *
 * 形象理解：
 *  每次修改只"长出"新分支，未修改部分像"磁吸"一样附着到旧版本树上
 *
 * 图示修改过程（修改位置2）：
 *
 *  版本v0:       节点0 (根)         版本v1:       节点3 (新根)
 *               /      \                      /        \
 *            节点1      节点2               节点4        节点2(复用)
 *           /   \      /   \              /   \
 *        节点3 节点4 节点5 节点6        节点5(新) 节点6(新)
 *
 * 注意：实际叶子节点不展开，这里为示意保留层次
 */

template <Addable E>
class PersistentSegTree {
 protected:
  struct SegNode {
    E elem;           // 存储节点的数据.
    size_t left = 0;  // 分别代表左右子节点的下标.
    size_t right = 0;

    // 默认构造
    SegNode() = default;

    // 完美转发构造（推荐首选）
    template <typename... Args>
    explicit SegNode(Args&&... args) : elem(std::forward<Args>(args)...) {}

    // 五法则 + noexcept移动
    SegNode(SegNode&&) noexcept = default;
    SegNode(const SegNode&) = default;
    SegNode& operator=(SegNode&&) noexcept = default;
    SegNode& operator=(const SegNode&) = default;
    ~SegNode() = default;
  };

  // nodes存储的是线段树的节点.
  E one;
  size_t size;
  std::vector<SegNode> nodes;
  std::vector<size_t> roots;

  template <typename... Args>
  size_t assignNode(Args&&... args) {
    auto idx = nodes.size();
    nodes.emplace_back(std::forward<Args>(args)...);
    return idx;
  }

  template <RandomAccessContainer<E> Container>
  void build(Container&& arr, size_t left, size_t right, size_t curr_idx) {
    if (left == right) {
      auto& pos = left;
      // 叶子节点, 构造停止, 向上返回.
      // 当左边界和右边界相等的时候,
      // 意味着当前节点覆盖的区间内只包含一个数,
      // 因为nodes[curr_idx] = arr[left] + ... + arr[right];
      // 所以当left = right时, nodes[curr_idx]存储了一个长度为1的数列的和.
      nodes[curr_idx].elem = arr[pos];
      return;
    }

    // 线段树二分的中点.
    auto mid = (left + right) / 2;

    // 分配两个新节点用来存储左右孩子.
    auto left_idx = assignNode(), right_idx = assignNode();
    // 将当前节点的左右指针分别指向新分配的节点.
    nodes[curr_idx].left = left_idx, nodes[curr_idx].right = right_idx;

    // 还是线段树传统的分治思想.
    // 构造左右两个部分的和.
    build(arr, left, mid, left_idx);
    build(arr, mid + 1, right, right_idx);

    nodes[curr_idx].elem = nodes[left_idx].elem + nodes[right_idx].elem;
  }

  E query(size_t left, size_t right, size_t curr_left, size_t curr_right,
          size_t curr_idx) {
    // 如果当前节点覆盖区间完全在待查找区间外部.
    // 为什么会出现这个分支查找朴素线段树.
    if (curr_left > right || curr_right < left) {
      return one;
    }

    // 如果当前节点覆盖区间被包含就直接返回.
    if (curr_left >= left && curr_right <= right) {
      return nodes[curr_idx].elem;
    }

    // 二分查找
    auto mid = (curr_left + curr_right) / 2;
    auto left_idx = nodes[curr_idx].left, right_idx = nodes[curr_idx].right;

    // 查找两侧区间
    return query(left, right, curr_left, mid, left_idx) +
           query(left, right, mid + 1, curr_right, right_idx);
  }

  // 单点更新, 直接替换版本
  // pos和val是待更新节点的位置和新值,
  // curr_* 区间代表当前节点覆盖区间.
  // old_curr_idx和new_curr_idx代表当前节点的新旧版本在nodes中的下标.
  void update(size_t pos, size_t val, size_t curr_left, size_t curr_right,
              size_t old_curr_idx, size_t new_curr_idx) {
    // 如果左右边界相等,
    // 那么当前节点覆盖区间长度为1.
    if (curr_left == curr_right) {
      // 这里判断pos处在哪个分区的工作在后面判断.
      // 因为单点判断不是很复杂.
      assert(pos == curr_left);
      nodes[new_curr_idx].elem = val;
      return;
    }

    // 二分法中点
    auto mid = (curr_left + curr_right) / 2;

    // 如果待更新节点在左半边
    if (pos <= mid) {
      // 新节点需要分配.
      // 如果要上多线程环境那在这里加锁.
      // 不说了, 一说起OS我就心痛.
      // My heart breaks!
      auto new_left_idx = assignNode();

      // 旧的孩子节点直接转移.
      auto old_left_idx = nodes[old_curr_idx].left;
      auto right_idx = nodes[old_curr_idx].right;

      // 更新当前节点的孩子指针
      nodes[new_curr_idx].left = new_left_idx,
      nodes[new_curr_idx].right = right_idx;

      // 更新左子树, 因为待更新节点在左子树.
      update(pos, val, curr_left, mid, old_left_idx, new_left_idx);
    } else {
      // 如果待更新节点在左半边
      assert(pos >= mid + 1);

      // 分配, 转移, 同理
      auto new_right_idx = assignNode();
      auto old_right_idx = nodes[old_curr_idx].right;
      auto left_idx = nodes[old_curr_idx].left;

      // 更新孩子指针.
      nodes[new_curr_idx].right = new_right_idx,
      nodes[new_curr_idx].left = left_idx;

      // 更新右子树
      update(pos, val, mid + 1, curr_right, old_right_idx, new_right_idx);
    }

    // 在更新完成孩子节点之后, 当前节点的值就是孩子的值的和.
    nodes[new_curr_idx].elem = nodes[nodes[new_curr_idx].left].elem +
                               nodes[nodes[new_curr_idx].right].elem;
  }

  // 增量更新节点, 和普通更新比起来, 除了val变成diff,
  // 参数的含义都差不多.
  void updateDiff(size_t pos, size_t diff, size_t curr_left, size_t curr_right,
                  size_t old_curr_idx, size_t new_curr_idx) {
    // 更新的值为0就打道回府.
    if (diff == one) {
      // 直接从旧状态转移过来.
      nodes[new_curr_idx] = nodes[old_curr_idx];
      return;
    }

    // 如果覆盖区间大小为1了, 那么直接进行增量更新.
    if (curr_left == curr_right) {
      assert(pos == curr_left);
      nodes[new_curr_idx].elem = nodes[old_curr_idx].elem + diff;
      return;
    }

    // 二分法.
    auto mid = (curr_left + curr_right) / 2;

    // 一样的套路...
    if (pos <= mid) {
      auto new_left_idx = assignNode();
      auto old_left_idx = nodes[old_curr_idx].left;
      auto right_idx = nodes[old_curr_idx].right;

      nodes[new_curr_idx].left = new_left_idx,
      nodes[new_curr_idx].right = right_idx;

      updateDiff(pos, diff, curr_left, mid, old_left_idx, new_left_idx);
    } else {
      assert(pos >= mid + 1);

      auto new_right_idx = assignNode();
      auto old_right_idx = nodes[old_curr_idx].right;
      auto left_idx = nodes[old_curr_idx].left;

      nodes[new_curr_idx].right = new_right_idx,
      nodes[new_curr_idx].left = left_idx;

      updateDiff(pos, diff, mid + 1, curr_right, old_right_idx, new_right_idx);
    }

    // 这里的值更新也变成从旧版本增量转移.
    // 不需要从孩子节点转移了.
    // 增量更新就这点好.
    nodes[new_curr_idx].elem = nodes[old_curr_idx].elem + diff;
  }

 public:
  // 构造函数集
  // 使用大小和幺元进行构造
  PersistentSegTree(size_t size, const E& one) : one(one), size(size) {
    nodes.reserve(size);
    nodes.emplace_back(one);
  }

  // 使用右值进行幺元构造
  PersistentSegTree(size_t size, E&& one) : one(std::move(one)), size(size) {
    nodes.reserve(size);
    nodes.emplace_back(one);
  }

  // 使用默认构造方法构造出来的元素值作为幺元构造
  PersistentSegTree(size_t size) : PersistentSegTree(size, E{}) {}

  // 使用一个容器进行构造.
  template <RandomAccessContainer<E> Container>
  PersistentSegTree(Container&& arr, const E& one)
      : PersistentSegTree(arr.size(), one) {
    roots.emplace_back(assignNode());
    build(std::forward<Container>(arr), 0, size - 1, roots.back());
  }

  // 使用容器和右值幺元构造
  template <RandomAccessContainer<E> Container>
  PersistentSegTree(Container&& arr, E&& one)
      : PersistentSegTree(arr.size(), std::move(one)) {
    roots.emplace_back(assignNode());
    build(std::forward<Container>(arr), 0, size - 1, roots.back());
  }

  // 使用容器和默认幺元构造.
  template <RandomAccessContainer<E> Container>
  PersistentSegTree(Container&& arr)
      : PersistentSegTree(std::forward<Container>(arr), E{}) {}

  // 使用容器建树
  template <RandomAccessContainer<E> Container>
  void build(Container&& arr) {
    assert(arr.size() >= size);
    // 创造一个新的根出来.
    roots.emplace_back(assignNode());
    // 开始建树
    build(std::forward<Container>(arr), 0, size - 1, roots.back());
  }

  // 单点更新
  void update(size_t pos, const E& elem) {
    // 构造出来需要的val.
    E val(elem);

    // 获取旧版本的根节点
    auto old_root = roots.back();

    // 创造一个新版本的根.
    roots.emplace_back(assignNode());
    auto new_root = roots.back();

    // 开始更新.
    update(pos, val, 0, size - 1, old_root, new_root);
  }

  // 增量更新, 方法相同.
  void updateDiff(size_t pos, const E& elem) {
    E diff(elem);
    auto old_root = roots.back();
    roots.emplace_back(assignNode());
    auto new_root = roots.back();
    updateDiff(pos, diff, 0, size - 1, old_root, new_root);
  }

  // 获得版本的数量.
  size_t getVersionCount() { return roots.size(); }

  // 查找, 参数包含版本号, 左边界和右边界.
  E query(size_t version, size_t left, size_t right) {
    // 找到相应版本的根节点.
    auto root_idx = roots[version];
    return query(left, right, 0, size - 1, root_idx);
  }

  // 默认查找最新版本.
  E query(size_t left, size_t right) {
    if (getVersionCount() == 0) {
      return one;
    } else {
      return query(getVersionCount() - 1, left, right);
    }
  }
};

/**
 * KthTree（基于主席树的区间第K小查询）
 *
 * 问题：给定数组a[1..n]，查询任意区间[l, r]中第k小的数
 *
 * 解决方案：主席树 + 权值线段树 + 离散化
 *
 * 核心思想：
 *  1. 离散化：将原始数组的值映射到排名空间[1, m] (m为去重后元素个数)
 *  2. 权值线段树：每个线段树版本维护值域区间内元素的出现频次
 *  3. 可持久化：按数组顺序构建版本链，每个版本i对应前缀区间[1, i]
 *
 * 关键原理：
 *  对于查询[l, r]：
 *      root_r - root_{l-1} = 区间[l, r]的权值线段树
 *
 * 操作流程：
 *
 *  预处理阶段：
 *    版本0: 空树
 *    版本1: 插入a[1]对应的离散值
 *    版本2: 插入a[2]对应的离散值
 *    ...
 *    版本n: 插入a[n]对应的离散值
 *
 *  查询阶段：
 *    root_left = roots[l-1]   -> 对应前缀[1, l-1]
 *    root_right = roots[r]    -> 对应前缀[1, r]
 *    在两棵树同步递归：
 *        cnt = (右树左子节点值 - 左树左子节点值)
 *        if k <= cnt: 进入左子树
 *        else: k -= cnt, 进入右子树
 *
 * 结构示例：
 *  数组: [2, 1, 3, 4, 1] (离散化后: 1->1, 2->2, 3->3, 4->4)
 *  版本树:
 *    v0: 空树
 *    v1: [2] -> 位置2的计数=1
 *        [1,4]:1
 *           |
 *        [2,2]:1
 *
 *    v2: [2,1] -> 位置1的计数=1
 *        [1,4]:2
 *         /   \
 *    [1,2]:1  [3,4]:1
 *      /   \
 * [1]:1 [2]:1
 *
 *    v3: [2,1,3] -> 位置3的计数=1
 *        [1,4]:3
 *         /   \
 *    [1,2]:1  [3,4]:2
 *
 *  查询[2,3]区间的第1小值：
 *      使用树v3 - 树v1 = [2,3]区间
 *        根节点: [1,4]: (3-1)=2
 *        左子树: [1,2]: (1-1)=0
 *        右子树: [3,4]: (2-0)=2  -> 第1小在右子树
 *        在右子树中: [3,4]的第1小 -> 3
 *
 * 动态开点优势：
 *  每个版本只需O(log m)的新节点，总空间O(n log m)
 */

class KthTree : public PersistentSegTree<size_t> {
 protected:
  // 这里这个kthElem尤其值得注意.
  // left和right分别代表当前查找的区间的最小值和最大值.
  // K小树是权值线段树的特殊应用, 因此这个线段树相当于维护了这样一个数组:
  // arr[0] - arr[n] 分别代表了在这个原数组中0 - n的个数.
  // 这里每次update这个数组, 对 pos 进行 1 的增量更新,
  // 就代表值为pos的元素增加了一个.
  // left_idx和right_idx代表的是当前节点的两个版本.
  // 就是两个版本的线段树中的覆盖[left, right]区间的节点.
  size_t kthElem(size_t k_remaining, size_t left, size_t right, size_t left_idx,
                 size_t right_idx) {
    if (left == right) {
      return left;
    }

    // 这里代表的是中间位置的值.
    auto mid = (left + right) / 2;

    // 这里我们要获取两个左侧孩子的值
    // 下面这行代码获取的是节点的内容的差值.
    // 假设left_idx所存放的节点是从left_ver获得的,
    // right_idx存放的是right_ver版本的线段树获得的.
    // 那么, 下面这个差值就是数组在(left_ver, right_ver - 1]
    // 这个区间内的值在[left, mid]区间内的元素的个数.
    auto left_diff =
        nodes[nodes[right_idx].left].elem - nodes[nodes[left_idx].left].elem;

    if (k_remaining < left_diff) {
      // 这个分支表明在待查询的区间内,
      // 介于[left, mid]这个区间内的元素有left_diff个,
      // 也就是前0到left_diff - 1小的元素都在左子树
      // 所以我们要查找的就是在左子树中的第k小
      return kthElem(k_remaining, left, mid, nodes[left_idx].left,
                     nodes[right_idx].left);
    } else {
      // 这个分支表明当前需要查找的第k小元素并不在左半区间.
      // 那么它理应在右半区间.
      return kthElem(k_remaining - left_diff, mid + 1, right,
                     nodes[left_idx].right, nodes[right_idx].right);
    }
  }

 public:
  KthTree(size_t max_val) : PersistentSegTree<size_t>(max_val + 1, 0) {
    roots.push_back(0);
  }

  // 我们每次进行更新,
  // 假设我们当前已经更新到a[i],
  // 那么当我们updateDiff(a[i + 1], 1)时,
  // 我们相当于创造了一个新版本的线段树,
  // 这个线段树是a[0]-a[i+1]的权值线段树,
  // 而旧版本的则是a[0]-a[i]的权值线段树.
  // 因为我们已经存在一个全0的初始版本线段树了,
  // 那么第x+1版的线段树, 可以递归地认识到,
  // 就是a[0] - a[x]的权值线段树.
  template <RandomAccessContainer<size_t> Container>
  KthTree(Container&& arr, size_t max_val)
      : PersistentSegTree<size_t>(max_val + 1, 0) {
    roots.push_back(0);
    for (const auto& it : arr) {
      updateDiff(it, 1);
    }
  }

  // 添加一个元素.
  void add(size_t elem) { updateDiff(elem, 1); }
  
  // 查找区间中第K小的元素.
  // 注意这里K是从0开始, 也就是第0,1,...,n-1小的元素.
  // left和right表示区间的左右边界, 这里是闭区间,
  // 也就是说left和right包含在区间当中.
  // 这里left, right是从0开始的, 和STL风格相统一.
  size_t kthElem(size_t k, size_t left, size_t right) {
    return kthElem(k, 0, size - 1, roots[left], roots[right + 1]);
  }
};
