/**
 * @file simple_segtree.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-06-29
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <ostream>
#include <utility>
#include <vector>

#include "../concepts.cpp"

/**
 *  线段树（Segment Tree）是一种二叉树结构，用于高效地执行区间查询与更新操作。
 *  每个节点管理数组的一个区间 [l,
 *  r]，叶节点对应单个元素，内部节点的值由左右子区间合并得出（如求和、最小值等）。
 *  构建时从根 [0, n–1]
 *  递归分割中点，将区间划分为左右两半，最终叶节点存元素值，内部节点存左右子节点的合并结果。
 *  查询时，对于目标区间 [L,
 * R]，若当前节点区间完全包含于目标区间，直接返回其值；
 *  若不相交则返回操作的“单位值”（如和为0、最小值为
 *  +∞），否则递归查询左右子并合并结果。 更新时，递归定位到包含目标索引 idx
 *  的叶节点，更新其值，然后回溯更新各祖先节点的合并值。 构建时间为
 *  O(n)，查询与更新时间均为 O(log n)。
 */
///
///                     [0..7] (sum)
///                      /       \
///             [0..3](sum)     [4..7](sum)
///             /     \           /     \
///         [0..1]   [2..3]   [4..5]   [6..7]
///         /   \     /   \     /   \     /   \
///       [0]  [1] [2]  [3]  [4]  [5]  [6]  [7]

/// 我们的线段树模板必须具备加法能力. 我们这里所有运算均用加法表示,
/// 如果是基本数据类型的其他运算, 如max, min等, 我们可以定义一个新的结构体,
/// 且这个结构体可以和这一基本类型隐式互转.
/// 例如我们可以定义:
/// struct SpecialInt {
///    int data;
///    SpecialInt operator+(SpecialInt b) const;
///    SpecialInt(int i): data(i) {}
///    operator int() const {return data;}
/// };
/// 通过这种方法, 我们就可以对任何一种基本的数据类型进行类似运算符重载的操作.
/// 在我们的模板中, 线段树进行的是求和运算,
/// 因为这种运算可以最好地展示出线段树的特征.

/// 如果线段树需要进行单点的更新, 那么最通俗的接口无疑是将某一点更新为某个值,
/// 如果是这样做, 我们就需要知道, 在这个点的位置, 这个节点到底更新了多少.
/// 因此, 在这样的接口下面, 我们需要知道加法运算的逆运算, 也就是求逆元的运算.
/// 所以`update`接口要求类型参数具有减法的性质.

/// 累加运算, 这是我选取加法的重要原因: 乘法需要用作累加器,
/// 但是我不希望看到太多重载. 在区间更新接口`updateDiff`部分,
/// 我们可以看到, 当我们不能给一个节点打上懒惰标签时,
/// 我们需要进行一种累加的运算, 即快速将运算重复`size_t n`次的运算.
/// 这种运算对于lazy_tag功能格外重要.

/// RandomAccessContainer作为容器类型, 拿来写泛型很有用

template <Addable DT>
class SimpleSegTree {
 protected:
  /// 线段树节点, 一个节点包含两个数据变量,
  /// 其中`val`代表本身的数据大小, 而`lazy_tag`则代表了懒加载的修改标记.
  /// 虽然更新的时候可以不管lazy_tag, 但是在查询的时候, 必须将懒标记置为0.
  /// 同时, 在懒惰标记存在的情况系, 如果要对某个点进行全量更新,
  /// 务必要确保将当前节点的和左右子节点的懒惰标记置为0.
  /// 平凡析构至少要保证四方法.
  struct SegNode {
    DT val;
    DT lazy_tag;

    SegNode(const SegNode &) = default;
    SegNode(SegNode &&) = default;
    SegNode &operator=(const SegNode &) = default;
    SegNode &operator=(SegNode &&) = default;

    SegNode(const DT &val, const DT &tag) : val(val), lazy_tag(tag) {}
  };

  DT one;  // 运算的幺元.
           // 幺元是代数系统中个概念. 要求:
           // x + one = x, one + x = x.
           // 例如, 对于算术加法运算, 幺元是0.
           // 这个线段树和普通的线段树的最大区别就是通过幺元增加普适性.
  std::size_t size;  // 线段树维护的数组的尺寸.
                     // 注意, size指的是用于构造线段树的数组的尺寸,
                     // 不是实际存放数据的数组长度.
                     // 也就是说, size != data.size()
                     // 事实上, 在实践中(竞赛中), 4 * size = data.size()
  std::vector<SegNode> data;  // 存放数据的数组.

  /// 线段树的私有构造接口.
  /// 线段树的构造需要一个随机访问容器, 否则复杂度会爆炸.
  /// 参数的含义如下:
  /// `arr`是构造用的随机访问容器.
  /// `left_edge` 和 `right_edge` 代表当前更新节点所覆盖的区间.
  /// 也就是说, 当前节点的值应当是 [left_edge, right_edge] 的元素的和.
  /// `curr_idx` 代表当前节点的下标.
  /// 更新完成后, 满足条件:
  /// data[curr_idx] =
  ///   arr[left_edge] + arr[left_edge + 1] + ... + arr[right_edge]
  template <RandomAccessContainer<DT> Container>
  void build(const Container &arr, size_t left_edge, size_t right_edge,
             size_t curr_idx) {
    // 验证一下参数, 这里因为代码量小, 断言也少,
    // AI的测试代码很不给力, 就当看个乐子,
    // 板子题作为测试样板, 其实功能的覆盖相当单一.
    // 我之前的版本, 单点更新和区间更新如果混用就会出错.
    assert(left_edge <= right_edge);

    // 如果左右边界相同, 达到递归退出条件.
    if (left_edge == right_edge) {
      // 那么当前节点维护的就是pos位置的值了.
      const size_t &pos = left_edge;
      data[curr_idx].val = arr[pos];
      return;
    }

    // 二分的中点.
    // 要是同时需要知道目标状态和当前状态的情况,
    // 这里就会用curr_left和curr_right.
    auto mid = (left_edge + right_edge) / 2;
    // 左右孩子节点的下标.
    // 因为算是一颗完全二叉树, 所以可以用顺序方法存储二叉树,
    // 且这时空间不会被过多浪费.
    // 如果是普通的二叉树, 一定不能用顺序方法.
    // 当然, 其实用那种动态开点的思路也可以.
    // 可以说, 完全二叉树是线段树存在的基础.
    auto left_idx = curr_idx * 2, right_idx = curr_idx * 2 + 1;

    // 构造左右子树.
    // 在左右子树构造完成后,
    // `data[left_idx]` 存储了 `arr` 在 [left_edge, mid] 区间的和.
    // `data[right_idx]` 存储了 `arr` 在 [mid + 1, right_edge] 区间的和.
    build(arr, left_edge, mid, left_idx);
    build(arr, mid + 1, right_edge, right_idx);

    // 我们这个时候已经知道
    // [left_edge, mid] 和 [mid + 1, right_edge] 区间上的和了,
    // 那么只要将二者相加, 就可以知道整个区间的和.
    data[curr_idx].val = data[left_idx].val + data[right_idx].val;
  }

  // 只有在懒惰标记存在的时候, 这个pushDown才成立.
  // 这一点一定要切记.
  // 这个方法的作用在于将当前节点的懒惰标记置0,
  // 更新当前节点的数值, 并且更新下层节点的懒惰标记,
  // 将变更的存储转移到下级.
  void pushDown(size_t curr_left, size_t curr_right, size_t curr_idx)
    requires Accumulateable<DT>
  {
    // 如果叶子节点上存在懒惰标记.
    // 这时只需要将懒惰标记重置, 并将原先的懒惰标记的值附加到`val`上.
    if (curr_left == curr_right) {
      data[curr_idx].val = data[curr_idx].val + data[curr_idx].lazy_tag;
      data[curr_idx].lazy_tag = one;
      return;
    }

    // 拿到左右孩子节点的下标.
    auto left_idx = curr_idx * 2, right_idx = curr_idx * 2 + 1;

    // 更新左右孩子节点的懒惰标记,
    // 考虑到原先他们的懒惰标记可能已经不为0了,
    // 所以正确的方法是将新的懒惰标记附加到旧的上面.
    data[left_idx].lazy_tag = data[left_idx].lazy_tag + data[curr_idx].lazy_tag;
    data[right_idx].lazy_tag =
        data[right_idx].lazy_tag + data[curr_idx].lazy_tag;

    // 更新真值.
    // 如果一个节点的真值为val, 懒惰标记为tag, 那么说明此时:
    // arr[curr_left] + ... + arr[curr_left] 在过去的值是val,
    // 在val被计算出来之后, 数组从左到右, 每个元素都已经被要求增加tag.
    // 也就是说, 现在这个节点的值应该是:
    // actual_val = arr[left_edge] + tag
    //            + arr[left_edge + 1] + tag
    //            + ...
    //            + arr[right_edge] + tag
    //            = val + (tag + tag + ... + tag)
    //            = val + tag * interval_len
    // 其中 `interval_len` 是区间的长度,
    // [left_edge, right_edge] 的长度应该是:
    // right_edge - left_edge + 1 (这个应该算是简单植树问题[doge])
    data[curr_idx].val = data[curr_idx].val +
                         data[curr_idx].lazy_tag * (curr_right - curr_left + 1);

    data[curr_idx].lazy_tag = this->one;
  }

  // 区间增量更新.
  // 所谓懒惰标记, 一般就是用在这里.
  // 这里出现了两个区间, 分别是:
  // [left_edge, right_edge] 代表本次更新中需要被更新的数组区间.
  //    也就是说 arr[left_edge] 到 arr[right_edge],
  //    每个元素都要加上一个diff
  // [curr_left, curr_right] 代表当前节点所覆盖的区间
  //    也就是说当前节点的值是arr数组在[left_edge, right_edge]
  //    区间内的和.
  // 一定注意, 这里的递归写法不建议用于生产环境.
  // 生产环境下, 如果控制不好容易栈溢出.
  // 当然一般中小数据量也不可能爆栈就是了.
  void updateDiff(size_t left_edge, size_t right_edge, const DT &diff,
                  size_t curr_left, size_t curr_right, size_t curr_idx)
    requires Accumulateable<DT>
  {
    // 这里是用来减少错误, 加快速度的.
    // 如果增量的值等于幺元, 就相当于对于区间内的元素值不变.
    // 那直接忽略这次更新请求.
    if (diff == this->one) {
      return;
    }

    // 如果当前节点所覆盖的区间和目标区间完全没有交集,
    // 那么直接退出, 因为这个区间内根本就不需要更新.
    // 注意这个分支是会被覆盖的.
    // 例如:
    // [0, 9]的线段树上, [4, 5]区间更新时,
    // 第一次递归产生两个区间的更新:
    // [0, 4], [5, 9].
    // 第二次递归, 在[0, 4]上产生:
    // [0, 2], [3, 4].
    // 这个时候, [0, 2]和我们的目标区间[4, 5]
    // 就完全没有交集.
    if (curr_right < left_edge || curr_left > right_edge) {
      return;
    }

    // 如果当前的区间完全在目标区间内部.
    // 那么直接更新懒惰标签并退出.
    // 我们这个算法性能高的表现就在这里,
    // 按照正常的逻辑, 我们这个时候依然要向下递归,
    // 那这样的话就和n次单点的更新没有区别了.
    // 但是因为我们制造了这个懒惰标记, 我们的更新可以暂存.
    // 所以我们就可以点到为止, 这就是典型的离线思想.
    // 在[0, 9]的线段树中, 我们如果更新[3, 5],
    // 就会触发这个分支.
    // 当我们更新[3, 5]时, 我们的覆盖区间为[0, 9].
    // 通过递归, 获得:
    // [0, 4], [5, 9],
    // 在 [0, 4] 区间的调用中, 获得区间
    // [0, 2], [3, 4], 其中 [3, 4]在[3, 5]内部.
    if (curr_left >= left_edge && curr_right <= right_edge) {
      data[curr_idx].lazy_tag = data[curr_idx].lazy_tag + diff;
      return;
    }

    // 这里写一个辅助函数, 用来计算两个区间重叠的长度.
    // 因为是我自己写的, 所以比较麻烦, 如果常数过大, 可以怀疑这里.
    // 可以用AI的版本换掉它.
    auto overlap_length = [](size_t l1, size_t r1, size_t l2,
                             size_t r2) -> size_t {
      if (l2 > r1 || l1 > r2) {
        return 0;
      }

      std::pair<size_t, size_t> left_pair(l1, l2), right_pair(r1, r2);
      if (l2 < l1) {
        left_pair = std::pair<size_t, size_t>(l2, l1);
      }
      if (r2 < r1) {
        right_pair = std::pair<size_t, size_t>(r2, r1);
      }

      return (right_pair.first - left_pair.second) + 1;
    };

    // 计算当前覆盖的区间和目标区间的重叠长度.
    auto curr_overlap =
        overlap_length(left_edge, right_edge, curr_left, curr_right);
    // 如果不能设置懒惰标记, 就要在线地更新我们的值.
    // 因为我们当前的覆盖区间不完全在目标区间内部,
    // 这意味这其中有些值是不需要加上diff的,
    // 所以懒惰标签不能用.
    // 因为我们这里只是提供一个增量, 所以过去的修改我们完全可以存着.
    // 因此, 我们这里无需下推(pushDown).
    // 我们不需要把当前的懒惰标签重置为0.
    // 总之就是一个"拖"字决.
    data[curr_idx].val = data[curr_idx].val + diff * curr_overlap;

    // 二分查找, 获取左右区间和孩子节点下标.
    auto mid = (curr_left + curr_right) / 2;
    auto left_idx = curr_idx * 2, right_idx = curr_idx * 2 + 1;

    // 更新左右子树.
    updateDiff(left_edge, right_edge, diff, curr_left, mid, left_idx);
    updateDiff(left_edge, right_edge, diff, mid + 1, curr_right, right_idx);
  }

  // 区间查询.
  // 参数含义同上方updateDiff.
  DT query(size_t left_edge, size_t right_edge, size_t curr_left,
           size_t curr_right, size_t curr_idx) {
    // 如果当前覆盖区间和目标区间完全不相交, 返回幺元.
    // 这种情况产生的原因同上.
    if (left_edge > curr_right || right_edge < curr_left) {
      return this->one;
    }

    // 因为查询的过程需要真值,
    // 所以节点的val成员变量必须直接地反映节点的值.
    // 因此只要是能进行区间增量更新的线段树,
    // 就要进行下推操作.
    if constexpr (Accumulateable<DT>) {
      if (data[curr_idx].lazy_tag != this->one) {
        pushDown(curr_left, curr_right, curr_idx);
      }
    }

    // 如果覆盖区间在目标区间内部, 那么直接退出并返回当前节点值.
    if (curr_left >= left_edge && curr_right <= right_edge) {
      return data[curr_idx].val;
    }

    // 否则进行二分查找.
    auto mid = (curr_left + curr_right) / 2;
    auto left_idx = curr_idx * 2, right_idx = curr_idx * 2 + 1;

    return query(left_edge, right_edge, curr_left, mid, left_idx) +
           query(left_edge, right_edge, mid + 1, curr_right, right_idx);
  }

  // 单点增量更新, 这是最简单的一个更新函数.
  // `pos` 代表待更新节点的位置.
  void updateDiff(size_t pos, const DT &diff, size_t curr_left,
                  size_t curr_right, size_t curr_idx) {
    // 如果覆盖区间长度为1, 说明已经到了叶子节点.
    if (curr_left == curr_right) {
      // 这里我们不处理pos是否等与我们的curr_left和curr_right
      // 因为这个逻辑在后面被处理了.
      // 如果在这里处理的话代码量太大, 而且复杂度会飙升.
      assert(curr_left == curr_right);
      data[curr_idx].val = data[curr_idx].val + diff;
      return;
    }

    // 更新当前节点, 直接更新val就好.
    // 原因在updateDiff中已经介绍过了
    data[curr_idx].val = data[curr_idx].val + diff;

    // 二分
    auto mid = (curr_left + curr_right) / 2;
    auto left_idx = curr_idx * 2, right_idx = curr_idx * 2 + 1;

    // 这里才是上面保证curr_left == pos的逻辑.
    // 我们始终保证pos在覆盖区间内.
    if (pos <= mid) {
      updateDiff(pos, diff, curr_left, mid, left_idx);
    } else {
      assert(pos >= mid + 1);
      updateDiff(pos, diff, mid + 1, curr_right, right_idx);
    }
  }

  // 单点的值更新.
  // 这里因为涉及了查询的操作, 所以需要下推.
  // 也就是要始终确保lazy_tag == 0.
  void update(size_t pos, const DT &val, size_t curr_left, size_t curr_right,
              size_t curr_idx) {
    // 和单点版的updateDiff一样,
    // curr_left == pos 在后方代码中被保证.
    if (curr_left == curr_right) {
      assert(curr_left == pos);
      data[curr_idx].val = val;
      // 把懒惰标签重置了, 毕竟之前的修改我们要像金正日一样不认账.
      data[curr_idx].lazy_tag = one;
      return;
    }

    // 直接二分
    auto mid = (curr_left + curr_right) / 2;
    auto left_idx = curr_idx * 2, right_idx = curr_idx * 2 + 1;

    // 因为涉及到左右子节点的查询, 所以左右子节点都要进行下推操作.
    // 这里只确保当前节点的懒惰标签被重置,
    // 左右子节点的更新当然是在update和下方代码中实现.
    // 这样的好处, 就是不太容易出错, 不用在公有方法中处理这种case.
    // 算是提供了一个很薄的抽象层.
    // 当然这只是对于可累加的, 可能做过区间修改的类型而言的.
    if constexpr (Accumulateable<DT>) {
      if (data[curr_idx].lazy_tag != one) {
        pushDown(curr_left, curr_right, curr_idx);
      }
    }

    if (pos <= mid) {
      // 更新左子树
      // 完成后左子树已经经过下推操作.
      update(pos, val, curr_left, mid, left_idx);

      // 对右子节点进行下推操作.
      if constexpr (Accumulateable<DT>) {
        if (data[left_idx].lazy_tag != one) {
          pushDown(mid + 1, curr_right, right_idx);
        }
      }
    }
    if (pos >= mid + 1) {
      // 更新右子树
      // 完成后右子树已经经过下推操作.
      update(pos, val, mid + 1, curr_right, right_idx);

      // 对左子节点进行下推操作.
      if constexpr (Accumulateable<DT>) {
        if (data[left_idx].lazy_tag != one) {
          pushDown(curr_left, mid, left_idx);
        }
      }
    }

    // 查询左右区间的和, 并用二者的和作为当前覆盖区间内的和.
    data[curr_idx].val = data[left_idx].val + data[right_idx].val;
  }

 public:
  // 构造函数, 从一个容器构造.
  template <RandomAccessContainer<DT> Container>
  SimpleSegTree(const Container &arr, const DT &one)
      : one(one),
        size(arr.size()),
        // 这里我们要找到一个差不多最小的数据大小, 实际上4n就可以了.
        // 但是这是模板, 是范例, 要求就要严格.
        // 这个最小可以达到 2n, 在某些特殊情况下.
        // 最大也绝不超过4n.
        data(1 << (static_cast<size_t>(std::ceil(std::log2(size)) + 1)),
             {one, one}) {
    // 构造线段树, [0, size - 1] 对应的节点下标为1.
    build(arr, 0, size, 1);
  }

  // 使用默认构造函数构造出来的数据,
  // 我们认为他是幺元.
  template <RandomAccessContainer<DT> Container>
    requires requires() {
      { DT() };
    }
  SimpleSegTree(const Container &arr) : SimpleSegTree(arr, DT{}) {}

  // 构造空线段树.
  // 因为是空的线段树, 所以每个元素都是幺元.
  SimpleSegTree(size_t size, const DT &one)
      : one(one),
        size(size),
        data(1 << (static_cast<size_t>(std::ceil(std::log2(size))) + 1),
             {one, one}) {}

  // 同样, 使用自动生成的幺元.
  SimpleSegTree(size_t size)
    requires requires() {
      { DT() };
    }
      : SimpleSegTree(size, DT{}) {}

  // 拷贝构造方法.
  SimpleSegTree(const SimpleSegTree &) = default;
  SimpleSegTree(SimpleSegTree &&) = default;

  // 重新构造线段树, 使用一个容器.
  template <RandomAccessContainer<DT> Container>
  void build(const Container &arr) {
    // 调整一下容器尺寸和存放数据的数组的尺寸.
    this->size = arr.size();
    data.resize(1 << (static_cast<size_t>(std::ceil(std::log2(size))) + 1),
                {one, one});
    // 根节点的覆盖区间为[0, size - 1].
    // 下标为1
    build(arr, 0, size - 1, 1);
  }

  // 单点更新
  void update(size_t pos, const DT &val) { update(pos, val, 0, size - 1, 1); }

  // 区间增量更新
  void updateDiff(size_t left_edge, size_t right_edge, const DT &diff)
    requires Accumulateable<DT>
  {
    // 根节点的覆盖区间为[0, size - 1].
    // 下标为1
    updateDiff(left_edge, right_edge, diff, 0, size - 1, 1);
  }

  // 单点增量更新
  void updateDiff(size_t pos, const DT &diff) {
    // 根节点的覆盖区间为[0, size - 1].
    // 下标为1
    updateDiff(pos, diff, 0, size - 1, 1);
  }

  DT query(size_t left_edge, size_t right_edge) {
    // 根节点的覆盖区间为[0, size - 1].
    // 下标为1
    return query(left_edge, right_edge, 0, size - 1, 1);
  }
};

class SimpleSegTreeTester {
  // Test Type 1: A custom struct for string concatenation (Addable, but not
  // Accumulateable)
  struct StringConcat {
    std::string s;

    StringConcat(const std::string &str = "") : s(str) {}
    StringConcat(const char *c_str) : s(c_str) {}

    StringConcat operator+(const StringConcat &other) const {
      return StringConcat(s + other.s);
    }

    bool operator==(const StringConcat &other) const { return s == other.s; }
  };

 public:
  void run_all_tests() {
    std::cout << "--- Running All Segment Tree Tests ---" << std::endl;
    test_accumulateable_type();
    test_custom_non_accumulateable_type();
    std::cout << "\n--- All Tests Completed ---" << std::endl;
  }

 private:
  void test_accumulateable_type() {
    std::cout << "\n[TEST CASE 1] Accumulateable Type (long long)" << std::endl;
    std::vector<long long> initial_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    SimpleSegTree<long long> tree(initial_data, 0LL);

    std::cout << "  - Testing construction and initial queries..." << std::endl;
    assert(tree.query(0, 9) == 55);
    assert(tree.query(0, 0) == 1);
    assert(tree.query(3, 5) == 4 + 5 + 6);
    std::cout << "    SUCCESS: Initial queries are correct." << std::endl;

    std::cout << "  - Testing range update (updateDiff)..." << std::endl;
    // Add 10 to elements in range [2, 6]
    // Original: {1, 2, 13, 14, 15, 16, 17, 8, 9, 10}
    tree.updateDiff(2, 6, 10);
    assert(tree.query(2, 6) ==
           (3 + 10) + (4 + 10) + (5 + 10) + (6 + 10) + (7 + 10));
    assert(tree.query(0, 9) == 55 + 5 * 10);
    assert(tree.query(0, 1) == 1 + 2);
    assert(tree.query(7, 9) == 8 + 9 + 10);
    std::cout << "    SUCCESS: Range update works as expected." << std::endl;

    std::cout << "  - Testing point update (update)..." << std::endl;
    // Set element at index 0 to 100
    // Data: {100, 2, 13, 14, 15, 16, 17, 8, 9, 10}
    tree.update(0, 100);
    assert(tree.query(0, 0) == 100);
    assert(tree.query(0, 1) == 100 + 2);
    // Total sum was 105, now it's 105 - 1 + 100 = 204
    assert(tree.query(0, 9) == 204);
    std::cout << "    SUCCESS: Point update works correctly." << std::endl;

    std::cout << "  - Testing chained lazy updates..." << std::endl;
    // Add 5 to [0, 4]
    tree.updateDiff(0, 4, 5);
    // Add -2 to [3, 7]
    tree.updateDiff(3, 7, -2);
    // Expected data after all ops:
    // Original: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    // Op1 (+10 to [2,6]): {1, 2, 13, 14, 15, 16, 17, 8, 9, 10}
    // Op2 (set [0] to 100): {100, 2, 13, 14, 15, 16, 17, 8, 9, 10}
    // Op3 (+5 to [0,4]): {105, 7, 18, 19, 20, 16, 17, 8, 9, 10}
    // Op4 (-2 to [3,7]): {105, 7, 18, 17, 18, 14, 15, 6, 9, 10}
    assert(tree.query(0, 0) == 105);  // 100 + 5
    assert(tree.query(1, 1) == 7);    // 2 + 5
    assert(tree.query(2, 2) == 18);   // 3 + 10 + 5
    assert(tree.query(3, 3) == 17);   // 4 + 10 + 5 - 2
    assert(tree.query(4, 4) == 18);   // 5 + 10 + 5 - 2
    assert(tree.query(5, 5) == 14);   // 6 + 10 - 2
    assert(tree.query(6, 6) == 15);   // 7 + 10 - 2
    assert(tree.query(7, 7) == 6);    // 8 - 2
    assert(tree.query(8, 9) == 9 + 10);
    std::cout << "    SUCCESS: Chained lazy updates are handled correctly."
              << std::endl;
  }

  void test_custom_non_accumulateable_type() {
    std::cout << "\n[TEST CASE 2] Custom Non-Accumulateable Type (StringConcat)"
              << std::endl;
    std::vector<StringConcat> initial_data = {"A", "B", "C", "D", "E"};
    SimpleSegTree<StringConcat> tree(initial_data, StringConcat(""));

    std::cout << "  - Testing construction and initial queries..." << std::endl;
    assert(tree.query(0, 4) == "ABCDE");
    assert(tree.query(1, 3) == "BCD");
    assert(tree.query(4, 4) == "E");
    std::cout << "    SUCCESS: Initial queries are correct." << std::endl;

    std::cout << "  - Testing point update (update)..." << std::endl;
    tree.update(2, "X");  // Update "C" to "X"
    assert(tree.query(0, 4) == "ABXDE");
    assert(tree.query(1, 3) == "BXD");
    std::cout << "    SUCCESS: Point update works correctly." << std::endl;

    std::cout << "  - Verifying range update is not available..." << std::endl;
    // The following line would cause a compilation error because StringConcat
    // is not Accumulateable tree.updateDiff(0, 2, StringConcat("Z"));
    std::cout << "    SUCCESS: Range update correctly disabled by concept "
                 "constraints."
              << std::endl;
  }
};
