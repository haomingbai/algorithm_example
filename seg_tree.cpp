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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

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
template <typename T>
concept Addable = requires(T a, T b) {
  { a + b } -> std::convertible_to<T>;
};

/// 如果线段树需要进行单点的更新, 那么最通俗的接口无疑是将某一点更新为某个值,
/// 如果是这样做, 我们就需要知道, 在这个点的位置, 这个节点到底更新了多少.
/// 因此, 在这样的接口下面, 我们需要知道加法运算的逆运算, 也就是求逆元的运算.
/// 所以`update`接口要求类型参数具有减法的性质.
template <typename T>
concept Subtractable = requires(T a, T b) {
  { a - b } -> std::convertible_to<T>;
};

/// 累加运算, 这是我选取加法的重要原因: 乘法需要用作累加器,
/// 但是我不希望看到太多重载. 在区间更新接口`updateDiff`部分,
/// 我们可以看到, 当我们不能给一个节点打上懒惰标签时,
/// 我们需要进行一种累加的运算, 即快速将运算重复`size_t n`次的运算.
/// 这种运算对于lazy_tag功能格外重要.
template <typename T>
concept Accumulateable = requires(T a, size_t b) {
  { a * b } -> std::convertible_to<T>;
};

/// 朴素版本的线段树, 类型参数要求加法运算
template <Addable DataType>
class SimpleSegTree {
  size_t size;   /// 线段树可以用来表示一个数组, 同时快速查询它的一些性质.
                 /// `size`变量表示了这样的数组的长度
                 /// 在整个朴素线段树中, 几乎所有的下标都从1开始,
                 /// 因此下标的范围是[1, size].
  DataType one;  /// 幺元是代数系统中的重要概念, 有幺元才有逆元.
                 /// 如果一个代数系统存在幺元且满足结合率,
                 /// 那么这个代数系统称为 **半群**.
                 /// 如果对于半群中的每一个元素都存在逆元, 那么称为群.
                 /// 幺元e的性质满足:
                 /// 对于任意的a, a * e = e * a = a.
  std::vector<DataType>
      data;  /// 这里是我们用来存放数据的位置.
             /// 因为在朴素的线段树中, 当线段树的大小确定,
             /// 节点的数量也就确定. 所以这里向量的大小在构造的时候就确定了.
  std::vector<DataType>
      lazy_tag;  /// 这个是存放懒加载用到的懒惰标签的, 长度和`data`相同.
                 /// 这里的代码习惯很不好, 是一个高耦合的设计,
                 /// 通过下标将`lazy_tag`和`data`粘合在一起.
                 /// 但是考虑到这里注释清晰, 亦不为过.
                 /// 且我在写这个模板的时候还不知道"懒惰标签"的存在,
                 /// 因此犯下这种错误恐怕也情有可原.
                 /// 事实上, 这里可以这样写(但是函数实现得跟着改):
                 /// struct Node {
                 ///    DataType dat;
                 ///    DataType lazy_tag;
                 /// };
                 /// std::vector<Node> data;
                 /// 为了防止日后不理解, 因此作此声明.

  /// 下推函数, 用来将懒惰标签下移.
  /// 例如我在存放[1, 9]的和的节点处存在懒惰标签'x', 这意味着:
  /// 在数组下标[1, 9]范围内的每个元素, 都应该加上'x'. 但是我为了减少操作的数量,
  /// 因此没有进行这个操作. 如果这个时候存在一个查询, 要查[1, 9]的和,
  /// 我就可以进行一次下推操作. 下推操作只更新当前的节点,
  /// 并把惰性标签移动到了它的两个孩子上,
  /// 如果对于它的子节点依然要进行更新或者查询,
  /// 那么就需要对两个孩子节点继续进行下推, 直到满足条件位置.
  /// 这里的一个基本原理是:
  /// 我如果查询[1, 9]的和,
  /// 那么我其实不需要知道[1, 5]和[6, 9]这两个区间的和分别是什么.
  /// 通理, 更新[6, 7]也只需要更新[6, 9] 和 [1, 9] 的和,
  /// 其余的那些区间, 其实我们只要写成一个懒惰标签即可.
  /// 这个标签的含义, 一言以蔽之, 就是: 在进行查询或者部分更新时,
  /// 务必注意要先完成之前堆积的批量操作:
  /// 在这个区间的位置处的每个元素上加上'x',
  /// 然后在能在当前节点的值的基础上, 得到当前节点的真值.
  void pushDown(size_t left_edge, size_t right_edge, size_t curr_idx)
    requires Accumulateable<DataType>
  {
    /// 验证一下参数的有效性
    assert(left_edge <= right_edge);

    /// 检测懒惰标签是否为幺元, 如果不是开始操作.
    if (lazy_tag[curr_idx] != one) {
      /// 获取懒惰标签的值, 并且将懒惰标签重置为幺元
      auto diff = lazy_tag[curr_idx];
      lazy_tag[curr_idx] = one;

      /// 对当前节点进行更新:
      /// 假设懒惰标签的值是3.
      /// 既然我们都知道, 在[1, 9]的节点存在3的懒惰标签意味着[1, 9]都需要加上3,
      /// 才能得到真实的更新后的值,
      /// 那么, 考虑到这个节点存放的值是[1, 9]的和, 那么在这个长度为9的区间上,
      /// 只要加上 3 * 9 = 27, 就可以得到区间的真实的和.
      data[curr_idx] = data[curr_idx] + diff * (right_edge - left_edge + 1);

      /// 如果左边界等与右边界, 那么意味着我们处在叶子节点, 存放的是单个元素,
      /// 没有孩子节点需要更新了.
      /// 因此在这种情况下, 函数可以直接返回.
      if (right_edge == left_edge) {
        return;
      }

      /// 在其余的情况下, 我们发现当前节点的左右孩子还没有更新,
      /// 那我们需要告诉他们: 他们在当前的状态上, 每个元素都需要额外再加上一个3.
      auto left_child_idx = curr_idx * 2, right_child_idx = curr_idx * 2 + 1;

      /// 这里在更新的时候, 注意使用 +diff 而不是直接 =diff.
      /// 这样的话, 如果左右孩子有遗留的懒惰标签,
      /// 我们也不需要通过更新消除这些遗留的标签,
      /// 而是直接将新的标签附加在旧的上面就可以了.
      lazy_tag[left_child_idx] = lazy_tag[left_child_idx] + diff;
      lazy_tag[right_child_idx] = lazy_tag[right_child_idx] + diff;

      return;
    }
  }

  /// 当累加不能用的时候, 很明显我们也不可能进行区间修改,
  /// 因此这里写个占位的就可以了.
  void pushDown(size_t left_edge, size_t right_edge, size_t curr_idx) {}

  /// 构造线段树
  /// 这里我们使用一个向量去构造我们的线段树.
  /// 我们这里传入的参数分别代表:
  /// 数据, 当前调用覆盖到的数据范围的左边界, 当前覆盖的右边界,
  /// 当前处理的节点的下标
  /// 这里我们默认对象中的两个向量都初始化为幺元了.
  void build(const std::vector<DataType> &vec, size_t left_edge,
             size_t right_edge, size_t curr_idx) {
    /// 简单验证一下参数
    assert(left_edge <= right_edge);

    /// 如果更新到叶子节点了, 这个时候区间长度为1,
    /// 那么左边界就等于右边界. 这就是我们的递归的退出条件.
    if (left_edge == right_edge) {
      /// 此时, 我们将数据写入就好
      data[curr_idx] = vec[left_edge];
      return;
    }

    /// 获取中点, 准备二分.
    auto mid = (left_edge + right_edge) / 2;
    /// 左右孩子对应的节点的下标, 具体为什么孩子节点的下标唯一可以查资料,
    /// 这里解释起来太麻烦了.
    auto left_child_idx = curr_idx * 2, right_child_idx = curr_idx * 2 + 1;

    /// 如果我们需要知道[1, 9]区间的和是多少, 那么我们至少需要知道
    /// [1, 5]和[6, 9]的和是多少. 但是我们现在不知道,
    /// 所以我们先构造两个孩子节点.
    /// 我们递归调用`build`函数, 在期望中,
    /// 它可以完成[1, 5]和[6, 9]区间的所有节点的构造.
    build(vec, left_edge, mid, left_child_idx);
    build(vec, mid + 1, right_edge, right_child_idx);

    /// 在完成构造之后, 我们只需要对[1, 5]和[6, 9]的和相加,
    /// 就可以得到我们需要的[1, 9]的和.
    data[curr_idx] = data[left_child_idx] + data[right_child_idx];

    return;
  }

  /// 线段树的查询操作
  /// 查询操作中, 分别包含了两个区间:
  /// 第一个区间是目标区间, 这个区间始终不变, 代表我们希望通过查找得到和的区间.
  /// 第二个区间是我们当前可以覆盖的区间, 表明这个区间内任何一个子区间的和,
  /// 我们都可以查的到.
  /// 最后我们用一个index表示当前覆盖的区间的总和对应的节点.
  DataType query(size_t left_target, size_t right_target, size_t left_range,
                 size_t right_range, size_t curr_idx) {
    assert(left_range <= right_range);

    /// 我们需要更新待查找区间, 因为我们需要各个总和的节点加载完成.
    /// 加载好了才能进行查询, 因此我们向下把懒惰标记推一层.
    /// 记住下推的实现是对的就可以了, 这里不要管怎么实现.
    pushDown(left_range, right_range, curr_idx);

    /// 如果我们遇到的是这样的情景:
    /// left_target --- left_range --- right_range --- right_target
    /// 那么我们就会发现, 目标的和的一部分就是我们这整个区间的和.
    /// 这个时候, 我们返回整个区间的和就好了.
    if (left_target <= left_range && right_target >= right_range) {
      return data[curr_idx];
    }

    /// 如果我们发现, 目标区间和我们覆盖的区间没有交集
    /// (通常这不应该出现),
    /// 那么直接返回幺元才是明智的做法, 毕竟多加一次幺元对结果也不产生影响.
    if (left_target > right_range || right_target < left_range) {
      return one;
    }

    /// 只覆盖了一个元素的区间, 要么在目标区间内, 要么在外,
    /// 不可能出现这种量子叠加态: 我又在里面又在外面.
    assert(left_range < right_range);

    /// 还是二分......
    auto result = one;
    auto mid = (right_range + left_range) / 2;

    /// 左侧, 如果目标区间和我们的左边有交集, 即:
    /// left_range --- left_target --- mid --- right_target --- right_range
    /// 或者:
    /// left_target --- left_range --- mid --- ...
    /// 总之就是和左孩子管辖的区间有交集了, 我们就查询一下左孩子,
    /// 并且将结果中包含左孩子的查询结果.
    if (left_target <= mid) {
      auto target_idx = curr_idx * 2;
      result = result +
               query(left_target, right_target, left_range, mid, target_idx);
    }

    /// 同理, 如果是右半部分和目标有交集(这和左半部分有交集可以同时存在),
    /// 那么我们就查询右半部分, 并在结果里面体现一下......
    if (right_target >= mid + 1) {
      auto target_idx = curr_idx * 2 + 1;
      result = result + query(left_target, right_target, mid + 1, right_range,
                              target_idx);
    }

    return result;
  }

  /// 线段树的更新
  /// 这个更新是单点更新, 要求直接将旧的某个点的值直接替换成新的值.
  /// 这个更新函数只有在类型参数可以实现减法, 即逆元运算的时候才会生效.
  /// 类型参数不支持的时候, 用户没有办法从公共的接口访问到这个函数.
  /// 这里我们传入的, 分别是:
  /// 新的值, 待修改元素的位置
  /// 当前函数调用能够覆盖的范围, 当前的节点.
  void update(DataType val, size_t pos, size_t curr_left, size_t curr_right,
              size_t curr_idx) {
    /// 如果到达了叶子节点, 就直接将新的值覆盖上去即可.
    if (curr_left == curr_right) {
      assert(curr_right == pos);
      data[curr_idx] = val;
      return;
    }

    /// 准备折半查找
    auto mid = (curr_left + curr_right) / 2, left_child_idx = curr_idx * 2,
         right_child_idx = curr_idx * 2 + 1;
    /// 备份旧的左右孩子节点的值
    auto old_left = data[left_child_idx], old_right = data[right_child_idx];

    /// 更新, 根据`pos`在`mid`的左侧还是右侧决定查找的位置.
    if (pos <= mid) {
      assert(pos >= curr_left);
      update(val, pos, curr_left, mid, left_child_idx);
    } else {
      assert(pos <= curr_right);
      update(val, pos, mid + 1, curr_right, right_child_idx);
    }

    /// 获取新值
    auto left = data[left_child_idx], right = data[right_child_idx];
    /// 获取差值, 事实上总有一边应该差值为0.
    auto update_left = left - old_left, update_right = right - old_right;
    assert((update_left == 0) || (update_right == 0));

    /// 更新.
    auto up = update_left + update_right;
    data[curr_idx] += up;
  }

  /// 增量更新, 包含懒惰标记
  /// 这里除了`diff`代表差值, `curr_idx`代表当前覆盖的区间的和所存放的节点,
  /// 还包含了两个区间:
  /// 目标区间表示区间修改的目标, 也就是说,
  /// diff = 3, left_target = 1, right_target = 9
  /// 代表将下标为[1, 9]的每个元素都加上3
  /// 当前区间代表当前调用可以完成修改的区间.
  /// 我们假设要对[4, 6]进行修改, 那么我们将会遇到:
  /// [1, 9] 分治, 递归调用, 对[1, 5]和[6, 9]进行处理.
  /// [1, 5] 分治得到[1, 3]和[4, 5].
  /// [1, 3] 不参与更新, [4, 5]完全处在目标之中, 给懒惰标记加上3.
  /// [6, 9] 分治得到[6, 7], 然后 [6, 7]分治出[6, 6].
  /// 最后对[6, 6]加上懒惰标记.
  /// 最后递归向上完成上方节点的更新......
  void updateDiff(DataType diff, size_t left_target, size_t right_target,
                  size_t curr_left_edge, size_t curr_right_edge,
                  size_t curr_idx) {
    /// 如果可覆盖的区间完全在目标区间内, 那么直接加上懒惰标记就可以.
    /// 懒惰标记表明: 要得到这里的真值,
    /// 需要给下方的每个元素再加上懒惰标签上的值.
    if (curr_left_edge >= left_target && curr_right_edge <= right_target) {
      lazy_tag[curr_idx] = lazy_tag[curr_idx] + diff;
      return;
    }

    /// 如果覆盖区间和目标区间没有交集
    /// 这种情况本不该发生的......
    if (curr_right_edge < left_target || curr_left_edge > right_target) {
      assert(0);
      return;
    }

    /// 既然又要分治了, 那么就更新一下, 把惰性加载向下推一层.
    /// 永远要记住, 惰性标记应该是一条水平方向上的线,
    /// 也就是任何时候从根到叶子只能碰到一次惰性标签, 不然要乱套.
    pushDown(curr_left_edge, curr_right_edge, curr_idx);

    /// 分治的基本套路, 不讲了......
    auto mid = (curr_left_edge + curr_right_edge) / 2;
    auto left_child_idx = curr_idx * 2, right_child_idx = curr_idx * 2 + 1;

    /// 这里是算出交集的大小的, AI写的, 就姑且看一下吧.
    /// 现在我个人命名风格在向谷歌靠拢, 这种变量还是倾向于用下划线.
    /// 因为是ChatGPT写的, 所以这个其实码风偏向微软.
    /// 不过我可以把它强行解释成函数, 然后依照谷歌对于函数的规范,
    /// 用驼峰命名法.
    auto intervalIntersectionLength = [](size_t a, size_t b, size_t c,
                                         size_t d) -> size_t {
      size_t left = std::max(a, c);   /// 交集区间的左端点
      size_t right = std::min(b, d);  /// 交集区间的右端点

      if (left > right) {
        return 0;  /// 无交集
      }
      return right - left + 1;  /// 交集长度（闭区间包含端点）
    };

    /// 交集的大小
    auto num = intervalIntersectionLength(left_target, right_target,
                                          curr_left_edge, curr_right_edge);

    /// 更新当前的节点, 这里就是算出交集的长度,
    /// 然后将当前的节点的值加上 diff * n
    data[curr_idx] = data[curr_idx] + diff * num;

    /// 分治老套路, 左边和右边, 如果哪边有交集就更新一下哪边孩子.
    if (left_target <= mid) {
      updateDiff(diff, left_target, right_target, curr_left_edge, mid,
                 left_child_idx);
    }

    if (right_target >= mid + 1) {
      updateDiff(diff, left_target, right_target, mid + 1, curr_right_edge,
                 right_child_idx);
    }

    return;
  }

  /// 单点的增量更新, 直接写真值就不用懒加载了.
  /// 还是传统的二分+递归
  /// 有了增量就不用像上次那样更新完叶子才能拿到增量, 然后回溯上来了.
  void updateDiff(DataType diff, size_t pos, size_t curr_left_edge,
                  size_t curr_right_edge, size_t curr_idx) {
    assert(curr_left_edge <= curr_right_edge);

    data[curr_idx] = data[curr_idx] + diff;

    if (curr_right_edge == curr_left_edge) {
      assert(curr_left_edge == pos);
      return;
    }

    auto mid = (curr_left_edge + curr_right_edge) / 2;
    auto left_child_idx = curr_idx * 2, right_child_idx = curr_idx * 2 + 1;

    if (pos <= mid) {
      assert(pos >= curr_left_edge);
      updateDiff(diff, pos, curr_left_edge, mid, left_child_idx);
    } else {
      assert(pos <= curr_right_edge);
      updateDiff(diff, pos, mid + 1, curr_right_edge, right_child_idx);
    }

    return;
  }

 public:
  /// 构造函数, 这里要做的就是把两个数组初始化好, 然后根据向量中的数据建树.
  /// 注意这里传入的向量, 为了好算, 0 节点不存放数据, 从 1 节点开始.
  SimpleSegTree(const std::vector<DataType> &vec, DataType one)
      : size(vec.size() - 1),
        /// 这个数组的长度是计算得到的, 比传统的四倍更精确, 更优化......
        data(1 << (static_cast<size_t>(std::ceil(std::log2(vec.size()))) + 1),
             one),
        lazy_tag(
            1 << (static_cast<size_t>(std::ceil(std::log2(vec.size()))) + 1),
            one),
        one(one) {
    build(vec, 1, size, 1);
  }

  /// 这里是从一个已有的向量构造线段树.
  void build(const std::vector<DataType> &vec) {
    std::fill(data.begin(), data.end(), one);
    std::fill(lazy_tag.begin(), lazy_tag.end(), one);
    build(vec, 1, size, 1);
  }

  /// 这里是查找[left_target, right_target]的下标区间上元素的和.
  DataType query(size_t left_target, size_t right_target) {
    return query(left_target, right_target, 1, size, 1);
  }

  /// 更新, 更新`pos`位置的值为`val`
  /// 需要元素能够求逆元.
  void update(DataType val, size_t pos)
    requires Subtractable<DataType>
  {
    update(val, pos, 1, size, 1);
  }

  /// 区间增量更新, 更新区间内的值为旧的加上`diff`
  /// 需要元素满足累加功能
  void updateDiff(DataType diff, size_t left_target, size_t right_target)
    requires(Accumulateable<DataType>)
  {
    updateDiff(diff, left_target, right_target, 1, size, 1);
  }

  /// 单点增量更新
  void updateDiff(DataType diff, size_t pos) {
    updateDiff(diff, pos, 1, size, 1);
  }
};
