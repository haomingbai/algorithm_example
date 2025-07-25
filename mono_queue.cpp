/**
 * @file mono_queue.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-07-25
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <deque>       // 使用双端队列作为单调队列的底层容器
#include <functional>  // 用于 std::less 和 std::greater
#include <iostream>
#include <vector>  // 用于测试用例

/**
 * @brief 单调队列模板类 (Monotonic Queue)
 * @tparam T 队列中存储的元素类型
 * @tparam Compare 比较函数对象，用于定义单调性。
 * - std::less<T> (默认): 维护一个单调递增队列 (队首最小)。
 * - std::greater<T>: 维护一个单调递减队列 (队首最大)。
 */
template <typename T, typename Compare = std::less<T>>
class MonoQueue {
 public:
  // --- 数据成员 (Data Members) ---
  // 按照竞赛模板风格，将底层容器设为公有，方便直接访问或调试。

  /**
   * @brief 底层双端队列 (deque)
   * @details
   * 使用 std::deque 是因为它支持高效的队首和队尾的插入与删除操作。
   * - push(): 从队尾操作。
   * - pop(): 从队首操作。
   * - front(): 访问队首。
   */
  std::deque<T> q;

  // --- 核心修改操作 (Core Modifiers) ---

  /**
   * @brief 将一个新元素推入队列尾部，同时维持其单调性。
   * @param value 要推入的新元素。
   * @details
   * 这是单调队列最核心的操作。为了保持单调性，在插入新元素 `value` 之前，
   * 它会从队尾开始，持续移除所有“破坏”单调规则的旧元素。
   * * 例如，对于一个单调递减队列（用于求窗口最大值）：
   * 如果队尾元素比 `value` 小，说明它不可能再成为未来任何窗口的最大值了
   * (因为 `value` 不仅比它大，还比它晚进入窗口)，所以可以安全地从队尾移除。
   */
  void push(const T& value) {
    // 实例化比较器，用于判断元素间的顺序。
    Compare comp;

    // --- 核心逻辑：维持单调性 ---
    // 当队列不为空，且队尾元素根据比较规则应该被新元素替换时，
    // (例如，对于单调递减队列 comp(q.back(), value) 即 q.back() < value 为真时)
    // 就将队尾元素弹出。
    while (!q.empty() && comp(q.back(), value)) {
      q.pop_back();
    }

    // 将新元素加入队尾。
    // 经过上面的循环，队列的单调性得到了保证。
    q.push_back(value);
  }

  /**
   * @brief 从队首移除一个元素。
   * @details
   * 在滑动窗口的应用中，当窗口向右移动时，最左侧的元素需要被考虑是否移除。
   * 注意：这个函数只负责移除队首元素。调用者需要自行判断队首元素是否
   * 等于滑出窗口的那个元素。
   */
  void pop() {
    // --- 核心逻辑：移除队首 ---
    // 确保队列不为空，防止对空队列执行 pop 操作导致未定义行为。
    if (!q.empty()) {
      q.pop_front();
    }
  }

  // --- 访问操作 (Accessors) ---

  /**
   * @brief 返回对队首元素的常引用。
   * @return const T& 队首元素的引用。
   * @details
   * 由于队列的单调性，队首元素始终是当前队列（窗口）中的最值。
   * - 单调递增队列: 返回最小值。
   * - 单调递减队列: 返回最大值。
   */
  const T& front() const {
    // 直接返回底层 deque 的队首元素。
    // 调用者应确保队列不为空。
    return q.front();
  }

  /**
   * @brief 返回对队尾元素的常引用。
   * @return const T& 队尾元素的引用。
   */
  const T& back() const {
    // 直接返回底层 deque 的队尾元素。
    // 调用者应确保队列不为空。
    return q.back();
  }

  // --- 容量/状态查询 (Capacity/Status) ---

  /**
   * @brief 检查队列是否为空。
   * @return bool 如果队列为空返回 true，否则返回 false。
   */
  bool empty() const { return q.empty(); }

  /**
   * @brief 返回队列中元素的数量。
   * @return size_t 队列中的元素个数。
   */
  size_t size() const { return q.size(); }
};

// --- 使用示例：解决滑动窗口最大值问题 ---
void slidingWindowMaximumExample() {
  std::cout << "--- 滑动窗口最大值示例 ---" << std::endl;

  // 输入数据
  std::vector<int> nums = {1, 3, -1, -3, 5, 3, 6, 7};
  int k = 3;  // 窗口大小

  std::cout << "输入数组: ";
  for (int num : nums) std::cout << num << " ";
  std::cout << "\n窗口大小 k = " << k << std::endl;

  // 创建一个单调递减队列，用于寻找窗口内的最大值。
  // 使用 std::greater<int> 作为比较器。
  MonoQueue<int, std::greater<int>> mq;
  std::vector<int> results;

  // 遍历输入数组
  for (int i = 0; i < nums.size(); ++i) {
    // --- 步骤 1: 将当前元素推入单调队列 ---
    // push 操作会自动维护队列的单调递减性。
    mq.push(nums[i]);

    // --- 步骤 2: 检查窗口是否形成 ---
    // 当遍历过的元素数量达到 k 时，第一个窗口形成了。
    if (i >= k - 1) {
      // --- 步骤 2a: 获取当前窗口的最大值 ---
      // 队首元素 mq.front() 就是当前窗口的最大值。
      results.push_back(mq.front());

      // --- 步骤 2b: 准备下一次滑动 ---
      // 检查队首元素是否是即将滑出窗口的那个元素。
      // 即将滑出窗口的元素是 nums[i - k + 1]。
      if (mq.front() == nums[i - k + 1]) {
        mq.pop();
      }
    }
  }

  std::cout << "所有窗口的最大值: ";
  for (int res : results) {
    std::cout << res << " ";
  }
  std::cout << std::endl << std::endl;
}
