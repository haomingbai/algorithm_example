/**
 * @file mono_stack.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-07-25
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <cstddef>
#include <utility>
#include <vector>

#include "./concepts.cpp"

template <FullyComparable T>
struct MonoStack {
  std::vector<T> dat;

  MonoStack() = default;
  MonoStack(size_t size) { dat.reserve(size); }

  void reserve(size_t n) { dat.reserve(n); }

  // 维持栈顶最大.
  template <typename... Args>
  void push(Args &&...args) {
    T val(std::forward<Args>(args)...);
    while (!dat.empty() && dat.back() < val) {
      dat.pop_back();
    }
    dat.emplace_back(std::move(val));
  }

  T top() {
    return dat.back();
  }

  void pop() {
    dat.pop_back();
  }
};
