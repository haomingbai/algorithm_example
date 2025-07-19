/**
 * @file concepts.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-07-06
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <concepts>
#include <cstddef>
#include <iterator>
#include <type_traits>

template <typename T>
concept Addable = requires(T a, T b) {
  { a + b } -> std::convertible_to<T>;
};

template <typename T>
concept Subtractable = requires(T a, T b) {
  { a - b } -> std::convertible_to<T>;
};

template <typename T>
concept Multiplyable = requires(T a, T b) {
  { a * b } -> std::convertible_to<T>;
};

template <typename T>
concept Dividable = requires(T a, T b) {
  { a / b } -> std::convertible_to<T>;
};

template <typename Container, typename E>
concept RandomAccessContainer =
    requires(const Container &c, std::size_t index) {
      // 要求有size()成员函数，返回类型可转换为size_t
      { c.size() } -> std::convertible_to<std::size_t>;

      // 要求支持随机访问（下标操作符）
      { c[index] } -> std::convertible_to<E>;
    };

template <typename T1, typename T2>
concept AddableWith = requires(T1 a, T2 b) {
  { a + b } -> std::convertible_to<std::common_type_t<T1, T2>>;
};

template <typename T1, typename T2>
concept SubtractableWith = requires(T1 a, T2 b) {
  { a - b } -> std::convertible_to<std::common_type_t<T1, T2>>;
};

template <typename T1, typename T2>
concept MultiplyableWith = requires(T1 a, T2 b) {
  { a * b } -> std::convertible_to<std::common_type_t<T1, T2>>;
};

template <typename T1, typename T2>
concept DividableWith = requires(T1 a, T2 b) {
  { a / b } -> std::convertible_to<std::common_type_t<T1, T2>>;
};

template <typename T>
concept Accumulateable = requires(T a, size_t b) {
  { a * b } -> std::convertible_to<T>;
};

template <typename T>
concept Partable = requires(T a, size_t b) {
  { a / b } -> std::convertible_to<T>;
};

template <typename T>
concept Negativable = requires(T a) {
  { -a } -> std::convertible_to<T>;
};

template <typename Container, typename E>
concept RandomStdContainer =
    RandomAccessContainer<Container, E> && requires(Container arr) {
      { arr.begin() };
      { arr.end() } -> std::same_as<decltype(arr.begin())>;
      { std::next(arr.begin()) } -> std::same_as<decltype(arr.begin())>;
      { std::prev(arr.end()) } -> std::same_as<decltype(arr.begin())>;
      { arr.begin() < arr.end() } -> std::convertible_to<bool>;
      { arr.empty() } -> std::convertible_to<bool>;
    };

template <typename T>
concept FullyComparable = requires(T a, T b) {
  { a < b } -> std::convertible_to<bool>;
  { a == b } -> std::convertible_to<bool>;
  { a > b } -> std::convertible_to<bool>;
};
