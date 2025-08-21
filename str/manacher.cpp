/**
 * @file manacher.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-08-21
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <algorithm>
#include <array>
#include <concepts>
#include <vector>

#include "../concepts.cpp"

struct ManacherResult {
  long start, len;
};

template <std::equality_comparable E, RandomStdContainer<E> Container>
ManacherResult Manacher(Container &&str) {
  if (str.empty()) {
    return {.start = -1, .len = 0};
  }

  if (str.size() == 1) {
    return {.start = 0, .len = 1};
  }

  enum Type : bool { ODD, EVEN };

  std::array<std::vector<long>, 2> palindrome_lens{
      std::vector<long>(str.size(), 1), std::vector<long>(str.size() - 1, 0)};

  // 先讨论奇数长度的最长回文串.
  {
    // 这两个变量表示当前最靠右的回文串的左右边界.
    // 初始状态下, 因为单字符是回文串,
    // 所以二者下标为0, 因为第0个字符为中心的最长回文串就是str[0].
    long L = 0, R = 0;

    for (long i = 1; i < str.size(); i++) {
      // 这个curr_len表示从这个长度开始扩张.
      long curr_len;

      // 分两种情况讨论, 如果被遍历到的i
      // 在当前最右的回文串内部, 那么可以从对称位置开始扩张.
      // 如果不在, 那么直接用朴素算法开始扩张.
      if (i <= R) {
        // 这个时候满足:
        // R - i = mirror_i - L
        auto mirror_i = R - i + L;

        curr_len = std::min(palindrome_lens[ODD][mirror_i], (R - i) * 2 + 1);
      } else {
        curr_len = 1;
      }

      auto curr_stretch = curr_len / 2;

      while (i > curr_stretch && i + curr_stretch + 1 < str.size()) {
        // 尝试把边界延伸1
        // 如果新加入的两个字符相等, 则还能延伸1
        if (str[i + curr_stretch + 1] == str[i - curr_stretch - 1]) {
          curr_stretch++;
        } else {
          break;
        }
      }

      auto curr_l = i - curr_stretch, curr_r = i + curr_stretch;
      // 如果延伸出去了, 就更新L和R
      if (curr_r > R) {
        L = curr_l, R = curr_r;
      }

      // 更新长度.
      curr_len = curr_stretch * 2 + 1;
      palindrome_lens[ODD][i] = curr_len;
    }
  }

  // 偶数长度的串, 情况要复杂很多,
  // 这里朴素算法只能默认从0长度开始.
  {
    long L = 0, R = -1;

    for (long i = 0; i < str.size() - 1; i++) {
      // 这个curr_len表示从这个长度开始扩张.
      long curr_len;

      if (i + 1 <= R) {
        // R - i = mirror_i + 1 - L
        auto mirror_i = R - i + L - 1;
        curr_len = std::min(palindrome_lens[EVEN][mirror_i], (R - i) * 2);
      } else {
        curr_len = 0;
      }

      auto curr_stretch = curr_len / 2 - 1;
      while (i - curr_stretch > 0 && i + curr_stretch + 1 < str.size() - 1) {
        if (str[i + 1 + curr_stretch + 1] == str[i - curr_stretch - 1]) {
          curr_stretch++;
        } else {
          break;
        }
      }

      auto curr_l = i + curr_stretch + 1, curr_r = i - curr_stretch;
      if (curr_r > R) {
        L = curr_l, R = curr_r;
      }

      curr_len = (curr_stretch + 1) * 2;
      palindrome_lens[EVEN][i] = curr_len;
    }
  }

  ManacherResult result;

  {
    long start = 0, len = 1;
    for (long i = 0; i < str.size(); i++) {
      if (palindrome_lens[ODD][i] > len) {
        auto curr_stretch = palindrome_lens[ODD][i] / 2;
        start = i - curr_stretch;
        len = palindrome_lens[ODD][i];
      }
    }
    for (long i = 0; i < str.size() - 1; i++) {
      if (palindrome_lens[EVEN][i] > len) {
        auto curr_stretch = palindrome_lens[EVEN][i] / 2 - 1;
        start = i - curr_stretch;
        len = palindrome_lens[EVEN][i];
      }
    }

    result = {.start = start, .len = len};
  }

  return result;
}
