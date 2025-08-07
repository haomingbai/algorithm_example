/**
 * @file kmp.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-08-03
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "../concepts.cpp"

size_t PatternMatching(const std::string_view src,
                       const std::string_view pattern);

template <typename E, RandomResizableContainer<size_t> SizeContainer,
          RandomStdContainer<E> CharContainer>
void PrefixFunction(SizeContainer &prefixes, const CharContainer &str) {
  if (str.empty()) {
    return;
  }

  prefixes.resize(str.size(), 0);

  for (size_t i = 1; i < str.size(); i++) {
    auto curr = prefixes[i - 1];
    while (curr != 0 && str[curr] != str[i]) {
      curr = prefixes[curr - 1];
    }
    if (str[curr] == str[i]) {
      curr++;
    }
    prefixes[i] = curr;
  }
}

size_t PatternMatching(const std::string_view src,
                       const std::string_view pattern) {
  if (pattern.empty()) {
    // 匹配失败返回-1是本能.
    return SIZE_MAX;
  }
  std::string str(pattern);
  str.push_back(0);
  str += src;

  std::vector<size_t> prefixes;
  PrefixFunction<char>(prefixes, str);
  for (size_t i = 0, offset = pattern.size() + 1; i < src.size(); i++) {
    if (prefixes[i + offset] >= pattern.size()) {
      // 让返回的下标指向第一次完成匹配的子串的第一个字符.
      return i - (offset - 1);
    }
  }

  // 这就是没有匹配到.
  return SIZE_MAX;
}
