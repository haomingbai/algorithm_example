/**
 * @file trie.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-08-03
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <array>
#include <cassert>
#include <cstddef>
#include <numeric>
#include <string_view>
#include <vector>

struct SimpleTrie {
  static const char start_ = 'a';
  struct TrieNode {
    // next数组代表这条边上下一个节点的下标.
    // weight数组代表边上的权重,
    // 也就是通过这条边的字符串的个数.
    std::array<size_t, 26> next_, weight_;
    // 在这个位置结尾的字符串的数量.
    size_t end_cnt_;

    // 普通的构造函数, 确保数组都被初始化成0了.
    TrieNode() : next_({}), weight_({}), end_cnt_(0) {}
    TrieNode(const TrieNode &) = default;
    TrieNode(TrieNode &&) = default;
  };

  std::vector<TrieNode> nodes_;

  size_t assignNode() {
    auto idx = nodes_.size();
    nodes_.emplace_back();
    return idx;
  }

  SimpleTrie() : nodes_(1) {}

  void reserve(size_t n) { nodes_.reserve(n); }

  void add(const std::string_view str) {
    // 树根在0的位置.
    size_t curr_node_idx = 0;
    for (auto &it : str) {
      // 计算相对起始位字母的偏移量.
      auto offset = it - start_;
      // 若无下一节点则分配新节点.
      // 这里边权才表明路径上是否存在字符串.
      // 哪怕边上挂着节点, 只要边权为0, 那也可能是存过又删光了.
      if (!nodes_[curr_node_idx].next_[offset]) {
        nodes_[curr_node_idx].next_[offset] = assignNode();
      }
      // 边权自增, 表明通过这条边的字符串又多了一条.
      nodes_[curr_node_idx].weight_[offset]++;
      // 通过边移动到下一节点.
      curr_node_idx = nodes_[curr_node_idx].next_[offset];
    }

    // 表明增加了一个串在此结束.
    nodes_[curr_node_idx].end_cnt_++;
  }

  size_t count(const std::string_view str) {
    // 树根在0的位置.
    size_t curr_node_idx = 0;
    for (auto &it : str) {
      auto offset = it - start_;
      // 如果不存在字母对应的边, 那么不存在这样的字符串
      if (!nodes_[curr_node_idx].weight_[offset]) {
        return 0;
      }
      // 顺着边移动到下一节点
      curr_node_idx = nodes_[curr_node_idx].next_[offset];
    }

    // 以这个为前缀的字符串肯定是有了,
    // 但是end_cnt_决定了有几个在这里结尾的字符串.
    return nodes_[curr_node_idx].end_cnt_;
  }

  // 找到前缀满足待匹配字符的个数
  // 如在 "abc", "abcd" 中查找 "ab" 就应该找到2
  size_t countPrefix(const std::string_view prefix) {
    // 所有字符串都可以认为以空字符作为前缀.
    if (prefix.empty()) {
      return nodes_[0].end_cnt_ + std::accumulate(nodes_[0].weight_.begin(),
                                                  nodes_[0].weight_.end(), 0);
    }

    // 树根在0的位置.
    size_t curr_node_idx = 0;

    for (size_t i = 0; i < prefix.length() - 1; i++) {
      auto offset = prefix[i] - start_;
      // 如果不存在字母对应的边, 那么不存在前缀满足的字符串
      if (!nodes_[curr_node_idx].weight_[offset]) {
        return 0;
      }
      // 顺着边移动到下一节点
      curr_node_idx = nodes_[curr_node_idx].next_[offset];
    }

    // 最后一个字母的边的边权就是待查找的
    // 前缀是待匹配字符串的已加入的字符串的数量.
    return nodes_[curr_node_idx].weight_[prefix.back() - start_];
  }

  // 很危险的函数, 不会有任何检查.
  void removeUnchecked(const std::string_view str, size_t num = 1) {
    // 树根在0的位置.
    size_t curr_node_idx = 0;
    for (auto &it : str) {
      auto offset = it - start_;
      // 顺着边移动到下一节点
      nodes_[curr_node_idx].weight_[offset] -= num;
      curr_node_idx = nodes_[curr_node_idx].next_[offset];
    }

    // 如果能减去这么多个串, 那就减掉.
    nodes_[curr_node_idx].end_cnt_ -= num;
  }

  bool remove(const std::string_view str, size_t num = 1) {
    if (count(str) >= num) {
      removeUnchecked(str, num);
      return true;
    } else {
      return false;
    }
  }
};
