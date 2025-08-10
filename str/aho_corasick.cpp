/**
 * @file aho_corasick.cpp
 * @brief Aho-Corasick 自动机的 C++ 实现
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-08-07
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 * 这是一个功能完整的 Aho-Corasick
 * 自动机实现, 用于在一段文本中高效地同时查找多个模式串.  主要特性:
 * 1. 使用 Trie (字典树) 结构存储模式串集合.
 * 2. 通过 BFS 构建失败指针 (Fail Pointers) , 处理匹配失败时的快速跳转.
 * 3. 在构建失败指针后, 将 Trie
 * 图优化为确定性有限状态自动机 (DFA) , 使得匹配文本时每个字符的转移成本为
 * O(1).
 * 4. 支持查找所有匹配, 查找第一个匹配以及使用回调函数处理每个匹配.
 * 5. 字母表可配置 (当前为 'a'-'z') .
 */

#include <array>
#include <cassert>
#include <cstddef>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// 匹配结果的结构体
struct AcMatchResult {
  // str_idx : 匹配到的模式串在其集合中的 ID (0-based)
  size_t str_idx;
  // pos : 匹配项在被搜索文本中的起始位置 (0-based)
  size_t pos;
};

// start_: 字母表的起始位置
// size_: 字母表的大小
template <char start_, size_t size_>
struct AhoCorasick {
  // AC 自动机的 Trie 节点定义
  struct AcTrieNode {
    // next_ : 指向子节点的指针数组. 下标由字符在字母表中的偏移量决定.
    // 值为 0 表示没有对应的子节点 (在 buildFail 完成前) .
    // 在 buildFail 完成后, 0 表示跳转回根节点.
    std::array<size_t, size_> next_;
    // outputs_ : 存储在此节点结束的模式串的 ID (pattern id).
    // 一个节点可能对应多个模式串的结束 (例如模式串 "a", "ba" 都以 'a' 结尾) .
    std::vector<size_t> outputs_;
    // fail_ :
    // 失败指针, 指向的节点代表当前字符串的最长真后缀,
    // 该后缀同时也是所有模式串的前缀. 这是 AC 自动机的核心,
    // 用于在失配时进行高效跳转.
    size_t fail_;

    AcTrieNode() : next_(), outputs_(), fail_(0) { next_.fill(0); }
    AcTrieNode(const AcTrieNode &) = default;
    AcTrieNode(AcTrieNode &&) = default;
  };

  // nodes_ : 节点存储区. nodes_[0] 被保留为根节点.
  // 这是一个 "arena" 或者 "pool", 所有节点都在这里分配.
  std::vector<AcTrieNode> nodes_;
  // strs_ : 存储所有添加的模式串. 其下标即为模式串的 ID (pid).
  std::vector<std::string> strs_;
  // fail_flag_ : 一个 "脏" 标记, 用于指示失败指针是否已经构建.
  // 添加新字符串后, 此标记会变为 false, 需要在匹配前重新构建.
  bool fail_flag_ = false;

  AhoCorasick() : nodes_(1 /* root */), strs_() {}

  // 分配一个新节点, 并返回其在 nodes_ 数组中的下标
  inline size_t assignNode() {
    auto res = nodes_.size();
    nodes_.emplace_back();
    return res;
  }

  // 为外部调用者提供一个接口, 可以提前为节点分配内存, 以避免在 addStr
  // 过程中发生多次 vector 扩容.
  void reserve(size_t n) { nodes_.reserve(n); }

  // 将字符 c 转换为 next_ 数组的索引 (0..size_-1).
  // @return 如果字符在定义的字母表范围内, 返回 true 并通过 out
  // 参数传出索引；否则返回 false.
  static inline bool char_to_index(char c, size_t &out) {
    // 使用 unsigned char 进行转换, 以防止在某些平台上 char 为有符号类型时,
    // 负值字符 (如 ASCII > 127) 导致未定义行为.
    unsigned char uc = static_cast<unsigned char>(c);
    unsigned char base = static_cast<unsigned char>(start_);
    if (uc >= base && uc < static_cast<unsigned char>(base + size_)) {
      out = static_cast<size_t>(uc - base);
      return true;
    }
    return false;
  }

  /**
   * @brief 向 Trie 树中添加一个模式串.
   * @param str 要添加的模式串.
   * @return 该模式串的 ID (0-based), 可用于在匹配结果中识别.
   * @note 添加任何字符串都会导致 fail_flag_ 被设置为 false,
   * 意味着在下一次匹配前必须重新调用 buildFail().
   * @throw std::invalid_argument 如果字符串中包含不在字母表内的字符.
   * @complexity O(L), 其中 L 是模式串 str 的长度.
   */
  size_t addStr(std::string_view str) {
    fail_flag_ = false;  // 标记为 "脏" , 需要重新构建失败指针
    size_t curr = 0;     // 从根节点开始
    for (const auto &ch : str) {
      size_t offset;
      if (!char_to_index(ch, offset)) {
        // 如果模式串包含无效字符, 则抛出异常.
        // 也可以根据需求改为忽略或跳过.
        throw std::invalid_argument("addStr: character out of alphabet range");
      }
      if (nodes_[curr].next_[offset] == 0) {
        // 如果路径不存在, 则创建新节点
        nodes_[curr].next_[offset] = assignNode();
      }
      curr = nodes_[curr].next_[offset];
    }
    // 模式串的 ID 就是它在 strs_ 数组中的索引
    size_t pid = strs_.size();
    strs_.emplace_back(str);
    // 将模式串 ID 添加到其结束节点的 outputs_ 列表中
    nodes_[curr].outputs_.push_back(pid);
    return pid;
  }

  /**
   * @brief 构建失败指针和优化 Trie 图. 这是 AC 自动机的核心步骤.
   * @details
   * 此函数完成两项主要工作:
   * 1.  **构建失败指针 (Fail Links)**: 使用广度优先搜索 (BFS) 为每个节点计算其
   * fail_ 指针.  同时, 将 fail_ 指向节点的 outputs_
   * 合并到当前节点, 确保匹配时不会遗漏任何子串匹配.
   * 2.  **构建DFA (Deterministic Finite Automaton)**: 填充 Trie 中缺失的 next_
   * 转移.  对于任意节点 v 和字符 c, 如果 v 没有 c 的转移, 则令 v 的 c 转移指向
   * v.fail 的 c 转移.  这使得匹配时不再需要循环查找 fail 指针, 每次转移都是
   * O(1) 的.
   * @complexity O(N * A), 其中 N 是所有模式串的总长度, A 是字母表大小.
   */
  void buildFail() {
    // 如果已经构建过失败指针/dfa, 直接返回 (避免重复合并 outputs)
    // 这一检查同时避免多次调用时 outputs 重复插入的问题.
    if (fail_flag_) {
      return;
    }

    // 如果没有任何节点 (理论上不会发生, 因为构造函数至少创建了根) ,
    // 仍然把标志置为已构建并返回, 保证后续 match_* 不会报错.
    if (nodes_.empty()) {
      assert(0);
      fail_flag_ = true;
      return;
    }

    // 根节点的失败指针指向自己 (用 0 表示根) .
    // 这样在后面查找 fail 链时可以安全地回退到 0 而不越界.
    nodes_[0].fail_ = 0;

    std::queue<size_t> q;

    // --- 初始化第一层子节点 ---
    // 把根节点的所有直接子节点 (depth == 1) 放入队列, 作为 bfs 的起点.
    // 同时确保根节点的缺失转移显式为 0 (方便后续在 bfs 中引用) .
    for (size_t ch = 0; ch < size_; ch++) {
      size_t nxt = nodes_[0].next_[ch];
      if (nxt) {
        // 第一层节点的 fail 指向根
        nodes_[nxt].fail_ = 0;
        q.push(nxt);
      } else {
        // 对于根, 缺失的转移继续保持为 0 (根的 next_[ch] == 0 表示回到根)
        // 这样在后面用 nodes_[nodes_[parent].fail_].next_[ch] 时不会越界.
        nodes_[0].next_[ch] = 0;
      }
    }

    // --- bfs 构建 fail 指针并同时在 bfs 中填补缺失转移 (将 trie 优化为
    // dfa) --- bfs 保证我们先处理浅层节点 (短前缀) , 因此当我们在处理某个节点
    // parent 时,  nodes_[ parent ].fail_ 指向的节点的 next_
    // 表已经准备好, 可以安全引用.
    while (!q.empty()) {
      auto parent = q.front();
      q.pop();

      // 遍历整个字母表来处理 parent 的每个字符转移
      for (size_t ch = 0; ch < size_; ch++) {
        auto nxt = nodes_[parent].next_[ch];

        if (nxt) {
          // 情况 a: parent 有一条实际的子边 nxt (trie 中存在的边)
          // 需要为 nxt 计算 fail 指针:
          // 从 parent 的 fail 开始沿 fail 链上查找第一个对 ch 有转移的节点.
          size_t f = nodes_[parent].fail_;
          while (f && !nodes_[f].next_[ch]) {
            // 继续向上回退, 直到根 (0) 或找到有 ch 转移的节点
            f = nodes_[f].fail_;
          }

          // 如果找到节点 f 的 next_[ch] 非 0, 则 nxt.fail
          // 指向它；否则指向根 (0).  这保证了 nxt.fail
          // 指向当前状态的最长真后缀状态.
          if (nodes_[f].next_[ch]) {
            f = nodes_[f].next_[ch];
          } else {
            assert(!f);
          }
          nodes_[nxt].fail_ = f;

          // 合并输出列表: 把 fail 节点的 outputs 合并到当前节点的 outputs 中.
          // 这样在匹配时, 仅检查当前节点的 outputs
          // 就能报告包括后缀在内的所有匹配.  注意: 由于我们在函数开头用
          // fail_flag_ 防止重复运行, 这里不会造成重复合并.
          // 这一步骤成立的原因是这里和fail共享相同的后缀,
          // 确切来讲, 这里的后缀就有是fail处结尾的全部字符串.
          // 具体的原因可以查找KMP章节,
          // 深入理解oi-wiki中前缀函数的第二次优化的段落.
          // 如果一个字符串在某一节点匹配到了"abcabcabc",
          // 它一定能匹配到fail位置对应的 "abcabc"
          // 和fail位置节点的fail指针指向的 "abc"
          auto &dst = nodes_[nxt].outputs_;
          const auto &src = nodes_[nodes_[nxt].fail_].outputs_;
          dst.insert(dst.end(), src.begin(), src.end());

          // 把子节点加入 bfs 队列继续向下处理
          q.push(nxt);
        } else {
          // 情况 b: parent 在 trie 中**没有** ch 的子边 (缺失转移)
          // 为了把 trie 转换为 dfa, 我们把缺失的转移直接填补为
          // parent.fail 的 ch 转移 (因为在失配时会回退到 fail, 再用 fail
          // 的转移) .  因为我们是按 bfs (由浅到深) 处理节点,
          // nodes_[ parent].fail_ 的 next_ 已经被准备好了
          // (或者在根处为0) , 所以这个赋值是安全的.
          // 这里应该是可以避免一种转移, 也就是说,
          // 哪怕模式串里面不存在 "abcabc",
          // 只存在 "abcab" 和 "abc",
          // 这里让"abcabc"匹配到 "abcab" 之后也可以直接转移到 "abcabc",
          // 进而直接匹配到 "abc".
          // 如果没有这一步, 那么在模式匹配的算法中就要反复地进行fail转移.
          // 有了这一步, next可以直接向前转移, 精妙世无双.
          nodes_[parent].next_[ch] = nodes_[nodes_[parent].fail_].next_[ch];
        }
      }
    }

    // 标记构建完成, 后续 match_* 可以安全使用 dfa 转移和合并后的 outputs.
    fail_flag_ = true;
  }

  // 检查自动机是否已构建
  bool built() const { return fail_flag_; }

  // 根据 ID 获取原始模式串
  const std::string &getString(size_t idx) const { return strs_[idx]; }

  /**
   * @brief 在文本中查找所有匹配的模式串.
   * @param text 要搜索的文本.
   * @return 包含所有匹配结果的 vector.
   * @throw std::logic_error 如果在调用前没有构建失败指针 (即未调用 buildFail).
   * @complexity O(M + K), 其中 M 是文本长度, K 是报告的匹配总数.
   */
  std::vector<AcMatchResult> match_all(std::string_view text) const {
    if (!built()) {
      throw std::logic_error("AhoCorasick::match_all: buildFail not called");
    }
    std::vector<AcMatchResult> res;
    size_t node = 0;  // 从根节点开始
    for (size_t i = 0; i < text.size(); i++) {
      size_t idx;
      if (!char_to_index(text[i], idx)) {
        // 如果遇到不在字母表中的字符, 直接回到根状态.
        node = 0;
        continue;
      }

      // 因为已经优化为 DFA, 所以可以直接进行状态转移, 无需处理失配.
      // 每个字符都对应一个 O(1) 的转移.
      node = nodes_[node].next_[idx];

      // 检查当前状态是否有输出.
      // 因为 outputs_ 已经合并, 所以这里包含了所有以当前位置结尾的匹配.
      for (auto pid : nodes_[node].outputs_) {
        size_t plen = strs_[pid].size();
        // 计算匹配的起始位置:
        // i 是匹配结束字符的 0-based 索引.
        // (i + 1) 是匹配结束字符后的位置.
        // (i + 1) - plen 就是起始位置的 0-based 索引.
        size_t start_pos = (i + 1) - plen;
        res.push_back({pid, start_pos});
      }
    }
    return res;
  }

  /**
   * @brief 使用回调函数处理每一个匹配项, 避免一次性创建巨大的结果数组.
   * @tparam Callback 可调用对象, 签名应兼容 void(size_t pattern_id, size_t
   * start_pos).
   * @param text 要搜索的文本.
   * @param cb 回调函数.
   * @complexity O(M + K), 同 match_all.
   */
  template <typename Callback>
  void match_callback(std::string_view text, Callback cb) const {
    if (!built()) {
      throw std::logic_error(
          "AhoCorasick::match_callback: buildFail not called");
    }
    size_t node = 0;
    for (size_t i = 0; i < text.size(); i++) {
      size_t idx;
      if (!char_to_index(text[i], idx)) {
        node = 0;
        continue;
      }
      node = nodes_[node].next_[idx];
      for (auto pid : nodes_[node].outputs_) {
        size_t plen = strs_[pid].size();
        size_t start_pos = (i + 1) - plen;
        cb(pid, start_pos);
      }
    }
  }

  /**
   * @brief 查找第一个出现的匹配项.
   * @param text 要搜索的文本.
   * @param out 如果找到匹配, 结果将通过此引用参数返回.
   * @return 如果找到任何匹配, 则返回 true；否则返回 false.
   * @details 返回的是按文本结束位置最早出现的匹配.
   */
  bool match_first(std::string_view text, AcMatchResult &out) const {
    if (!built()) {
      throw std::logic_error("AhoCorasick::match_first: buildFail not called");
    }
    size_t node = 0;
    for (size_t i = 0; i < text.size(); i++) {
      size_t idx;
      if (!char_to_index(text[i], idx)) {
        node = 0;
        continue;
      }
      node = nodes_[node].next_[idx];
      if (!nodes_[node].outputs_.empty()) {
        // 找到第一个, 立即报告并返回
        size_t pid = nodes_[node].outputs_.front();
        size_t plen = strs_[pid].size();
        size_t start_pos = (i + 1) - plen;
        out = {pid, start_pos};
        return true;
      }
    }
    return false;
  }

  // 清空自动机, 重置为初始状态.
  void clear() {
    nodes_.clear();
    nodes_.emplace_back();  // 重新创建根节点
    strs_.clear();
    fail_flag_ = false;
  }
};
