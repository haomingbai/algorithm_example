#include <array>
#include <cstddef>
#include <string>
#include <vector>

template <char START = 'a', size_t ALPH = 26>
struct Trie {
  struct Node {
    std::array<int, ALPH> ch;
    int cnt;
    Node() : ch(), cnt(0) { ch.fill(-1); }
  };

  std::vector<Node> nodes;

  Trie() : nodes(1) {}

  void insert(const std::string &s) {
    int c = 0;
    for (char ch : s) {
      int x = ch - START;
      if (nodes[c].ch[x] == -1) {
        nodes[c].ch[x] = nodes.size();
        nodes.emplace_back();
      }
      c = nodes[c].ch[x];
    }
    nodes[c].cnt++;
  }

  int count(const std::string &s) {
    int c = 0;
    for (char ch : s) {
      int x = ch - START;
      if (nodes[c].ch[x] == -1) return 0;
      c = nodes[c].ch[x];
    }
    return nodes[c].cnt;
  }

  bool contains(const std::string &s) { return count(s) > 0; }

  bool hasPrefix(const std::string &s) {
    int c = 0;
    for (char ch : s) {
      int x = ch - START;
      if (nodes[c].ch[x] == -1) return false;
      c = nodes[c].ch[x];
    }
    return true;
  }
};
