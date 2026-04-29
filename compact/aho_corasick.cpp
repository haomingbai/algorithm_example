#include <array>
#include <cstddef>
#include <queue>
#include <string>
#include <string_view>
#include <vector>

template <char START = 'a', size_t ALPH = 26>
struct AhoCorasick {
  struct Node {
    std::array<int, ALPH> next;
    int fail;
    std::vector<int> out;
    Node() : next(), fail(0) { next.fill(-1); }
  };

  std::vector<Node> nodes;
  std::vector<std::string> pats;
  bool built = false;

  AhoCorasick() : nodes(1) {}

  static int c2i(char c) {
    int idx = (unsigned char)c - (unsigned char)START;
    if (idx >= 0 && (size_t)idx < ALPH) return idx;
    if constexpr (START == 'a' && ALPH == 26) {
      if (c >= 'A' && c <= 'Z') return c - 'A';
    }
    return -1;
  }

  int add(std::string_view s) {
    built = false;
    int c = 0;
    for (char ch : s) {
      int x = c2i(ch);
      if (x < 0) continue;
      if (nodes[c].next[x] == -1) {
        nodes[c].next[x] = nodes.size();
        nodes.emplace_back();
      }
      c = nodes[c].next[x];
    }
    int pid = pats.size();
    pats.emplace_back(s);
    nodes[c].out.push_back(pid);
    return pid;
  }

  void build() {
    if (built) return;
    std::queue<int> q;
    nodes[0].fail = 0;
    for (size_t c = 0; c < ALPH; c++) {
      if (nodes[0].next[c] != -1) {
        nodes[nodes[0].next[c]].fail = 0;
        q.push(nodes[0].next[c]);
      } else {
        nodes[0].next[c] = 0;
      }
    }
    while (!q.empty()) {
      int u = q.front(); q.pop();
      for (size_t c = 0; c < ALPH; c++) {
        int v = nodes[u].next[c];
        if (v != -1) {
          nodes[v].fail = nodes[nodes[u].fail].next[c];
          auto &dst = nodes[v].out;
          const auto &src = nodes[nodes[v].fail].out;
          dst.insert(dst.end(), src.begin(), src.end());
          q.push(v);
        } else {
          nodes[u].next[c] = nodes[nodes[u].fail].next[c];
        }
      }
    }
    built = true;
  }

  struct Match { int pid, pos; };

  std::vector<Match> matchAll(std::string_view text) {
    if (!built) build();
    std::vector<Match> res;
    int c = 0;
    for (size_t i = 0; i < text.size(); i++) {
      int x = c2i(text[i]);
      if (x < 0) { c = 0; continue; }
      c = nodes[c].next[x];
      for (int pid : nodes[c].out) {
        res.push_back({pid, (int)(i + 1) - (int)pats[pid].size()});
      }
    }
    return res;
  }
};

template <char START = 'a', size_t ALPH = 26>
struct AhoCorasickTopo {
  struct Node {
    std::array<int, ALPH> next;
    int fail;
    Node() : next(), fail(0) { next.fill(-1); }
  };

  std::vector<Node> nodes;
  std::vector<int> pat_end;
  std::vector<int> topo;
  bool built = false;

  AhoCorasickTopo() : nodes(1) {}

  static int c2i(char c) {
    int idx = (unsigned char)c - (unsigned char)START;
    if (idx >= 0 && (size_t)idx < ALPH) return idx;
    if constexpr (START == 'a' && ALPH == 26) {
      if (c >= 'A' && c <= 'Z') return c - 'A';
    }
    return -1;
  }

  int add(std::string_view s) {
    built = false;
    int c = 0;
    for (char ch : s) {
      int x = c2i(ch);
      if (x < 0) continue;
      if (nodes[c].next[x] == -1) {
        nodes[c].next[x] = nodes.size();
        nodes.emplace_back();
      }
      c = nodes[c].next[x];
    }
    int pid = pat_end.size();
    pat_end.push_back(c);
    return pid;
  }

  void build() {
    if (built) return;
    std::queue<int> q;
    nodes[0].fail = 0;
    for (size_t c = 0; c < ALPH; c++) {
      if (nodes[0].next[c] != -1) {
        nodes[nodes[0].next[c]].fail = 0;
        q.push(nodes[0].next[c]);
      } else {
        nodes[0].next[c] = 0;
      }
    }
    while (!q.empty()) {
      int u = q.front(); q.pop();
      for (size_t c = 0; c < ALPH; c++) {
        int v = nodes[u].next[c];
        if (v != -1) {
          nodes[v].fail = nodes[nodes[u].fail].next[c];
          q.push(v);
        } else {
          nodes[u].next[c] = nodes[nodes[u].fail].next[c];
        }
      }
    }
    size_t n = nodes.size();
    std::vector<int> indeg(n, 0);
    for (size_t i = 1; i < n; i++) indeg[nodes[i].fail]++;
    for (size_t i = 0; i < n; i++) if (indeg[i] == 0) q.push(i);
    topo.clear(); topo.reserve(n);
    while (!q.empty()) {
      int u = q.front(); q.pop();
      topo.push_back(u);
      if (u == 0) continue;
      if (--indeg[nodes[u].fail] == 0) q.push(nodes[u].fail);
    }
    built = true;
  }

  std::vector<long long> count(std::string_view text) {
    if (!built) build();
    std::vector<long long> occ(nodes.size(), 0);
    int c = 0;
    for (char ch : text) {
      int x = c2i(ch);
      if (x < 0) { c = 0; continue; }
      c = nodes[c].next[x];
      occ[c]++;
    }
    for (int u : topo) {
      if (u == 0) continue;
      occ[nodes[u].fail] += occ[u];
    }
    std::vector<long long> res(pat_end.size(), 0);
    for (size_t i = 0; i < pat_end.size(); i++) res[i] = occ[pat_end[i]];
    return res;
  }
};
