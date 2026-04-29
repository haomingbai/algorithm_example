#include <cstddef>
#include <vector>

struct PersistentSegTree {
  struct Node {
    long long sum;
    int l, r;
    Node() : sum(0), l(0), r(0) {}
  };

  std::vector<Node> tree;
  std::vector<int> roots;
  int n;

  PersistentSegTree(int n) : n(n) {
    tree.reserve(n * 20);
    tree.emplace_back();
    roots.push_back(0);
  }

  int newNode() {
    tree.emplace_back();
    return tree.size() - 1;
  }

  int build(int l, int r) {
    int p = newNode();
    if (l == r) return p;
    int m = (l + r) / 2;
    tree[p].l = build(l, m);
    tree[p].r = build(m + 1, r);
    return p;
  }

  int update(int prev, int l, int r, int pos, int val) {
    int p = newNode();
    tree[p] = tree[prev];
    tree[p].sum += val;
    if (l == r) return p;
    int m = (l + r) / 2;
    if (pos <= m) tree[p].l = update(tree[prev].l, l, m, pos, val);
    else tree[p].r = update(tree[prev].r, m + 1, r, pos, val);
    return p;
  }

  long long query(int u, int v, int l, int r, int ql, int qr) {
    if (ql <= l && r <= qr) return tree[v].sum - tree[u].sum;
    int m = (l + r) / 2;
    long long res = 0;
    if (ql <= m) res += query(tree[u].l, tree[v].l, l, m, ql, qr);
    if (qr > m) res += query(tree[u].r, tree[v].r, m + 1, r, ql, qr);
    return res;
  }

  void add(int pos, int val) {
    roots.push_back(update(roots.back(), 0, n - 1, pos, val));
  }

  long long query(int version_l, int version_r, int ql, int qr) {
    return query(roots[version_l], roots[version_r], 0, n - 1, ql, qr);
  }
};

struct KthTree {
  struct Node {
    int cnt, l, r;
    Node() : cnt(0), l(0), r(0) {}
  };

  std::vector<Node> tree;
  std::vector<int> roots;
  int max_val;

  KthTree(int max_val) : max_val(max_val) {
    tree.reserve(max_val * 20);
    tree.emplace_back();
    roots.push_back(0);
  }

  int newNode() {
    tree.emplace_back();
    return tree.size() - 1;
  }

  int update(int prev, int l, int r, int pos) {
    int p = newNode();
    tree[p] = tree[prev];
    tree[p].cnt++;
    if (l == r) return p;
    int m = (l + r) / 2;
    if (pos <= m) tree[p].l = update(tree[prev].l, l, m, pos);
    else tree[p].r = update(tree[prev].r, m + 1, r, pos);
    return p;
  }

  int kth(int u, int v, int l, int r, int k) {
    if (l == r) return l;
    int m = (l + r) / 2;
    int cnt = tree[tree[v].l].cnt - tree[tree[u].l].cnt;
    if (k < cnt) return kth(tree[u].l, tree[v].l, l, m, k);
    return kth(tree[u].r, tree[v].r, m + 1, r, k - cnt);
  }

  void add(int val) {
    roots.push_back(update(roots.back(), 0, max_val, val));
  }

  int kth(int l, int r, int k) {
    return kth(roots[l], roots[r + 1], 0, max_val, k);
  }
};
