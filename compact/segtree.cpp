#include <cmath>
#include <cstddef>
#include <vector>

template <typename T>
struct SegTree {
  struct Node {
    T val, tag;
  };

  size_t n;
  std::vector<Node> tree;
  T zero;

  SegTree(size_t n, T zero = T{}) : n(n), zero(zero) {
    tree.resize(1 << (static_cast<size_t>(std::ceil(std::log2(n)) + 1)), {zero, zero});
  }

  template <typename Container>
  SegTree(const Container &arr, T zero = T{}) : n(arr.size()), zero(zero) {
    tree.resize(1 << (static_cast<size_t>(std::ceil(std::log2(arr.size())) + 1)), {zero, zero});
    build(arr, 0, n - 1, 1);
  }

  template <typename Container>
  void build(const Container &arr, int l, int r, int p) {
    if (l == r) { tree[p].val = arr[l]; return; }
    int m = (l + r) / 2;
    build(arr, l, m, p * 2);
    build(arr, m + 1, r, p * 2 + 1);
    tree[p].val = tree[p * 2].val + tree[p * 2 + 1].val;
  }

  void pushDown(int p, int l, int r) {
    if (tree[p].tag == zero) return;
    int m = (l + r) / 2;
    tree[p * 2].tag = tree[p * 2].tag + tree[p].tag;
    tree[p * 2 + 1].tag = tree[p * 2 + 1].tag + tree[p].tag;
    tree[p * 2].val = tree[p * 2].val + tree[p].tag * (m - l + 1);
    tree[p * 2 + 1].val = tree[p * 2 + 1].val + tree[p].tag * (r - m);
    tree[p].tag = zero;
  }

  void update(int ql, int qr, T diff, int l, int r, int p) {
    if (ql <= l && r <= qr) {
      tree[p].tag = tree[p].tag + diff;
      tree[p].val = tree[p].val + diff * (r - l + 1);
      return;
    }
    pushDown(p, l, r);
    int m = (l + r) / 2;
    if (ql <= m) update(ql, qr, diff, l, m, p * 2);
    if (qr > m) update(ql, qr, diff, m + 1, r, p * 2 + 1);
    tree[p].val = tree[p * 2].val + tree[p * 2 + 1].val;
  }

  T query(int ql, int qr, int l, int r, int p) {
    if (ql <= l && r <= qr) return tree[p].val;
    pushDown(p, l, r);
    int m = (l + r) / 2;
    T res = zero;
    if (ql <= m) res = res + query(ql, qr, l, m, p * 2);
    if (qr > m) res = res + query(ql, qr, m + 1, r, p * 2 + 1);
    return res;
  }

  void update(int l, int r, T diff) { update(l, r, diff, 0, n - 1, 1); }
  T query(int l, int r) { return query(l, r, 0, n - 1, 1); }
};
