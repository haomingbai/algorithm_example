#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <optional>

template <typename T, typename Compare = std::less<T>>
struct AVLTree {
  struct Node {
    T val;
    Node *p, *ch[2];
    int h;
    size_t cnt, sz;
    Node(const T &v, Node *p = nullptr) : val(v), p(p), h(0), cnt(1), sz(1) { ch[0] = ch[1] = nullptr; }
    void upH() { h = 1 + std::max(ch[0] ? ch[0]->h : -1, ch[1] ? ch[1]->h : -1); }
    void upS() { sz = (ch[0] ? ch[0]->sz : 0) + (ch[1] ? ch[1]->sz : 0) + cnt; }
    int bf() const { return (ch[0] ? ch[0]->h : -1) - (ch[1] ? ch[1]->h : -1); }
  };

  Node *root = nullptr;
  Compare cmp;

  AVLTree() : root(nullptr) {}
  ~AVLTree() { clear(root); }
  AVLTree(const AVLTree &) = delete;
  AVLTree &operator=(const AVLTree &) = delete;

  void insert(const T &v) {
    if (!root) { root = new Node(v); return; }
    Node *c = root, *fa = nullptr;
    while (true) {
      if (cmp(v, c->val)) {
        c->sz++;
        if (!c->ch[0]) { c->ch[0] = new Node(v, c); fa = c; break; }
        c = c->ch[0];
      } else if (cmp(c->val, v)) {
        c->sz++;
        if (!c->ch[1]) { c->ch[1] = new Node(v, c); fa = c; break; }
        c = c->ch[1];
      } else { c->cnt++; c->sz++; return; }
    }
    fix(fa);
  }

  void remove(const T &v) {
    Node *nd = find(v);
    if (!nd) return;
    if (nd->cnt > 1) { nd->cnt--; for (Node *p = nd; p; p = p->p) p->sz--; return; }
    erase(nd);
  }

  Node *find(const T &v) const {
    Node *c = root;
    while (c) {
      if (cmp(v, c->val)) c = c->ch[0];
      else if (cmp(c->val, v)) c = c->ch[1];
      else return c;
    }
    return nullptr;
  }

  bool contains(const T &v) const { return find(v) != nullptr; }

  size_t count(const T &v) const {
    Node *nd = find(v);
    return nd ? nd->cnt : 0;
  }

  size_t rank(const T &v) const {
    size_t r = 1;
    Node *c = root;
    while (c) {
      if (cmp(v, c->val)) { c = c->ch[0]; }
      else if (cmp(c->val, v)) { r += (c->ch[0] ? c->ch[0]->sz : 0) + c->cnt; c = c->ch[1]; }
      else { r += (c->ch[0] ? c->ch[0]->sz : 0); return r; }
    }
    return r;
  }

  T kth(size_t k) const {
    Node *c = root;
    while (c) {
      size_t ls = c->ch[0] ? c->ch[0]->sz : 0;
      if (k <= ls) c = c->ch[0];
      else if (k > ls + c->cnt) { k -= ls + c->cnt; c = c->ch[1]; }
      else return c->val;
    }
    return T{};
  }

  std::optional<T> pred(const T &v) const {
    Node *c = root; std::optional<T> r;
    while (c) { if (cmp(c->val, v)) { r = c->val; c = c->ch[1]; } else c = c->ch[0]; }
    return r;
  }

  std::optional<T> succ(const T &v) const {
    Node *c = root; std::optional<T> r;
    while (c) { if (cmp(v, c->val)) { r = c->val; c = c->ch[0]; } else c = c->ch[1]; }
    return r;
  }

  size_t size() const { return root ? root->sz : 0; }
  bool empty() const { return !root; }

private:
  void clear(Node *n) { if (!n) return; clear(n->ch[0]); clear(n->ch[1]); delete n; }

  void transplant(Node *old, Node *nw) {
    if (!old->p) root = nw;
    else if (old == old->p->ch[0]) old->p->ch[0] = nw;
    else old->p->ch[1] = nw;
    if (nw) nw->p = old->p;
  }

  void erase(Node *nd) {
    Node *bal = nullptr;
    if (!nd->ch[0] || !nd->ch[1]) {
      bal = nd->p;
      Node *ch = nd->ch[0] ? nd->ch[0] : nd->ch[1];
      transplant(nd, ch);
    } else {
      Node *s = nd->ch[1];
      while (s->ch[0]) s = s->ch[0];
      bal = s->p;
      for (Node *p = s->p; p && p != nd; p = p->p) p->sz -= s->cnt;
      if (s->p != nd) {
        transplant(s, s->ch[1]);
        s->ch[1] = nd->ch[1];
        if (s->ch[1]) s->ch[1]->p = s;
      } else bal = s;
      transplant(nd, s);
      s->ch[0] = nd->ch[0];
      if (s->ch[0]) s->ch[0]->p = s;
      s->upS(); s->upH();
    }
    delete nd;
    if (bal) fix(bal);
  }

  Node *rotate(Node *v, int d) {
    int o = 1 - d;
    Node *u = v->ch[o], *p = v->p;
    v->ch[o] = u->ch[d];
    if (u->ch[d]) u->ch[d]->p = v;
    u->ch[d] = v; v->p = u;
    u->p = p;
    if (p) { if (v == p->ch[0]) p->ch[0] = u; else p->ch[1] = u; }
    else root = u;
    v->upH(); v->upS(); u->upH(); u->upS();
    return u;
  }

  void fix(Node *n) {
    while (n) {
      n->upH(); n->upS();
      int b = n->bf();
      if (b > 1) {
        if (n->ch[0] && n->ch[0]->bf() < 0) rotate(n->ch[0], 0);
        n = rotate(n, 1);
      } else if (b < -1) {
        if (n->ch[1] && n->ch[1]->bf() > 0) rotate(n->ch[1], 1);
        n = rotate(n, 0);
      }
      n = n->p;
    }
  }
};
