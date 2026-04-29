#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

template <typename T>
struct SparseTable {
  std::vector<std::vector<T>> st;
  std::function<T(T, T)> op;
  int n;

  template <typename Container>
  SparseTable(const Container &a, std::function<T(T, T)> op = [](T x, T y) { return std::max(x, y); })
      : op(op), n(a.size()) {
    int k = std::log2(n) + 1;
    st.assign(n, std::vector<T>(k));
    for (int i = 0; i < n; i++) st[i][0] = a[i];
    for (int j = 1; j < k; j++)
      for (int i = 0; i + (1 << j) <= n; i++)
        st[i][j] = op(st[i][j - 1], st[i + (1 << (j - 1))][j - 1]);
  }

  T query(int l, int r) {
    int k = std::log2(r - l + 1);
    return op(st[l][k], st[r - (1 << k) + 1][k]);
  }
};
