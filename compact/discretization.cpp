#include <algorithm>
#include <cstddef>
#include <vector>

struct Discretization {
  std::vector<long long> vals;

  template <typename Container>
  Discretization(const Container &a) {
    vals.assign(a.begin(), a.end());
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
  }

  int hash(long long v) {
    auto it = std::lower_bound(vals.begin(), vals.end(), v);
    if (it == vals.end() || *it != v) return -1;
    return it - vals.begin();
  }

  long long dehash(int idx) { return vals[idx]; }
  int size() const { return vals.size(); }
};
