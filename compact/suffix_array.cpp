#include <algorithm>
#include <cstddef>
#include <numeric>
#include <string>
#include <vector>

std::vector<int> suffixArray(const std::string &s) {
  int n = s.size();
  std::vector<int> sa(n), rk(n), nrk(n);
  for (int i = 0; i < n; i++) { sa[i] = i; rk[i] = s[i]; }

  for (int k = 1; k < n; k <<= 1) {
    auto cmp = [&](int a, int b) {
      if (rk[a] != rk[b]) return rk[a] < rk[b];
      int ra = a + k < n ? rk[a + k] : -1;
      int rb = b + k < n ? rk[b + k] : -1;
      return ra < rb;
    };
    std::sort(sa.begin(), sa.end(), cmp);
    nrk[sa[0]] = 0;
    for (int i = 1; i < n; i++)
      nrk[sa[i]] = nrk[sa[i - 1]] + (cmp(sa[i - 1], sa[i]) ? 1 : 0);
    rk = nrk;
    if (rk[sa[n - 1]] == n - 1) break;
  }
  return sa;
}

std::vector<int> lcpArray(const std::string &s, const std::vector<int> &sa) {
  int n = s.size();
  std::vector<int> rk(n), lcp(n - 1);
  for (int i = 0; i < n; i++) rk[sa[i]] = i;
  int h = 0;
  for (int i = 0; i < n; i++) {
    if (rk[i] == 0) continue;
    int j = sa[rk[i] - 1];
    if (h > 0) h--;
    while (i + h < n && j + h < n && s[i + h] == s[j + h]) h++;
    lcp[rk[i] - 1] = h;
  }
  return lcp;
}
