#include <algorithm>
#include <string>
#include <vector>

struct ManacherResult { int start, len; };

ManacherResult manacher(const std::string &s) {
  if (s.empty()) return {-1, 0};
  int n = s.size();
  std::string t;
  t.reserve(2 * n + 3);
  t += "^#";
  for (char c : s) { t += c; t += '#'; }
  t += '$';
  int m = t.size();
  std::vector<int> p(m, 0);
  int c = 0, r = 0;
  for (int i = 1; i < m - 1; i++) {
    int mirr = 2 * c - i;
    if (i < r) p[i] = std::min(r - i, p[mirr]);
    while (t[i + p[i] + 1] == t[i - p[i] - 1]) p[i]++;
    if (i + p[i] > r) { c = i; r = i + p[i]; }
  }
  int maxLen = 0, center = 0;
  for (int i = 1; i < m - 1; i++) {
    if (p[i] > maxLen) { maxLen = p[i]; center = i; }
  }
  int start = (center - maxLen) / 2;
  return {start, maxLen};
}
