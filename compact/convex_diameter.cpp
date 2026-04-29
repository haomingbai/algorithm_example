#include "geometry_point.cpp"
#include <algorithm>
#include <vector>

double convexDiameter(const std::vector<Point> &ch) {
  int n = ch.size();
  if (n < 2) return 0;
  if (n == 2) return ch[0].dist(ch[1]);
  int j = 1;
  double best = 0;
  for (int i = 0; i < n; i++) {
    int ni = (i + 1) % n;
    while (true) {
      int nj = (j + 1) % n;
      double c1 = std::fabs((ch[ni] - ch[i]).cross(ch[j] - ch[i]));
      double c2 = std::fabs((ch[ni] - ch[i]).cross(ch[nj] - ch[i]));
      if (c2 > c1) j = nj;
      else break;
    }
    best = std::max(best, ch[i].dist(ch[j]));
    best = std::max(best, ch[ni].dist(ch[j]));
  }
  return best;
}

double convexDiameterSquare(const std::vector<Point> &ch) {
  int n = ch.size();
  if (n < 2) return 0;
  if (n == 2) return (ch[0] - ch[1]).len2();
  int j = 1;
  double best = 0;
  for (int i = 0; i < n; i++) {
    int ni = (i + 1) % n;
    while (true) {
      int nj = (j + 1) % n;
      double c1 = std::fabs((ch[ni] - ch[i]).cross(ch[j] - ch[i]));
      double c2 = std::fabs((ch[ni] - ch[i]).cross(ch[nj] - ch[i]));
      if (c2 > c1) j = nj;
      else break;
    }
    best = std::max(best, (ch[i] - ch[j]).len2());
    best = std::max(best, (ch[ni] - ch[j]).len2());
  }
  return best;
}
