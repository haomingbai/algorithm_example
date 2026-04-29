#include "geometry_point.cpp"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

double nearestPointPair(std::vector<Point> &pts) {
  int n = pts.size();
  if (n < 2) return std::numeric_limits<double>::max();
  std::sort(pts.begin(), pts.end());
  auto rec = [&](auto self, int l, int r) -> double {
    if (r - l < 2) return std::numeric_limits<double>::max();
    if (r - l == 2) return pts[l].dist(pts[l + 1]);
    int m = l + (r - l) / 2;
    double mx = pts[m].x;
    double d = std::min(self(self, l, m), self(self, m, r));
    std::vector<Point*> strip;
    for (int i = l; i < r; i++)
      if ((pts[i].x - mx) * (pts[i].x - mx) < d) strip.push_back(&pts[i]);
    std::sort(strip.begin(), strip.end(), [](Point *a, Point *b) { return a->y < b->y; });
    for (int i = 0; i < (int)strip.size(); i++)
      for (int j = i + 1; j < (int)strip.size(); j++) {
        double dy = strip[j]->y - strip[i]->y;
        if (dy * dy >= d) break;
        d = std::min(d, strip[i]->dist(*strip[j]));
      }
    return d;
  };
  return rec(rec, 0, n);
}
