#include "geometry_line.cpp"
#include <vector>

double polygonArea(const std::vector<Point> &p) {
  double s = 0;
  int n = p.size();
  for (int i = 0; i < n; i++)
    s += p[i].cross(p[(i + 1) % n]);
  return std::fabs(s) / 2;
}

bool inPolygon(const Point &pt, const std::vector<Point> &p) {
  int n = p.size(), cnt = 0;
  for (int i = 0; i < n; i++) {
    Point a = p[i], b = p[(i + 1) % n];
    if (onSegment(pt, Segment(a, b)))
      return true;
    if (sgn(a.y - b.y) == 0)
      continue;
    if (a.y > b.y)
      std::swap(a, b);
    if (sgn(pt.y - a.y) >= 0 && sgn(pt.y - b.y) < 0 && sgn(cross(a, b, pt)) < 0)
      cnt++;
  }
  return cnt & 1;
}

Point polygonCentroid(const std::vector<Point> &p) {
  double sx = 0, sy = 0, sa = 0;
  int n = p.size();
  for (int i = 0; i < n; i++) {
    double a = p[i].cross(p[(i + 1) % n]);
    sx += (p[i].x + p[(i + 1) % n].x) * a;
    sy += (p[i].y + p[(i + 1) % n].y) * a;
    sa += a;
  }
  return {sx / (3 * sa), sy / (3 * sa)};
}
