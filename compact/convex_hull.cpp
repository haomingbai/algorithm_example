#include "geometry_point.cpp"
#include <algorithm>
#include <vector>

std::vector<Point> convexHull(std::vector<Point> pts) {
  int n = pts.size();
  if (n <= 2) return pts;
  std::sort(pts.begin(), pts.end());
  pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
  n = pts.size();
  if (n <= 2) return pts;
  std::vector<Point> stk(2 * n);
  int top = 0;
  for (int i = 0; i < n; i++) {
    while (top >= 2 && sgn(cross(stk[top - 2], stk[top - 1], pts[i])) <= 0) top--;
    stk[top++] = pts[i];
  }
  int tmp = top;
  for (int i = n - 2; i >= 0; i--) {
    while (top > tmp && sgn(cross(stk[top - 2], stk[top - 1], pts[i])) <= 0) top--;
    stk[top++] = pts[i];
  }
  top--;
  return {stk.begin(), stk.begin() + top};
}
