#include "geometry_point.cpp"

struct Line {
  Point p1, p2;
  Line() {}
  Line(Point p1, Point p2) : p1(p1), p2(p2) {}
  double a() const { return p2.y - p1.y; }
  double b() const { return p1.x - p2.x; }
  double c() const { return p2.x * p1.y - p1.x * p2.y; }
  double delta(const Point &p) const { return a() * p.x + b() * p.y + c(); }
};

struct Segment {
  Point p1, p2;
  Segment() {}
  Segment(Point p1, Point p2) : p1(p1), p2(p2) {}
  double len() const { return p1.dist(p2); }
};

bool parallel(const Line &l1, const Line &l2) {
  return sgn((l1.p2 - l1.p1).cross(l2.p2 - l2.p1)) == 0;
}

Point lineIntersect(const Line &l1, const Line &l2) {
  double s1 = (l1.p2 - l1.p1).cross(l2.p1 - l1.p1);
  double s2 = (l1.p2 - l1.p1).cross(l2.p2 - l1.p1);
  return {(l2.p1.x * s2 - l2.p2.x * s1) / (s2 - s1), (l2.p1.y * s2 - l2.p2.y * s1) / (s2 - s1)};
}

double pointToLine(const Point &p, const Line &l) {
  return std::fabs(l.delta(p)) / std::hypot(l.a(), l.b());
}

double pointToSegment(const Point &p, const Segment &s) {
  if (sgn((s.p2 - s.p1).dot(p - s.p1)) <= 0 || sgn((s.p1 - s.p2).dot(p - s.p2)) <= 0)
    return std::min(p.dist(s.p1), p.dist(s.p2));
  return pointToLine(p, Line(s.p1, s.p2));
}

bool onSegment(const Point &p, const Segment &s) {
  return sgn((p - s.p1).cross(s.p2 - s.p1)) == 0 && sgn((p - s.p1).dot(p - s.p2)) <= 0;
}

bool segmentsIntersect(const Segment &a, const Segment &b) {
  double d1 = cross(b.p1, b.p2, a.p1), d2 = cross(b.p1, b.p2, a.p2);
  double d3 = cross(a.p1, a.p2, b.p1), d4 = cross(a.p1, a.p2, b.p2);
  if (sgn(d1) * sgn(d2) < 0 && sgn(d3) * sgn(d4) < 0) return true;
  if (sgn(d1) == 0 && onSegment(a.p1, b)) return true;
  if (sgn(d2) == 0 && onSegment(a.p2, b)) return true;
  if (sgn(d3) == 0 && onSegment(b.p1, a)) return true;
  if (sgn(d4) == 0 && onSegment(b.p2, a)) return true;
  return false;
}

Point project(const Point &p, const Line &l) {
  double k = (l.p2 - l.p1).dot(p - l.p1) / (l.p2 - l.p1).len2();
  return l.p1 + (l.p2 - l.p1) * k;
}
