#include <cmath>
#include <algorithm>

const double EPS = 1e-8;
const double PI = std::acos(-1);

int sgn(double x) { return std::fabs(x) < EPS ? 0 : (x < 0 ? -1 : 1); }

struct Point {
  double x, y;
  Point() : x(0), y(0) {}
  Point(double x, double y) : x(x), y(y) {}
  Point operator+(const Point &b) const { return {x + b.x, y + b.y}; }
  Point operator-(const Point &b) const { return {x - b.x, y - b.y}; }
  Point operator*(double k) const { return {x * k, y * k}; }
  Point operator/(double k) const { return {x / k, y / k}; }
  bool operator<(const Point &b) const { return x != b.x ? x < b.x : y < b.y; }
  bool operator==(const Point &b) const { return sgn(x - b.x) == 0 && sgn(y - b.y) == 0; }
  double dot(const Point &b) const { return x * b.x + y * b.y; }
  double cross(const Point &b) const { return x * b.y - y * b.x; }
  double len() const { return std::hypot(x, y); }
  double len2() const { return x * x + y * y; }
  double dist(const Point &b) const { return std::hypot(x - b.x, y - b.y); }
  Point unit() const { double l = len(); return {x / l, y / l}; }
  Point normal() const { return {-y, x}; }
  Point rotate(double rad) const { return {x * std::cos(rad) - y * std::sin(rad), x * std::sin(rad) + y * std::cos(rad)}; }
  double angle() const { return std::atan2(y, x); }
};

double cross(const Point &a, const Point &b, const Point &c) {
  return (b - a).cross(c - a);
}

double dot(const Point &a, const Point &b, const Point &c) {
  return (b - a).dot(c - a);
}

double angle(const Point &a, const Point &b) {
  return std::acos(a.dot(b) / a.len() / b.len());
}
