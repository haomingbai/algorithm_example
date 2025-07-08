/**
 * @file geometry.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-07-07
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "concepts.cpp"

namespace geometry {

// 主模板（处理非浮点类型，默认禁用）
template <typename T, typename = void>
constexpr T epsilon_g = T(0);  // 非浮点类型可设为0或根据需求调整

// 浮点类型的特化版本（使用SFINAE约束）
template <typename T>
constexpr T epsilon_g<T, std::enable_if_t<std::is_floating_point_v<T>>> =
    std::is_same_v<T, float> ? T(1e-5f) :  // float特化值
        std::is_same_v<T, double> ? T(1e-8)
                                  :  // double特化值
        std::is_same_v<T, long double> ? T(1e-12)
                                       :  // long double特化值
        T(1e-5);                          // 其他浮点类型默认值

// 判断浮点数的符号, 如果符号为0, 说明无限接近0, 可以当成0看待.
template <typename T>
  requires(std::is_floating_point_v<T>)
constexpr int sgn(T x) {
  if (std::fabs(x) < epsilon_g<T>) {
    return 0;
  } else {
    return (x < 0) ? (-1) : (1);
  }
}

// 工具函数, 浮点数比较大小.
template <typename T, typename U>
  requires(std::is_floating_point_v<decltype(T{} - U{})>)
constexpr int fcmp(T x, U y) {
  return sgn(x - y);
}

// 创建类型概念, 其实这里一个is_floating_point_v就可以解决的,
// 但是为了语义明确, 那么这里就创造一个概念, 叫可旋转,
// 这个概念用来表示数据类型可以用来进行旋转运算.
template <typename T, typename F>
concept Rotateable =
    // 1) F 必须是浮点类型
    std::floating_point<F> &&

    // 2) 对 F 调用 sin/cos 合法，并返回可转换回 F
    requires(F x) {
      { std::sin(x) } -> std::convertible_to<F>;
      { std::cos(x) } -> std::convertible_to<F>;
    } &&

    // 3) T{} * sin(x) / cos(x) 合法，并返回可转换回 T
    requires(T t, F x) {
      { t * std::sin(x) } -> std::convertible_to<T>;
      { t * std::cos(x) } -> std::convertible_to<T>;
    };

// 点和向量类, 这里的一个方法是把点和向量当成一种类型
template <typename T>
struct Point2D {
  // 将向量和点定义成一种类型.
  template <typename U>
  using Vector2D = Point2D<U>;

  // 数据, 二维平面内的点包含两个坐标, x和y
  T x, y;
  Point2D() {};

  // 使用x和y坐标构造新点.
  template <typename ValT1, typename ValT2>
    requires(std::is_convertible_v<ValT1, T> && std::is_convertible_v<ValT2, T>)
  Point2D(const ValT1 &x, const ValT2 &y) : x(x), y(y) {}

  // 使用另外一个点构造新点.
  template <typename T2>
    requires(std::is_convertible_v<T2, T>)
  Point2D(const Point2D<T2> &another) : x(another.x), y(another.y) {}

  // 支持一下移动构造, 这里作为五方法的一部分, 可以学习一下.
  template <typename T2>
    requires(std::is_convertible_v<T2, T>)
  Point2D(Point2D<T2> &&another) : x(another.x), y(another.y) {}

  // 相等判定, 这里因为涉及类型的变换, 所以加法和减法等都在外部定义
  template <typename U>
  bool operator==(const Point2D<U> &p2) const
    requires std::is_floating_point_v<T> && std::is_floating_point_v<U> &&
             std::is_convertible_v<U, T>
  {
    return (fcmp(x, p2.x) == 0) && (fcmp(y, p2.y) == 0);
  }

  // 这里是条件较弱的相等判定, 在模板的实例化中处在较低的有限级.
  // 可以作为模板和模板特化的学习用例.
  template <typename U>
  bool operator==(const Point2D<U> &p2) const {
    return (x == p2.x) && (y == p2.y);
  }

  // 两点之间的距离, 变成成员函数主要是怕和std::distance撞了
  // 毕竟竞赛选手的恶习之一就是using namespace std;
  // 这里作为成员函数, 语义同样明确, 但是避免和标准空间发生冲突.
  // 因为返回类型较为复杂, 所以使用了后置的类型声明, auto不进接口是美德.
  template <typename T2>
  auto distanceWith(const Vector2D<T2> &p2)
      -> decltype(std::sqrt(T{} - T2{})) const {
    auto xdiff = x - p2.x, ydiff = y - p2.y;
    return std::sqrt(xdiff * xdiff + ydiff * ydiff);
  }

  // 向量的旋转语义.
  // 在这里使用的公式和复数乘法代表旋转的语义类似.
  // 例如对于 x + iy, 在它的基础上乘以 (cos(theta) + isin(theta))
  // 就可以代表角度为theta的逆时针旋转.
  // 而相应的结果就是 xcos(theta) - ysin(theta) + i(xsin(theta) + ycos(theta)).
  // 因此, 当一个向量逆时针旋转 t 度时, 产生的新向量为:
  // new_x = xcos(t) + ysin(t)
  // new_y = xsin(t) + ycos(t).
  template <typename F>
    requires Rotateable<T, F>
  auto rotateFor(const F rad) const -> Vector2D<decltype(T{} * std::sin(F{}))> {
    using ReturnType = Vector2D<decltype(T{} * std::sin(F{}))>;
    auto new_x = x * std::cos(rad) - y * std::sin(rad),
         new_y = x * std::sin(rad) + y * std::cos(rad);
    return ReturnType(new_x, new_y);
  }

  // 向量的模长.
  // 因为伟大的 ABDUL SAMAD KHAN 教授, 我对于范数的概念很不熟悉,
  // 为了避免贻笑大方之家, 我这里没有采用规范的norm(范数)的称谓,
  // 同时也是方便使用者调用.
  auto length() const -> decltype(std::sqrt(T{}))
    requires(requires(T a, T b) { std::sqrt(a * a + b * b); })
  {
    return std::sqrt(x * x + y * y);
  }

  // 单位向量.
  auto unit() const -> Point2D<decltype(T{} / std::sqrt(T{}))>
    requires(requires(T x, T y) { x / std::sqrt(x * x + y * y); })
  {
    using ReturnType = Point2D<decltype(T{} / std::sqrt(T{}))>;
    auto new_x = x / length();
    auto new_y = y / length();
    return ReturnType(new_x, new_y);
  }

  // 法向量, 这个调用不会破坏类型, 只要求负号存在.
  Vector2D<T> normal() const
    requires requires(T t) {
      { -t } -> std::convertible_to<T>;
    }
  {
    return Vector2D<T>(-y, x);
  }

  // 单位法向量, 这里就要求比较高了... .
  auto unitNormal() const -> Point2D<decltype(T{} / std::sqrt(T{}))>
    requires(requires(T x, T y) { x / std::sqrt(x * x + y * y); })
  {
    // 链式调用, 产生一个临时对象, 但是对于基本数据类型, 开销尚可接受.
    return normal().unit();
  }
};

// 从此以后, Point2D类型就可以表示向量, 且在实际应用中应当严格遵循别名规则.
template <typename T>
using Vector2D = Point2D<T>;

// 向量加法实现, 要求数据类型满足可加性.
template <typename T1, typename T2>
  requires(AddableWith<T1, T2>)
auto operator+(const Vector2D<T1> &p1, const Vector2D<T2> &p2)
    -> Vector2D<decltype(T1{} + T2{})> {
  using RT = decltype(T1{} + T2{});
  Vector2D<RT> res(p1.x + p2.x, p1.y + p2.y);
  return res;
}

// 向量减法实现, 要求减法实现.
template <typename T1, typename T2>
  requires(SubtractableWith<T1, T2>)
auto operator-(const Vector2D<T1> &p1, const Vector2D<T2> &p2)
    -> Vector2D<decltype(T1{} - T2{})> {
  using RT = decltype(T1{} - T2{});
  Vector2D<RT> res(p1.x - p2.x, p1.y - p2.y);
  return res;
}

// 要求可以正确实现乘法.
// 这里乘法的语义给了点乘, 因为给叉乘的话维度就上去了, 不满足可乘的性质了.
template <typename T1, typename T2>
  requires(MultiplyableWith<T1, T2>)
auto operator*(const Vector2D<T1> &p1, const Vector2D<T2> &p2)
    -> Vector2D<decltype(T1{} * T2{})> {
  using RT = decltype(T1{} * T2{});
  Vector2D<RT> res(p1.x * p2.x, p1.y * p2.y);
  return res;
}

// 要求可以实现累加, 这里的语义是将一个向量翻倍.
// 这里嘀咕一句, 这里的向量可以去用来实例化旁边的线段树了...
// 加法和减法都有了, 累加也有了...
template <typename T>
  requires(Accumulateable<T>)
Vector2D<T> operator*(const Vector2D<T> &p, size_t k) {
  Vector2D<T> res(p.x * k, p.y * k);
  return res;
}

// 除法实现...
template <typename T>
  requires(Partable<T>)
Vector2D<T> operator/(const Vector2D<T> &p, size_t k) {
  Vector2D<T> res(p.x / k, p.y / k);
  return res;
}

// 辅助萃取乘法结果的别名模板
template <typename T, typename U>
using multiply_result_t = std::common_type_t<
    std::decay_t<decltype(std::declval<T>() * std::declval<U>())>>;

// 复合概念：T*U 合法，且 (T*U) 的类型可减
template <typename T, typename U>
concept MultiplyThenSubtractable =
    MultiplyableWith<T, U> && Subtractable<multiply_result_t<T, U>>;

// 这里要求实现乘法, 且同时乘法的产物类型还要可以相减.
// 这里实现的是叉乘的数值版本.
// 因为在x-y平面上的向量的叉乘一定垂直于x-y平面, 因此只需要返回结果的纵坐标即可.
// 原理如下:
// 计算向量叉乘, 可以使用行列式,
//         |i  j  k |
// a x b = |x1 y1 0 | = k * (x1 * y2 - x2 * y1)
//         |x2 y2 0 |
// 因此, z坐标就可以方便地被计算.
template <typename T1, typename T2>
  requires(MultiplyThenSubtractable<T1, T2>)
auto crossProductValue(const Vector2D<T1> &p1, const Vector2D<T2> &p2)
    -> decltype((T1{} * T2{}) - (T1{} * T2{})) {
  return p1.x * p2.y - p1.y * p2.x;
}

// 这里是判断两个向量是否平行.
// 这里实现的是浮点数版本, 但是这个操作本身并不要求浮点,
// 反而浮点让整个过程更麻烦...
template <typename T1, typename T2>
  requires(requires(Vector2D<T1> v1, Vector2D<T2> v2) {
            crossProductValue(v1, v2);
          }) && std::is_floating_point_v<decltype(T1{} - T2{})>
bool isParallel(const Vector2D<T1> &v1, const Vector2D<T2> &v2) {
  return sgn(crossProductValue(v1, v2)) == 0;
}

// 这里是整形版本, 这个版本显然直接判断就可以了.
template <typename T1, typename T2>
  requires(requires(Vector2D<T1> v1, Vector2D<T2> v2) {
    crossProductValue(v1, v2);
  })
bool isParallel(const Vector2D<T1> &v1, const Vector2D<T2> &v2) {
  return (crossProductValue(v1, v2)) == 0;
}

// 二维线基础, 这里将线和线段进行分离.
template <typename T>
struct LineBase2D {
  Point2D<T> p1, p2;  // 两点式.

  // 使用两点构造.
  template <typename T1, typename T2>
    requires std::is_convertible_v<T1, T> && std::is_convertible_v<T2, T>
  LineBase2D(const Point2D<T1> &p1, const Point2D<T2> &p2) : p1(p1), p2(p2) {}

  // 使用另一条线构造.
  template <typename U>
    requires std::is_convertible_v<U, T>
  LineBase2D(const LineBase2D<U> &line) : p1(line.p1), p2(line.p2) {}
};

// 二维直线, 因为直线是连续的, 所以难免涉及浮点数运算, 除非上高精度.
// 那高精度这里就不考虑了, 高精度和计算几何直接松耦合速度太慢狗都不用.
template <typename T>
  requires std::is_floating_point_v<T>
struct Line2D : public LineBase2D<T> {
  using LineBase2D<T>::p1;
  using LineBase2D<T>::p2;

  // 点斜式表示直线.
  template <typename PointDataType, typename FloatType>
    requires(std::is_convertible_v<PointDataType, T> &&
             std::is_floating_point_v<FloatType>)
  Line2D(const Point2D<PointDataType> &p, FloatType angle) {
    p1 = p;

    // 确保角度的参数范围在0到pi之内.
    // 因为我们这里是一条直线, 所以任意一个方向就可以了.
    while (sgn(angle) == -1) {
      angle += std::numbers::pi_v<FloatType>;
    }
    while (fcmp(angle, std::numbers::pi_v<FloatType>) == 1) {
      angle -= std::numbers::pi_v<FloatType>;
    }

    // 如果我们这里的角度为 pi / 2, 那么方向向量就是 <0, 1>
    if (sgn(angle - std::numbers::pi_v<FloatType> / 2) == 0) {
      // 点加上方向向量, 虽然点和向量类型相同, 但是类型别名要做好.
      p2 = p1 + Vector2D<T>(0, 1);
    } else {
      // 否则的话:
      // dy/dx = tan(theta), 那么 delta(x) = 1 -> delta(y) = tan(theta)
      FloatType p2_x = 1, p2_y = std::tan(angle);
      p2 = p1 + Vector2D<T>(p2_x, p2_y);
    }
  }

  // 两点式
  // 这里为了参数类型正常就不在发布模式验证有效性了.
  // 在调试模式加一个assert得了.
  template <typename T1, typename T2>
    requires std::is_convertible_v<T1, T> && std::is_convertible_v<T2, T>
  Line2D(const Point2D<T1> &p1, const Point2D<T2> &p2) : LineBase2D<T>(p1, p2) {
    assert(p1 != p2);
  }

  // 斜截式, 使用F(0), F(1)
  template <typename F1, typename F2>
    requires std::is_convertible_v<F1, T> && std::is_convertible_v<F2, T>
  Line2D(F1 k, F2 b) {
    auto y_0 = b, y_1 = k + b;
    p1 = Point2D<T>(0, y_0), p2 = Point2D<T>(1, y_1);
  }

  // 一般式, 这里采用了`likely`关键字进行分支预测.
  // 对k为无穷的情况进行了单独的讨论.
  template <typename F1, typename F2, typename F3>
    requires std::is_convertible_v<F1, T> && std::is_convertible_v<F2, T> &&
             std::is_convertible_v<F3, T>
  Line2D(F1 a, F2 b, F3 c) {
    [[unlikely]] if (sgn(a) == 0 && sgn(b) == 0) {
      throw std::invalid_argument(
          "In ax + by + c, a and b cannot both be zero.");
    } else if (sgn(a) == 0) {
      auto y = (T)(-c) / (T)(b);
    } else [[likely]] {
      auto x_0 = (T)(0), x_1 = (T)(1);
      auto y_0 = -(a * x_0 + c) / b, y_1 = -(a * x_1 + c) / b;

      p1 = Point2D<T>(x_0, y_0), p2 = Point2D<T>(x_1, y_1);
    }
  }

  // 拷贝构造, 其实我觉得还可以过分一点, 就是把所有有两个点的都纳入进来.
  // 现在的情况已经考虑到线段和直线的转换了...
  template <typename U>
    requires std::is_convertible_v<U, T>
  Line2D(const LineBase2D<U> &line) : LineBase2D<T>(line) {}

  // 判断两直线相等, 这里要求类型能够判定相等.
  template <typename U>
    requires requires(T t, U u) {
      { t == u } -> std::convertible_to<bool>;
    }
  bool operator==(Line2D<U> line) {
    auto v1 = p2 - p1;
    auto v2 = line.p2 - line.p1;

    // 先判断是否平行, 不平行就一定是不相等.
    if (isParallel(v2, v1)) {
      Vector2D<T> flagVector = line.p1 - p1;
      // 平行的情况下, 如果二者有公共点, 那么就是重和的, 那就意味着相等.
      if (flagVector == Vector2D<T>(0, 0) || isParallel(v1, flagVector)) {
        return true;
      }
      // 否则就是不相等.
      else {
        return false;
      }
    } else {
      return false;
    }
  }
};

}  // namespace geometry
