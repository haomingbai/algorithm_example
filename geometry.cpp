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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstdlib>
#include <numbers>
#include <stdexcept>
#include <tuple>
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
constexpr int sign(T x) {
  if (std::fabs(x) < epsilon_g<T>) {
    return 0;
  } else {
    return (x < 0) ? (-1) : (1);
  }
}

// 为了防止模板写错, 这里给整数也写一个.
template <typename T>
constexpr int sign(T x) {
  if (x < 0) {
    return -1;
  } else if (x > 0) {
    return 1;
  } else {
    return 0;
  }
}

// 工具函数, 浮点数比较大小.
template <typename T, typename U>
  requires(
      std::is_floating_point_v<decltype(std::declval<T>() - std::declval<U>())>)
constexpr int fcmp(T x, U y) {
  return sign(x - y);
}

template <typename T, typename U>
constexpr int fcmp(T x, U y) {
  return sign(x - y);
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

    // 3) std::declval<T>() * sin(x) / cos(x) 合法，并返回可转换回 T
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
  // 注意: 在C++20之前, ==和!=并不是一种运算, 需要手动重载,
  // 但是我们这里已经是C++20了, 所以就不需要了.
  // 我相信在GCC 15已经发布的当下, C++20会更加普及.
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

  // 向量的取负语义.
  Point2D<T> operator-() const
    requires Negativable<T>
  {
    return Point2D<T>(-x, -y);
  }

  // 两点之间的距离, 变成成员函数主要是怕和std::distance撞了
  // 毕竟竞赛选手的恶习之一就是using namespace std;
  // 这里作为成员函数, 语义同样明确, 但是避免和标准空间发生冲突.
  // 因为返回类型较为复杂, 所以使用了后置的类型声明, auto不进接口是美德.
  template <typename T2>
  auto distanceWith(const Vector2D<T2> &p2)
      -> decltype(std::sqrt(std::declval<T>() - std::declval<T2>())) const {
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
  auto rotateFor(const F rad) const
      -> Vector2D<decltype(std::declval<T>() * std::sin(std::declval<F>()))> {
    using ReturnType =
        Vector2D<decltype(std::declval<T>() * std::sin(std::declval<F>()))>;
    auto new_x = x * std::cos(rad) - y * std::sin(rad),
         new_y = x * std::sin(rad) + y * std::cos(rad);
    return ReturnType(new_x, new_y);
  }

  // 向量的模长.
  // 因为伟大的 ABDUL SAMAD KHAN 教授, 我对于范数的概念很不熟悉,
  // 为了避免贻笑大方之家, 我这里没有采用规范的norm(范数)的称谓,
  // 同时也是方便使用者调用.
  auto length() const -> decltype(std::sqrt(std::declval<T>()))
    requires(requires(T a, T b) { std::sqrt(a * a + b * b); })
  {
    return std::sqrt(x * x + y * y);
  }

  // 单位向量.
  auto unit() const
      -> Vector2D<decltype(std::declval<T>() / std::sqrt(std::declval<T>()))>
    requires(requires(T x, T y) { x / std::sqrt(x * x + y * y); })
  {
    using ReturnType =
        Vector2D<decltype(std::declval<T>() / std::sqrt(std::declval<T>()))>;
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
  auto unitNormal() const
      -> Point2D<decltype(std::declval<T>() / std::sqrt(std::declval<T>()))>
    requires(requires(T x, T y) { x / std::sqrt(x * x + y * y); })
  {
    // 链式调用, 产生一个临时对象, 但是对于基本数据类型, 开销尚可接受.
    return normal().unit();
  }

  // 模长平方, 这个接口很常用就留着.
  T length2()
    requires Multiplyable<T> && Addable<T>
  {
    return x * x + y * y;
  }
};

// 从此以后, Point2D类型就可以表示向量, 且在实际应用中应当严格遵循别名规则.
template <typename T>
using Vector2D = Point2D<T>;

// 向量加法实现, 要求数据类型满足可加性.
template <typename T1, typename T2>
  requires(AddableWith<T1, T2>)
auto operator+(const Vector2D<T1> &p1, const Vector2D<T2> &p2)
    -> Vector2D<decltype(std::declval<T1>() + std::declval<T2>())> {
  using RT = decltype(std::declval<T1>() + std::declval<T2>());
  Vector2D<RT> res(p1.x + p2.x, p1.y + p2.y);
  return res;
}

// 向量减法实现, 要求减法实现.
template <typename T1, typename T2>
  requires(SubtractableWith<T1, T2>)
auto operator-(const Vector2D<T1> &p1, const Vector2D<T2> &p2)
    -> Vector2D<decltype(std::declval<T1>() - std::declval<T2>())> {
  using RT = decltype(std::declval<T1>() - std::declval<T2>());
  Vector2D<RT> res(p1.x - p2.x, p1.y - p2.y);
  return res;
}

// 要求可以正确实现乘法.
// 这里乘法的语义给了点乘, 因为给叉乘的话维度就上去了, 不满足可乘的性质了.
template <typename T1, typename T2>
  requires(MultiplyableWith<T1, T2>)
auto operator*(const Vector2D<T1> &p1, const Vector2D<T2> &p2)
    -> decltype(std::declval<T1>() * std::declval<T2>()) {
  using RT = decltype(std::declval<T1>() * std::declval<T2>());
  RT res(p1.x * p2.x + p1.y * p2.y);
  return res;
}

// 要求可以实现累加, 这里的语义是将一个向量翻倍.
// 这里嘀咕一句, 这里的向量可以去用来实例化旁边的线段树了...
// 加法和减法都有了, 累加也有了...
template <typename T, typename U>
  requires MultiplyableWith<T, U>
auto operator*(const Vector2D<T> &p, U k)
    -> Vector2D<decltype(std::declval<T>() * std::declval<U>())> {
  using RT = Vector2D<decltype(std::declval<T>() * std::declval<U>())>;
  RT res(p.x * k, p.y * k);
  return res;
}

// 除法实现...
template <typename T, typename U>
  requires(DividableWith<T, U>)
auto operator/(const Vector2D<T> &p, U k)
    -> Vector2D<decltype(std::declval<T>() / std::declval<U>())> {
  Vector2D<T> res(p.x / k, p.y / k);
  return res;
}

// 辅助萃取乘法结果的别名模板
template <typename T, typename U>
using multiply_result_t =
    std::decay_t<decltype(std::declval<T>() * std::declval<U>())>;

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
    -> decltype((std::declval<T1>() * std::declval<T2>()) -
                (std::declval<T1>() * std::declval<T2>())) {
  return p1.x * p2.y - p1.y * p2.x;
}

// 这里是判断两个向量是否平行.
// 这里实现的是浮点数版本, 但是这个操作本身并不要求浮点,
// 反而浮点让整个过程更麻烦...
template <typename T1, typename T2>
  requires(requires(Vector2D<T1> v1, Vector2D<T2> v2) {
    crossProductValue(v1, v2);
  })
bool isParallel(const Vector2D<T1> &v1, const Vector2D<T2> &v2) {
  return sign(crossProductValue(v1, v2)) == 0;
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

  // 对于纯数据, 默认构造必须有, 否则继承的时候有你好果汁.
  LineBase2D() {};

  // 算个delta, 对一些数要求了乘法, 加法和减法, 一般的有符号整数和浮点肯定能过.
  // 这里就是点到直线距离里面的判别式, 就是那个 delta / sqrt(a ^ 2 + b ^ 2)
  template <typename U>
  // 约束直接描述函数体内的核心操作
    requires requires(const Point2D<T> &pt, const Point2D<U> &pu) {
      // 检查 crossProductValue 是否有效，因为 delta 内部的 c 就是
      // crossProductValue
      { crossProductValue(pt, pt) };
      // 检查最终的表达式 a * p.x + b * p.y + c 是否有效
      {
        (pt.y - pt.y) * pu.x + (pt.x - pt.x) * pu.y + crossProductValue(pt, pt)
      };
    }
  auto delta(const Point2D<U> &p) const
      -> decltype(std::declval<T>() * std::declval<U>()) {
    // 这里是获得直线的一般式的算法.
    // ref: https://blog.csdn.net/madbunny/article/details/43955883
    auto a = p2.y - p1.y, b = p1.x - p2.x;
    auto c = p2.x * p1.y - p1.x * p2.y;

    using RT = decltype(std::declval<T>() * std::declval<U>());
    RT res = a * p.x + b * p.y + c;
    return res;
  }

  // 获得对应直线的一般式, 这里的约束比较长,
  // 但是对于常见的数据类型应该都可以通过.
  auto generalForm() const
      -> std::tuple<decltype(std::declval<T>() - std::declval<T>()),
                    decltype(std::declval<T>() - std::declval<T>()),
                    decltype(std::declval<T>() * std::declval<T>() -
                             std::declval<T>() * std::declval<T>())>
    requires Subtractable<T> &&
             Subtractable<decltype(std::declval<T>() * std::declval<T>())>
  {
    // 返回的类型一定要严谨.
    using RT = std::tuple<decltype(std::declval<T>() - std::declval<T>()),
                          decltype(std::declval<T>() - std::declval<T>()),
                          decltype(std::declval<T>() * std::declval<T>() -
                                   std::declval<T>() * std::declval<T>())>;

    // 这里和上面的delta使用的是相同的算法, 参考相同的链接即可.
    auto a = p2.y - p1.y, b = p1.x - p2.x;
    auto c = p2.x * p1.y - p1.x * p2.y;

    RT res = std::make_tuple(a, b, c);

    return res;
  }

  template <typename U>
    requires std::is_floating_point_v<T> && std::is_floating_point_v<U>
  auto project(const Point2D<U> &p) const
      -> Point2D<decltype(std::declval<T>() * std::declval<U>())> {
    using DT = decltype(std::declval<T>() * std::declval<U>());

    // 这里是算出一个系数k, 这个系数k通过计算向量在目标向量上的投影长度,
    // 进而计算出从起点到投影在两点构成的线段上的占比.
    // 思路还是来源与清华大学的算法竞赛教程, 作者是罗永军教授等人.
    DT k = ((p2 - p1) * (p - p1)) / ((p2 - p1).length2());

    return p1 + (p2 - p1) * k;
  }
};

// 平行检测, 注意这里共线也是被检测出平行的.
template <typename T1, typename T2>
bool isParallel(const LineBase2D<T1> &l1, const LineBase2D<T2> &l2) {
  return isParallel(l1.p2 - l1.p1, l2.p2 - l2.p1);
}

// 类型声明, 方便后续函数声明.
template <typename T>
struct LineSegment2D;

// 二维直线, 因为直线是连续的, 所以难免涉及浮点数运算, 除非上高精度.
// 那高精度这里就不考虑了, 高精度和计算几何直接松耦合速度太慢狗都不用.
template <typename T>
  requires std::is_floating_point_v<T>
struct Line2D : public LineBase2D<T> {
  using LineBase2D<T>::p1;
  using LineBase2D<T>::p2;
  using LineBase2D<T>::delta;

  // 点方向式表示直线.
  template <typename PointDataType, typename FloatType>
    requires(std::is_convertible_v<PointDataType, T> &&
             std::is_floating_point_v<FloatType>)
  Line2D(const Point2D<PointDataType> &p, FloatType angle) {
    p1 = p;

    // 确保角度的参数范围在0到pi之内.
    // 因为我们这里是一条直线, 所以任意一个方向就可以了.
    while (sign(angle) == -1) {
      angle += std::numbers::pi_v<FloatType>;
    }
    while (fcmp(angle, std::numbers::pi_v<FloatType>) == 1) {
      angle -= std::numbers::pi_v<FloatType>;
    }

    // 如果我们这里的角度为 pi / 2, 那么方向向量就是 <0, 1>
    if (sign(angle - std::numbers::pi_v<FloatType> / 2) == 0) {
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
    [[unlikely]] if (sign(a) == 0 && sign(b) == 0) {
      throw std::invalid_argument(
          "In ax + by + c, a and b cannot both be zero.");
    } else if (sign(b) == 0) {
      // 这个就是k为无穷的情况, 需要单独处理.
      // k为无穷的时候, 整个直线是一条竖直线.
      auto y_0 = static_cast<T>(0), y_1 = static_cast<T>(1);
      auto x = static_cast<T>(-c / a);

      p1 = Point2D<T>(x, y_0), p2 = Point2D<T>(x, y_1);
    } else [[likely]] {
      // 简单的kx+b模式, 这里还是不要搞事情.
      // 直线的数据类型一定是浮点的, 所以才能这么除.
      auto x_0 = static_cast<T>(0), x_1 = static_cast<T>(1);
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
  bool operator==(const Line2D<U> &line) const {
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

  // 两个点如果连线, 是否可以穿过直线.
  template <typename T1, typename T2>
    requires MultiplyableWith<T, T1> && MultiplyableWith<T, T2> &&
             Multiplyable<T> &&
             MultiplyableWith<decltype(std::declval<T>() * std::declval<T1>()),
                              decltype(std::declval<T>() * std::declval<T2>())>
  bool cross(const Point2D<T1> &p1, const Point2D<T2> &p2) const {
    auto d1 = delta(p1), d2 = delta(p2);

    if (sign(d1 * d2) > 0) {
      return false;
    } else {
      return true;
    }
  }

  // 判断线段和直线是否相交.
  template <typename U>
  bool cross(const LineSegment2D<U> &l) const;

  // 方向向量
  Vector2D<T> direction() const { return p2 - p1; }

  // 单位方向向量.
  Vector2D<T> directionUnit() const { return direction().unit(); }

  // 判定两条直线是否相交.
  // 这里首先是判断平行, 如果方向向量平行, 那么判断是否共线.
  // 相交和共线这里cross都返回true.
  template <typename U>
  bool cross(const Line2D<U> &l) const {
    if (isParallel(this->direction(), l.direction())) {
      if (sign(this->delta(l.p1)) == 0) {
        return true;
      }
      return false;
    } else {
      return true;
    }
  }

  template <typename F>
    requires requires(T t) {
      { (std::abs(t)) };
    }
  auto distanceWith(const Point2D<F> &p) const
      -> decltype(std::sqrt(std::declval<T>())) {
    // 这里是获得直线的一般式的算法.
    // ref: https://blog.csdn.net/madbunny/article/details/43955883
    auto a = p2.y - p1.y, b = p1.x - p2.x;
    auto c = p2.x * p1.y - p1.x * p2.y;

    // 获得判别式
    auto delta = this->delta(p);

    // 进行距离计算.
    auto res = std::abs(delta) / std::sqrt(a * a + b * b);
    return res;
  }

  // 两条直线之间的距离
  // 思路就是最典型的|c1-c2|/sqrt(a^2+b^2)
  template <typename U>
    requires std::is_convertible_v<U, T>
  auto distanceWith(const LineBase2D<U> &l) const {
    assert(isParallel(*this, l));

    [[assume(isParallel(*this, l))]];

    auto [a1, b1, c1] = this->generalForm();
    auto [a2, b2, c2] = l.generalForm();

    return std::abs(c1 - c2) / std::sqrt(a1 * a1 + b1 * b1);
  }

  // 两条直线的交点.
  // 约束类型随便写一点, 稍微测试一下就好, 反正浮点的圈子就那么大.
  template <typename U>
    requires std::is_convertible_v<U, T>
  auto crossingPoint(const Line2D<U> &l) const
      -> Point2D<decltype(std::declval<T>() * std::declval<U>())> {
    using DataType = decltype(std::declval<T>() * std::declval<U>());

    // ref: 算法竞赛 清华大学出版社
    DataType s1 = crossProductValue(p2 - p1, l.p1 - p1),
             s2 = crossProductValue(p2 - p1, l.p2 - p1);

    return Point2D<DataType>(l.p1.x * s2 - l.p2.x * s1,
                             l.p1.y * s2 - l.p2.y * s1) /
           (s2 - s1);
  }

  // 0代表平行, 1代表重和, 2代表相交
  // 计算两条直线之间的关系.
  template <typename U>
  int relationWith(const Line2D<U> l) const {
    if (isParallel(this->direction(), l.direction())) {
      if (sign(l.delta(p1)) == 0) {
        // 重和
        return 1;
      } else {
        return 0;
      }
    } else {
      return 2;
    }
  }
};

template <typename T>
struct LineSegment2D : public LineBase2D<T> {
  using LineBase2D<T>::p1;
  using LineBase2D<T>::p2;

  // 拷贝构造.
  // 现在的情况已经考虑到线段和直线的转换了...
  template <typename U>
    requires std::is_convertible_v<U, T>
  LineSegment2D(const LineBase2D<U> &line) : LineBase2D<T>(line) {}

  // 同样从两点构造
  template <typename T1, typename T2>
    requires std::is_convertible_v<T1, T> && std::is_convertible_v<T2, T>
  LineSegment2D(const Point2D<T1> &p1, const Point2D<T2> &p2)
      : LineBase2D<T>(p1, p2) {
    assert(p1 != p2);
  }

  // 点方向和长度构造线段
  template <typename P, typename F1, typename F2>
    requires std::is_convertible_v<P, T> && std::is_floating_point_v<F1> &&
             std::is_floating_point_v<F2>
  LineSegment2D(const Point2D<P> &p, F1 rad, F2 len) {
    p1 = Point2D<T>(p);

    // 同样先调整参数, 只是这里单位是2pi.
    while (sign(rad) < 0) {
      rad += 2 * std::numbers::pi_v<F1>;
    }
    while (fcmp(rad, 2 * std::numbers::pi_v<F1>) > 0) {
      rad -= 2 * std::numbers::pi_v<F1>;
    }

    // 找到对应终点.
    F1 unit_dx = std::cos(rad), unit_dy = std::sin(rad);
    p2 = p1 + Point2D<T>(unit_dx * len, unit_dy * len);
  }

  // 这里判定相等, 只有完全重合是相等.
  template <typename U>
    requires std::is_convertible_v<U, T>
  bool operator==(const LineSegment2D<U> &line) const {
    return (p1 == line.p1 && p2 == line.p2) || (p1 == line.p2 && p2 == line.p1);
  }

  // 线段之间的相交判定, 这里只讲浮点数的版本, 普通版逻辑相同, 只有细小差别.
  template <typename U>
    requires requires(const LineSegment2D<T> &l, const LineSegment2D<U> &s) {
      // 检查 delta 调用
      { l.delta(s.p1) };
      { s.delta(l.p1) };
    } &&
             // 使用标准概念，表示 T 和 U 之间可以进行全序比较
             std::totally_ordered_with<T, U>
  bool cross(const LineSegment2D<U> &seg) const {
    // 使用delta进行计算, 如果delta异号, 那么说明两点在直线异侧, 可能相交.
    auto d_a1 = this->delta(seg.p1), d_a2 = this->delta(seg.p2);
    // 同侧就别想了
    if (sign(d_a1 * d_a2) > 0) {
      return false;
    }

    // l1 和 l2检测完检测l2 和 l1, 思路相同.
    auto d_b1 = seg.delta(p1), d_b2 = seg.delta(p2);
    if (sign(d_b1 * d_b2) > 0) {
      return false;
    }

    // 如果二者共线
    if (sign(d_a1) == 0 && sign(d_a2) == 0 && sign(d_b1) == 0 &&
        sign(d_b2) == 0) {
      // 计算投影范围
      auto [this_xmin, this_xmax] = std::minmax(p1.x, p2.x);
      auto [this_ymin, this_ymax] = std::minmax(p1.y, p2.y);
      auto [seg_xmin, seg_xmax] = std::minmax(seg.p1.x, seg.p2.x);
      auto [seg_ymin, seg_ymax] = std::minmax(seg.p1.y, seg.p2.y);

      // 双投影检查（必须同时满足）
      bool x_overlap =
          fcmp(this_xmax, seg_xmin) >= 0 && fcmp(seg_xmax, this_xmin) >= 0;
      bool y_overlap =
          fcmp(this_ymax, seg_ymin) >= 0 && fcmp(seg_ymax, this_ymin) >= 0;
      return x_overlap && y_overlap;
    }

    // 全部在对方两侧, 检测通过.
    return true;
  }

  // 这个接口是用来判断一个点是否在我们的目标线段上.
  template <typename U>
    requires requires(const Point2D<T> &pt1, const Point2D<U> &pu) {
      // 检查 crossProductValue(Point<T>, Point<U>) 是否有效
      { crossProductValue(pt1, pu) };
      // 检查 dotProduct(Point<T>, Point<U>) 是否有效 (这里用 * 代替)
      { pt1 * pu };
    }
  bool isOnLineSegment(const Point2D<U> &p) const {
    // ref: 算法竞赛 清华大学出版社
    return sign(crossProductValue(p - p1, p2 - p1)) == 0 &&
           sign((p - p1) * (p - p2)) <= 0;
    // 这个算法的逻辑比较简单, 两个子表达式分别表达如下含义:
    // vec(p, p1), vec(p, p2) 是否平行, 这个子表达式可以说明p是否在直线上.
    // vec(p, p1), vec(p, p2) 是否具备相反的方向, 如果二者具有的方向相同,
    // 就表明p在线段的某一侧而非线段上.
    // 这里充分考虑了p在两个端点上的corner case.
  }

  // 这个函数求得的是点到线段的距离.
  template <typename U>
  // 因为根式的存在, 这里参与的数据类型必须是浮点数.
    requires std::is_floating_point_v<decltype(std::declval<T>() *
                                               std::declval<U>())>
  auto distanceWith(const Point2D<U> &p) const
      -> decltype(std::declval<T>() * std::declval<U>()) {
    // 如果p1p和p1p2方向相反或者p2p和p2p1方向相反, 那么点在线段的竖向两侧,
    // 返回目标点到端点的最小距离.
    if (sign((p2 - p1) * (p - p1)) <= 0 || sign((p1 - p2) * (p - p2)) <= 0) {
      return std::min((p - p1).length(), (p - p2).length());
    } else {
      // 否则返回点到直线的距离.
      return Line2D<T>(*this).distanceWith(p);
    }
  }
};

template <typename T>
  requires std::is_floating_point_v<T>  // -- 外层类模板约束
template <typename U>                   // -- 成员函数模板参数
bool Line2D<T>::cross(const LineSegment2D<U> &l) const {
  return this->cross(l.p1, l.p2);
}

}  // namespace geometry
