/**
 * @file asr.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-08-20
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <cmath>
#include <concepts>
#include <cstdlib>

/**
 * @brief Simpson's Rule for numerical integration.
 *
 * Simpson's Rule is a method for approximating the definite integral of a
 * function by fitting a quadratic polynomial to the function at three points
 * (the endpoints and the midpoint) and integrating that polynomial over the
 * interval [a, b]. The formula for the approximation is: (b - a)/6 * [f(a) +
 * 4f((a+b)/2) + f(b)].
 *
 * This method provides exact results for polynomials up to degree 3 and
 * generally offers better accuracy than the trapezoidal rule for smooth
 * functions.
 *
 * @tparam Func Callable type. Must be invocable with a floating-point argument
 * and return a value convertible to a floating-point type.
 * @tparam F1   Floating-point type for the lower bound.
 * @tparam F2   Floating-point type for the upper bound.
 *
 * @param f The function to be integrated. Should be continuous on [a, b].
 * @param a Lower limit of integration.
 * @param b Upper limit of integration.
 *
 * @return Approximation of the integral ∫_{a}^{b} f(x) dx.
 *         The return type is deduced as the common type of F1 and F2.
 *
 * @note This is a non-adaptive implementation that provides a single
 * approximation over the entire interval. For higher accuracy over complex
 * functions, consider using an adaptive method like Adaptive Simpson's Rule
 * (ASR).
 * @note Requires that `f` be defined and continuous on [a, b].
 *
 * Example usage:
 * @code{.cpp}
 * auto result = Simpson([](double x) { return std::sin(x); }, 0.0, 3.14159);
 * @endcode
 */
template <typename Func, std::floating_point F1, std::floating_point F2>
auto Simpson(Func f, F1 a, F2 b)
    -> decltype(std::declval<F1>() + std::declval<F2>()) {
  using ReturnType = decltype(std::declval<F1>() + std::declval<F2>());

  ReturnType mid = a + (b - a) / 2.0;
  ReturnType result = f(a) + f(b) + f(mid) * 4.0;
  result *= (b - a);
  result /= 6.0;

  return result;
}

/**
 * @brief Adaptive Simpson's Rule (ASR) for numerical integration.
 *
 * Adaptive Simpson's Rule is a recursive method for approximating the definite
 * integral of a function. It dynamically refines the integration interval by
 * subdividing it into smaller segments where the function exhibits more
 * variation, and uses coarse segments where the function is smooth. This allows
 * for efficient computation by minimizing the number of function evaluations
 * while achieving the desired accuracy.
 *
 * The method compares the Simpson's rule estimates on the current segment and
 * its two halves. If the difference between the composite estimate and the
 * whole segment estimate is within the error tolerance, the result is accepted.
 * Otherwise, the algorithm recurses on both halves.
 *
 * @tparam Func Callable type. Must be invocable with a floating-point argument
 * and return a value convertible to a floating-point type.
 * @tparam F1   Floating-point type for the lower bound.
 * @tparam F2   Floating-point type for the upper bound.
 * @tparam Ferr Floating-point type for the error tolerance.
 *
 * @param f     The function to be integrated. Must be continuous on [a, b].
 * @param a     Lower limit of integration.
 * @param b     Upper limit of integration.
 * @param error Maximum allowable error tolerance for the integral
 * approximation.
 *
 * @return Approximation of the integral ∫_{a}^{b} f(x) dx.
 *         The return type is deduced as the common type of F1 and F2.
 *
 * @note The function uses recursion and may cause stack overflow for extremely
 * tight tolerances or highly oscillatory functions over large intervals.
 * @note Requires that `f` be defined and continuous on [a, b].
 *
 * Example usage:
 * @code{.cpp}
 * auto result = ASR([](double x) { return std::sin(x); }, 0.0, 3.14159, 1e-6);
 * @endcode
 */
template <typename Func, std::floating_point F1, std::floating_point F2,
          std::floating_point Ferr>
auto ASR(Func f, F1 a, F2 b, Ferr error)
    -> decltype(std::declval<F1>() + std::declval<F2>()) {
  using ReturnType = decltype(std::declval<F1>() + std::declval<F2>());
  ReturnType mid = a + (b - a) / 2.0;

  // 先对整体使用辛普森法
  auto S0 = Simpson(f, a, b);
  // 再将区间分成两块求解
  auto S1 = Simpson(f, a, mid);
  auto S2 = Simpson(f, mid, b);

  // 求解此时的误差
  Ferr curr_error = std::abs(S0 - (S1 + S2));

  // 误差如果超过最大允许范围,
  // (因为在返回过程中, 差值会被除以15,
  // 所以curr_error是15倍的当前误差)
  // 那么就递归求解两侧的面积,
  // 同时要求两侧的误差不超过最大误差的一半.
  // 否则就直接返回结果.
  if (curr_error >= 15.0 * error) {
    ReturnType result =
        ASR(f, a, mid, error / 2.0) + ASR(f, mid, b, error / 2.0);
    return result;
  } else {
    ReturnType result = (S1 + S2) + (S1 + S2 - S0) / 15.0;
    return result;
  }
}
