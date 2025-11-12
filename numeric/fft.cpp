/**
 * @file fft.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-11-12
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <cmath>
#include <complex>
#include <concepts>
#include <vector>

// 辅助函数: 辅助完成位倒序置换.
// 这一步是为了将数据重新排列, 以便后续的蝶形运算可以原地进行.
// 使用 inline, 以便和接口分开.
template <std::floating_point F>
inline void ReverseBits(std::vector<std::complex<F>>& data) {
  // 获取样本点的数量.
  const int number_of_samples = data.size();
  for (int i = 1, j = 0; i < number_of_samples; ++i) {
    int bit = number_of_samples >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      std::swap(data[i], data[j]);
    }
  }
}

/**
 * @brief 快速傅里叶变换 (Fast Fourier Transform).
 *
 * 该函数采用 Cooley-Tukey 算法的迭代实现, 对输入信号进行原地变换.
 *
 * @tparam T 浮点类型, 如 float 或 double. 该约束通过 C++23 的 concept 实现.
 * @param data 输入的复数向量, 其大小必须是 2 的整数次幂.
 *             函数将直接修改此向量, 变换结果保存在原地.
 */
template <std::floating_point T>
void FastFourierTransform(std::vector<std::complex<T>>& data) {
  // 获取样本点的数量.
  const int number_of_samples = data.size();
  if (number_of_samples <= 1) {
    return;
  }

  // 步骤 1: 对输入数据进行位倒序置换.
  ReverseBits(data);

  // 步骤 2: 执行蝶形运算.
  // sub_problem_size 代表当前阶段子问题的规模, 从 2 开始, 每次翻倍.
  for (int sub_problem_size = 2; sub_problem_size <= number_of_samples;
       sub_problem_size <<= 1) {
    // 计算当前阶段的基本旋转因子 (twiddle factor).
    const T angle = -2.0 * M_PI / sub_problem_size;
    const std::complex<T> base_twiddle_factor(cos(angle), sin(angle));

    // 遍历所有子问题.
    for (int i = 0; i < number_of_samples; i += sub_problem_size) {
      // 当前旋转因子, 初始为 1.
      std::complex<T> twiddle_factor(1);
      // 对每个子问题中的元素进行蝶形运算.
      for (int j = 0; j < sub_problem_size / 2; ++j) {
        // u 是蝶形运算的第一个输入.
        std::complex<T> u = data[i + j];
        // v 是蝶形运算的第二个输入, 需要乘以旋转因子.
        std::complex<T> v = data[i + j + sub_problem_size / 2] * twiddle_factor;

        // 完成蝶形运算并存回原位.
        data[i + j] = u + v;
        data[i + j + sub_problem_size / 2] = u - v;

        // 更新旋转因子以供下一次迭代使用.
        twiddle_factor *= base_twiddle_factor;
      }
    }
  }
}

/**
 * @brief 快速傅里叶反变换 (Inverse Fast Fourier Transform).
 *
 * IFFT 的实现巧妙地利用了 FFT. 其核心思想是: IFFT(x) = (1/N) *
 * conj(FFT(conj(x))).
 *
 * @tparam T 浮点类型, 如 float 或 double.
 * @param data 输入的复数向量, 其大小必须是 2 的整数次幂.
 *             函数将直接修改此向量, 反变换结果保存在原地.
 */
template <std::floating_point T>
void InverseFastFourierTransform(std::vector<std::complex<T>>& data) {
  // 获取样本点的数量.
  const int number_of_samples = data.size();
  if (number_of_samples <= 1) {
    return;
  }

  // 步骤 1: 对输入信号的每个元素取共轭.
  for (auto& value : data) {
    value = std::conj(value);
  }

  // 步骤 2: 对共轭后的信号执行正向 FFT.
  FastFourierTransform(data);

  // 步骤 3: 对 FFT 结果再次取共轭, 并除以样本点数量 N.
  for (auto& value : data) {
    value = std::conj(value) / static_cast<T>(number_of_samples);
  }
}
