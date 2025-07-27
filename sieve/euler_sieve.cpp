/**
 * @file euler_sieve.cpp
 * @brief A basic implementation of Euler seive
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-07-27
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <cstddef>
#include <vector>

struct EulerSieve {
  std::vector<size_t> primes_;
  std::vector<unsigned char> is_prime_;

  EulerSieve(size_t max_val) : is_prime_(std::vector<unsigned char>(max_val + 1, true)) {
    is_prime_[0] = false;
    is_prime_[1] = false;
    primes_.reserve(max_val / 10);

    for (size_t i = 2; i <= max_val; i++) {
      if (is_prime_[i]) {
        primes_.push_back(i);
      }

      for (auto it = primes_.begin();
           it != primes_.end() && (*it) <= max_val / i; it++) {
        is_prime_[i * (*it)] = false;

        if (i % *it == 0) {
          break;
        }
      }
    }
  }

  size_t operator[](size_t idx) { return primes_[idx]; }

  bool is_prime(size_t val) { return is_prime_[val]; }
};
