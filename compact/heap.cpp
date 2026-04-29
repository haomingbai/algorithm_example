#include <cstddef>
#include <functional>
#include <vector>

template <typename T, typename Compare = std::less<T>>
struct Heap {
  std::vector<T> data;
  Compare cmp;

  Heap() : data(1) {}

  void push(const T &val) {
    data.push_back(val);
    int i = data.size() - 1;
    while (i > 1 && cmp(data[i / 2], data[i])) {
      std::swap(data[i], data[i / 2]);
      i /= 2;
    }
  }

  T pop() {
    T res = data[1];
    data[1] = data.back();
    data.pop_back();
    int i = 1, n = data.size() - 1;
    while (i * 2 <= n) {
      int j = i * 2;
      if (j + 1 <= n && cmp(data[j], data[j + 1])) j++;
      if (!cmp(data[i], data[j])) break;
      std::swap(data[i], data[j]);
      i = j;
    }
    return res;
  }

  T top() const { return data[1]; }
  bool empty() const { return data.size() <= 1; }
  size_t size() const { return data.size() - 1; }
};
