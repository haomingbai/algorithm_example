#include <cstddef>
#include <deque>
#include <functional>
#include <vector>

template <typename T, typename Compare = std::less<T>>
struct MonoQueue {
  std::deque<std::pair<T, int>> q;
  Compare cmp;

  void push(const T &val, int idx) {
    while (!q.empty() && !cmp(q.back().first, val)) q.pop_back();
    q.push_back({val, idx});
  }

  void pop(int idx) {
    if (!q.empty() && q.front().second == idx) q.pop_front();
  }

  T front() const { return q.front().first; }
  bool empty() const { return q.empty(); }
  size_t size() const { return q.size(); }
};

template <typename T>
std::vector<T> slidingWindowMax(const std::vector<T> &a, int k) {
  MonoQueue<T, std::greater<T>> mq;
  std::vector<T> res;
  for (int i = 0; i < (int)a.size(); i++) {
    mq.push(a[i], i);
    if (i >= k - 1) {
      res.push_back(mq.front());
      mq.pop(i - k + 1);
    }
  }
  return res;
}

template <typename T>
std::vector<T> slidingWindowMin(const std::vector<T> &a, int k) {
  MonoQueue<T, std::less<T>> mq;
  std::vector<T> res;
  for (int i = 0; i < (int)a.size(); i++) {
    mq.push(a[i], i);
    if (i >= k - 1) {
      res.push_back(mq.front());
      mq.pop(i - k + 1);
    }
  }
  return res;
}
