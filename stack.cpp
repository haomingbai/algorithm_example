/*************************************************************************
    > File Name: algorithm_example_code/stack.cpp
    > Author: Haoming Bai
    > Mail: haomingbai@hotmail.com
    > Created Time: Sat Sep 14 17:32:46 2024
 ************************************************************************/

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <utility>

template <typename T>
struct Node {
  Node<T>* next;
  T dat;
};

template <typename T>
class Stack {
  Node<T>* __top;
  size_t __len;

 public:
  Stack() : __top(nullptr), __len(0) {}
  void push(T&& dat) {
    __len++;
    auto __node = new Node<T>;
    __node->dat = std::move(dat);
    __node->next = std::move(this->__top);
    this->__top = __node;
  }
  void push(const T& dat) {
    __len++;
    auto __node = new Node<T>;
    __node->dat = dat;
    __node->next = std::move(this->__top);
    this->__top = __node;
  }
  const T& top() { return this->__top->dat; }
  T pop() {
    if (this->__len == 0) {
      return T{};
    }
    __len--;
    auto __tmp = __top->next;
    auto __res = std::move(__top->dat);
    delete __top;
    __top = __tmp;
    return std::move(__res);
  }
  size_t size() { return this->__len; }
  bool empty() { return __len ? false : true; }
  ~Stack() {
    while (!this->empty()) {
      this->pop();
    }
  }
};

int main() {
  Stack<int> test;
  for (auto i = 0; i < 10; i++) {
    test.push(i);
  }
  while (!test.empty()) {
    std::cout << test.pop() << std::endl;
  }
}
