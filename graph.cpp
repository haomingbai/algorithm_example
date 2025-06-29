#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <list>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#pragma once

#include "./dsu.cpp"

#if __cplusplus >= 202302L
#define CONSTEXPR23 constexpr
#else
#define CONSTEXPR23 /* empty */
#endif

template <typename T>
class adjacent_matrix {
  const std::size_t node_num;
  std::vector<std::vector<T>> mat;
  T inf;

  template <typename Func, typename... Args>
  auto dfs(std::size_t curr, std::vector<unsigned char> &visited, Func func,
           Args... args) -> void {
    visited[curr] = true;
    func(curr, std::forward<Args>(args)...);
    for (std::size_t i = 0; i < node_num; i++) {
      if (!visited[i] && mat[curr][i] != inf) {
        dfs(i, visited, func, std::forward<Args>(args)...);
      }
    }
  }

 public:
  CONSTEXPR23 adjacent_matrix(const std::size_t node_number)
      : node_num(node_number), inf(10000000) {
    mat.resize(node_num);
    for (auto &it : this->mat) {
      it.resize(node_num);
      std::fill(it.begin(), it.end(), inf);
    }
    for (std::size_t i = 0; i < node_num; i++) {
      mat[i][i] = 0;
    }
  }
  CONSTEXPR23 adjacent_matrix(const std::size_t node_number, const T &inf)
      : node_num(node_number), inf(inf) {
    mat.resize(node_num);
    for (auto &it : this->mat) {
      it.resize(node_num);
      std::fill(it.begin(), it.end(), inf);
    }
    for (std::size_t i = 0; i < node_num; i++) {
      mat[i][i] = 0;
    }
  }
  CONSTEXPR23 adjacent_matrix(const std::size_t node_number, const T &&inf)
      : node_num(node_number), inf(inf) {
    mat.resize(node_num);
    for (auto &it : this->mat) {
      it.resize(node_num);
      std::fill(it.begin(), it.end(), inf);
    }
    for (std::size_t i = 0; i < node_num; i++) {
      mat[i][i] = 0;
    }
  }

  CONSTEXPR23 auto dijkstra(std::size_t start) -> std::vector<T> {
    std::vector<T> res(node_num, inf);
    res[start] = 0;
    auto visited = new bool[node_num]();

    while (true) {
      // Find the node unvisited with the shortest distance.
      std::size_t node_to_proc = SIZE_MAX;
      for (std::size_t i = 0; i < node_num; i++) {
        if (node_to_proc == SIZE_MAX && !visited[i]) {
          node_to_proc = i;
        } else if (!visited[i] && res[i] < res[node_to_proc]) {
          node_to_proc = i;
        }
      }
      if (node_to_proc == SIZE_MAX || res[node_to_proc] == inf) {
        break;
      }
      visited[node_to_proc] = 1;

      // Release the node
      for (std::size_t i = 0; i < node_num; i++) {
        if (!visited[i]) {
          T dist = res[node_to_proc] + mat[node_to_proc][i];
          if (mat[node_to_proc][i] == inf || dist < res[node_to_proc]) {
            continue;  // Digit overflows
                       // Might be deleted if you can make sure that the range
                       // of T is enough.
          }
          if (dist < res[i]) {
            res[i] = dist;
          }
        }
      }
    }

    delete[] visited;
    return res;
  }

  auto dijkstra(std::size_t start, std::vector<std::size_t> &prev)
      -> std::vector<T> {
    if (start >= node_num) {
      throw std::invalid_argument("The arg start is to large!");
    }

    std::vector<T> res(node_num, inf);
    res[start] = 0;
    auto visited = new bool[node_num]();
    prev.clear();
    prev.resize(node_num, SIZE_MAX);

    while (true) {
      // Find the node unvisited with the shortest distance.
      std::size_t node_to_proc = SIZE_MAX;
      for (std::size_t i = 0; i < node_num; i++) {
        if (node_to_proc == SIZE_MAX && !visited[i]) {
          node_to_proc = i;
        } else if (!visited[i] && res[i] < res[node_to_proc]) {
          node_to_proc = i;
        }
      }
      if (node_to_proc == SIZE_MAX || res[node_to_proc] == inf) {
        break;
      }
      visited[node_to_proc] = true;

      // Release the node
      for (std::size_t i = 0; i < node_num; i++) {
        if (!visited[i]) {
          T dist = res[node_to_proc] + mat[node_to_proc][i];
          if (mat[node_to_proc][i] == inf || dist < res[node_to_proc]) {
            continue;  // Digit overflows
                       // Might be deleted if you can make sure that the range
                       // of T is enough.
          }
          if (dist < res[i]) {
            res[i] = dist;
            prev[i] = node_to_proc;
          }
        }
      }
    }

    delete[] visited;
    return res;
  }

  CONSTEXPR23 auto add_edge(std::size_t from, std::size_t to, const T &weight)
      -> void {
    if (from >= node_num || to >= node_num) {
      throw std::invalid_argument(
          "Invalid argument: the starting or end point is not in the graph");
    }
    mat[from][to] = weight;
  }

  auto spfa(std::size_t from, std::vector<T> &res) -> bool {
    if (from >= node_num) {
      throw std::invalid_argument(
          "Invalid argument: the starting or end point is not in the graph");
    }

    res.clear();
    res.resize(node_num, inf);
    res[from] = 0;
    std::queue<size_t> q;
    q.push(from);
    std::vector<unsigned char> in_queue(node_num, false);
    std::vector<std::size_t> cnt(node_num, 0);
    in_queue[from] = true;

    while (!q.empty()) {
      std::size_t to_proc = q.front();
      q.pop();
      in_queue[to_proc] = false;

      for (std::size_t i = 0; i < node_num; i++) {
        if (mat[to_proc][i] != inf) {
          T dist = res[to_proc] + mat[to_proc][i];
          if (dist < res[i]) {
            res[i] = dist;
            cnt[i] = cnt[to_proc] + 1;
            if (cnt[i] >= node_num) {
              return false;
              // The function fail becalue a negative circular.
            }
            if (!in_queue[i]) {
              q.push(i), in_queue[i] = true;
            }
          }
        }
      }
    }
    return true;  // The function succeed.
  }

  auto spfa(std::size_t from, std::vector<T> &res, std::vector<size_t> &prev)
      -> bool {
    if (from >= node_num) {
      throw std::invalid_argument(
          "Invalid argument: the starting or end point is not in the graph");
    }

    prev.clear();
    prev.resize(node_num, SIZE_MAX);
    res.clear();
    res.resize(node_num, inf);
    res[from] = 0;
    std::queue<size_t> q;
    q.push(from);
    std::vector<unsigned char> in_queue(node_num, false);
    std::vector<std::size_t> cnt(node_num, 0);
    in_queue[from] = true;

    while (!q.empty()) {
      std::size_t to_proc = q.front();
      q.pop();
      in_queue[to_proc] = false;

      for (std::size_t i = 0; i < node_num; i++) {
        if (mat[to_proc][i] != inf) {
          T dist = res[to_proc] + mat[to_proc][i];
          if (dist < res[i]) {
            prev[i] = to_proc;
            res[i] = dist;
            cnt[i] = cnt[to_proc] + 1;
            if (cnt[i] >= node_num) {
              return false;
              // The function fail becalue a negative circular.
            }
            if (!in_queue[i]) {
              q.push(i), in_queue[i] = true;
            }
          }
        }
      }
    }
    return true;  // The function succeed.
  }

  template <typename Func, typename... Args>
  auto dfs(std::size_t start, Func func, Args... args) -> void {
    if (start >= node_num) {
      throw std::invalid_argument(
          "Invalid argument: the starting or end point is not in the graph");
    }
    std::vector<unsigned char> visited(node_num, false);
    dfs(start, visited, func, std::forward<Args>(args)...);
  }

  template <typename Func, typename... Args>
  auto bfs(std::size_t start, Func func, Args... args) -> void {
    if (start >= node_num) {
      throw std::invalid_argument(
          "Invalid argument: the starting or end point is not in the graph");
    }
    std::queue<size_t> q;
    std::vector<unsigned char> visited(node_num, false);
    q.push(start);
    visited[start] = true;
    while (!q.empty()) {
      auto to_proc = q.front();
      q.pop();
      func(to_proc, std::forward(args)...);
      for (std::size_t i = 0; i < node_num; i++) {
        if (!visited[i] && mat[to_proc][i] != inf) {
          visited[i] = true;
          q.push(i);
        }
      }
    }
  }

  CONSTEXPR23 auto floyd() -> std::vector<std::vector<T>> {
    auto res = mat;

    for (std::size_t k = 0; k < node_num; k++) {
      for (std::size_t from = 0; from < node_num; from++) {
        for (std::size_t to = 0; to < node_num; to++) {
          res[from][to] = std::min(res[from][to], res[from][k] + res[k][to]);
        }
      }
    }

    return res;
  }

  auto floyd(std::vector<std::vector<std::size_t>> &prevs)
      -> std::vector<std::vector<T>> {
    auto res = mat;
    prevs.clear();
    prevs.resize(node_num, std::vector<std::size_t>(node_num, SIZE_MAX));
    for (std::size_t from = 0; from < node_num; from++) {
      for (std::size_t to = 0; to < node_num; to++) {
        if (mat[from][to] != inf) {
          prevs[from][to] = from;
        }
      }
    }

    for (std::size_t k = 0; k < node_num; k++) {
      for (std::size_t from = 0; from < node_num; from++) {
        for (std::size_t to = 0; to < node_num; to++) {
          auto res_prev = res[from][to];
          res[from][to] = std::min(res[from][to], res[from][k] + res[k][to]);
          if (res[from][to] != res_prev) {
            prevs[from][to] = prevs[k][to];
          }
        }
      }
    }

    return res;
  }

  auto remove_edge(std::size_t from, std::size_t to) -> void {
    mat[from][to] = inf;
  }
};

template <typename T>
struct undirected_edge {
  std::size_t node1, node2;
  T weight;
  undirected_edge(std::size_t n1, std::size_t n2, T weight) {
    node1 = n1, node2 = n2;
    this->weight = weight;
  }
  bool operator==(const undirected_edge &e) const {
    if (((node1 == e.node1 && node2 == e.node2) ||
         (node2 == e.node1 && node1 == e.node2)) &&
        weight == e.weight) {
      return true;
    }
    return false;
  }
  bool operator<(const undirected_edge &e) const { return weight < e.weight; }
};

template <typename T>
class adjacent_list {
  struct to_edge {
    std::size_t destination;
    T distance;
  };
  std::size_t node_num;
  std::vector<std::list<to_edge>> edges;
  T inf;
  template <typename Func, typename... Args>
  auto dfs(std::size_t curr, std::vector<unsigned char> &visited, Func func,
           Args... args) -> void {
    visited[curr] = true;
    func(curr, std::forward<Args>(args)...);
    for (auto &it : edges[curr]) {
      if (!visited[it.destination]) {
        dfs(it.destination, visited, func, std::forward<Args>(args)...);
      }
    }
  }

 public:
  auto kruskal() -> std::vector<undirected_edge<T>> {
    std::vector<undirected_edge<T>> edge_set;
    dsu set_union(node_num);
    std::vector<undirected_edge<T>> res;

    for (auto idx = 0uz; idx < node_num; idx++) {
      for (auto &to_edge : edges[idx]) {
        if (idx < to_edge.destination) {  // 只加 (u,v) 而不加 (v,u)
          edge_set.emplace(idx, to_edge.destination, to_edge.distance);
        }
      }
    }
    std::stable_sort(edge_set);

    std::size_t cnt = 0;

    for (auto &edge : edge_set) {
      if (cnt >= node_num - 1) {
        break;
      }

      auto root1(set_union.findRoot(edge.node1));
      auto root2(set_union.findRoot(edge.node2));

      if (root1 == root2) {
        continue;
      }

      cnt++;
      res.emplace_back(edge);
      set_union.unite(edge.node1, edge.node2);
    }

    return res;
  }

  adjacent_list(std::size_t node_number)
      : node_num(node_number), edges(node_number), inf(10000000) {}

  adjacent_list(std::size_t node_number, const T &inf)
      : node_num(node_number), edges(node_number), inf(inf) {}

  auto dijkstra(std::size_t from) -> std::vector<T> {
    if (from >= node_num) {
      throw std::invalid_argument("The starting point is to large!");
    }

    std::vector<T> dests(node_num, inf);
    dests[from] = 0;
    std::vector<unsigned char> visited(node_num, false);
    struct node_with_dist {
      std::size_t pos;
      T dist;
      node_with_dist(size_t pos, const T &dist) : pos(pos), dist(dist) {}
      bool operator<(const node_with_dist &a) const { return dist > a.dist; }
    };
    std::priority_queue<node_with_dist> heap;
    heap.emplace(from, 0);

    for (std::size_t i = 0; i < node_num && !heap.empty(); i++) {
      auto to_proc = heap.top().pos;
      heap.pop();

      // Find the unvisited node with the shortest distance.
      bool flag = false;
      while (visited[to_proc] == true) {
        if (heap.empty()) {
          flag = true;
          break;
        }
        to_proc = heap.top().pos;
        heap.pop();
      }
      if (flag) {
        break;
      }
      if (dests[to_proc] == inf) {
        break;
      }

      for (auto &it : edges[to_proc]) {
        if (it.distance == inf) {
          continue;
        }
        auto dist = it.distance + dests[to_proc];
        if (dist >= inf) {
          continue;
        }
        if (dist < dests[it.destination]) {
          node_with_dist node(it.destination, dist);
          dests[it.destination] = dist;
          heap.emplace(std::move(node));
        }
      }
    }

    return dests;
  }

  void add_edge(std::size_t from, std::size_t to, const T &distance) {
    if (from >= node_num || to >= node_num) {
      throw std::invalid_argument("The arg from or to is too large");
    }
    to_edge e = {.destination = to, .distance = distance};
    edges[from].push_back(e);
  }

  auto spfa(std::size_t from, std::vector<T> &res) -> bool {
    if (from >= node_num) {
      throw std::invalid_argument("The starting point is to large!");
    }

    res.clear();
    res.resize(node_num, inf);
    res[from] = 0;
    std::queue<std::size_t> q;
    q.push(from);
    std::vector<unsigned char> in_queue(node_num, false);
    std::vector<std::size_t> cnt(node_num, 0);
    in_queue[from] = true;

    while (!q.empty()) {
      std::size_t to_proc = q.front();
      q.pop();
      in_queue[to_proc] = false;

      for (auto &it : edges[to_proc]) {
        if (res[to_proc] == inf || it.distance > inf - res[to_proc]) {
          continue;
        }
        auto dist = it.distance + res[to_proc];
        if (dist < res[it.destination]) {
          res[it.destination] = dist;
          cnt[it.destination] = cnt[to_proc] + 1;
          if (cnt[it.destination] >= node_num) {
            return false;
          }
          if (!in_queue[it.destination]) {
            q.push(it.destination);
            in_queue[it.destination] = true;
          }
        }
      }
    }
    return true;
  }

  auto dijkstra(std::size_t from, std::vector<size_t> &prev) -> std::vector<T> {
    if (from >= node_num) {
      throw std::invalid_argument("The starting point is to large!");
    }

    std::vector<T> dests(node_num, inf);
    dests[from] = 0;
    std::vector<unsigned char> visited(node_num, false);
    struct node_with_dist {
      std::size_t pos;
      T dist;
      node_with_dist(size_t pos, const T &dist) : pos(pos), dist(dist) {}
      bool operator<(const node_with_dist &a) const { return dist > a.dist; }
    };
    std::priority_queue<node_with_dist> heap;
    heap.emplace(from, 0);
    prev.clear();
    prev.resize(node_num, SIZE_MAX);

    for (std::size_t i = 0; i < node_num && !heap.empty(); i++) {
      auto to_proc = heap.top().pos;
      heap.pop();

      // Find the unvisited node with the shortest distance.
      bool flag = false;
      while (visited[to_proc] == true) {
        if (heap.empty()) {
          flag = true;
          break;
        }
        to_proc = heap.top().pos;
        heap.pop();
      }
      if (flag) {
        break;
      }
      if (dests[to_proc] == inf) {
        break;
      }

      for (auto &it : edges[to_proc]) {
        if (it.distance == inf) {
          continue;
        }
        auto dist = it.distance + dests[to_proc];
        if (dist >= inf) {
          continue;
        }
        if (dist < dests[it.destination]) {
          node_with_dist node(it.destination, dist);
          dests[it.destination] = dist;
          heap.emplace(std::move(node));
          prev[it.destination] = to_proc;
        }
      }
    }

    return dests;
  }

  auto spfa(std::size_t from, std::vector<T> &res,
            std::vector<std::size_t> &prev) -> bool {
    if (from >= node_num) {
      throw std::invalid_argument("The starting point is to large!");
    }

    res.clear();
    res.resize(node_num, inf);
    res[from] = 0;
    std::queue<std::size_t> q;
    q.push(from);
    std::vector<unsigned char> in_queue(node_num, false);
    std::vector<std::size_t> cnt(node_num, 0);
    in_queue[from] = true;
    prev.clear();
    prev.resize(node_num, SIZE_MAX);

    while (!q.empty()) {
      std::size_t to_proc = q.front();
      q.pop();
      in_queue[to_proc] = false;

      for (auto &it : edges[to_proc]) {
        if (res[to_proc] == inf || it.distance > inf - res[to_proc]) {
          continue;
        }
        auto dist = it.distance + res[to_proc];
        if (dist < res[it.destination]) {
          prev[it.destination] = to_proc;
          res[it.destination] = dist;
          cnt[it.destination] = cnt[to_proc] + 1;
          if (cnt[it.destination] >= node_num) {
            return false;
          }
          if (!in_queue[it.destination]) {
            q.push(it.destination);
            in_queue[it.destination] = true;
          }
        }
      }
    }
    return true;
  }

  template <typename Func, typename... Args>
  auto dfs(std::size_t start, Func func, Args... args) -> void {
    if (start >= node_num) {
      throw std::invalid_argument(
          "The index of the starting point is too large!");
    }
    std::vector<unsigned char> visited(node_num, 0);
    dfs(start, visited, func, std::forward<Args>(args)...);
  }

  template <typename Func, typename... Args>
  auto bfs(std::size_t start, Func func, Args... args) -> void {
    if (start >= node_num) {
      throw std::invalid_argument(
          "The index of the starting point is too large!");
    }

    std::queue<std::size_t> q;
    q.push(start);
    std::vector<unsigned char> visited(node_num, false);

    while (!q.empty()) {
      std::size_t curr = q.front();
      q.pop();
      func(curr, std::forward<T>(args)...);
      for (auto &it : edges[curr]) {
        if (!visited[it.destination]) {
          q.push(it.destination);
        }
      }
    }
  }

  auto remove_edge(std::size_t from, std::size_t to) {
    for (auto it = edges[from].begin(); it != edges[from].end();) {
      auto old_it = it;
      it = std::next(it);

      if (old_it->destination == to) {
        edges[from].erase(old_it);
      }
    }
  }
};
