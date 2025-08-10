/**
 * @file graph.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-07-22
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <list>
#include <queue>
#include <utility>
#include <vector>

#include "./concepts.cpp"
#include "./dsu.cpp"

namespace graph {

// 在许多图论的算法当中, 我们常常涉及比较操作和松弛操作.
// 因此, 对于边权, 我们应当确保他们可以使用比较和相加.
template <FullyComparable EdgeType, typename DataType = char>
  requires Addable<EdgeType> && Subtractable<EdgeType>
class DirectedAdjList {
 public:
  // 这个边是包含终点和边权的边.
  // 因为在图论当中, 边的终点称为Head,
  // 所以这种边我称为HeadEdge.
  struct HeadEdge {
    size_t head_;
    DataType weight_;

    template <typename... Args>
    HeadEdge(size_t head, Args &&...args)
        : head_(head), weight_(std::forward<Args>(args)...) {}
  };

  template <typename Func, typename... Args>
  void dfs(size_t curr, Func func, std::vector<uint8_t> &visited,
           Args &&...args) {
    func(curr, nodes_[curr], std::forward<Args>(args)...);
    visited[curr] = true;

    for (auto &it : edges_[curr]) {
      if (!visited[it.head_]) {
        dfs(it.head_, func, visited, std::forward<Args>(args)...);
      }
    }
  }

 private:
  std::vector<std::list<HeadEdge>> edges_;
  std::vector<DataType> nodes_;
  EdgeType inf_;
  size_t size_;

 public:
  // 添加一条边
  // 这里考虑到构造函数的多样性,
  // 所以我们在简单函数上, 全程使用可变参数模板.
  // 使用完美转发机制保证全程无多余的对象构造.
  template <typename... Args>
  void addEdge(size_t from, size_t to, Args &&...weight) {
    edges_[from].emplace_back(to, std::forward<Args>(weight)...);
  }

  // 删除一条边
  // 返回值代表是否成功删除.
  // 如果成功返回true
  // 否则返回false
  void removeEdge(size_t from, size_t to) {
    for (auto it = edges_[from].begin(); it != edges_[from].end(); it++) {
      if (it->head_ == to) {
        edges_[from].erase(it);
        break;
      }
    }
    return;
  }

  // 删除很多条边.
  // 这里可以传入你最多要删除的边的数量.
  void removeEdge(size_t from, size_t to, size_t max_cnt) {
    if (max_cnt == 0) {
      return;
    }

    size_t cnt = 0;
    for (auto it = edges_[from].begin(); it != edges_[from].end();) {
      if (it->head_ == to) {
        assert(cnt <= max_cnt);
        [[assume(cnt <= max_cnt)]];
        if (cnt >= max_cnt) {
          break;
        }
        auto old_it = it;
        it++;
        edges_[from].erase(old_it);
        cnt++;
        continue;
      }
      it++;
    }

    return;
  }

  // 构造函数
  // 必须确定图的点数和边数
  template <typename... Args>
  DirectedAdjList(size_t size, Args &&...inf)
      : edges_(size, std::list<HeadEdge>()),
        nodes_(size),
        inf_(std::forward<Args>(inf)...),
        size_(size) {}

  // 深度优先搜索, 从start节点开始,
  // 函数Func需要接受如下参数:
  // 参数1: 节点下标, 类型为size_t
  // 参数2: 节点的数据, 这个视情况而定.
  // 参数3: Args, 表达剩余参数.
  template <typename Func, typename... Args>
  void dfs(size_t start, Func func, Args &&...args) {
    std::vector<uint8_t> visited(size_, 0);

    dfs(start, func, visited, std::forward<Args>(args)...);
  }

  // 广度优先搜索.
  template <typename Func, typename... Args>
  void bfs(size_t start, Func func, Args &&...args) {
    std::queue<size_t> waiting_list;
    std::vector<uint8_t> visited(size_, 0);
    waiting_list.push(start);

    while (!waiting_list.empty()) {
      auto curr = waiting_list.front();
      waiting_list.pop();
      if (!visited[curr]) {
        func(curr, nodes_[curr], std::forward<Args>(args)...);
        visited[curr] = true;

        for (auto &it : edges_[curr]) {
          if (!visited[it.head_]) {
            waiting_list.push(it.head_);
          }
        }
      }
    }
  }

  // 迪杰斯特拉算法,
  // 梦开始的地方...
  std::vector<size_t> dijkstra(size_t start, std::vector<EdgeType> &dist) {
    // 创建一个节点, 这里可以用来存储我们待加入的节点以及路程.
    struct NodeWithDistance {
      size_t idx;
      DataType distance;
      NodeWithDistance() = default;
      NodeWithDistance(size_t idx, const DataType &dat)
          : idx(idx), distance(dat) {}
      NodeWithDistance(size_t idx, DataType &&dat)
          : idx(idx), distance(std::move(dat)) {}
      bool operator<(const NodeWithDistance &n) const {
        return this->distance > n.distance;
      }

      bool operator<(NodeWithDistance &&n) const {
        return this->distance > n.distance;
      }
    };

    // prev数组代表的是每个点的上一个节点,
    // 在返回是, prev数组存储的是最优路径上,
    // 每个节点的上一个节点.
    // 如果存储了自己, 那么要么是走不动了,
    // 要么是到起点了.
    // 需要的话, 自己用个栈倒一下就行.
    std::vector<size_t> prev(size_, 0);
    std::vector<uint8_t> visited(size_, false);

    // 所谓堆优化, 就是保持一个优先队列.
    // 每次知道了从起点到一个点的距离, 就加入一条路径.
    // 不用是最优的路径, 只要每次找到一个节点的更优路径就好了.
    std::priority_queue<NodeWithDistance> pqueue;
    // 插入第一个点, 因为自己到自己的距离是已知的.
    pqueue.emplace(start, 0);

    // 更新从起点到各点的距离.
    dist.resize(size_);
    // 一般来讲, 起点到所有节点的距离是无穷的.
    for (size_t i = 0; i < size_; i++) {
      prev[i] = i;
      dist[i] = inf_;
    }
    // 自己到自己的距离为0.
    dist[start] = 0;

    // 计数机制, 这里主要是等不及那个pqueue跑空.
    // 因为最多更新N次, 所以是cnt >= N就直接退出.
    size_t cnt = 0;
    while (!pqueue.empty()) {
      // 如果更新了N个点, 那就退出.
      if (cnt >= size_) {
        break;
      }

      // 从优先队列中弹出一个节点.
      auto curr_node = pqueue.top();
      pqueue.pop();

      // 如果这个点的权重没有被确定,
      // 那么我们就找到了当前未被确定的点中最优的那个.
      // 和传统的Dijkstra一样.
      // 这里我们的就是每次找到当前没有被确定距离的点中,
      // 距离起点最短的节点.
      if (!visited[curr_node.idx]) {
        if (curr_node.distance == inf_) {
          break;
        }
        visited[curr_node.idx] = true;
        cnt++;

        auto curr_node_idx = curr_node.idx;
        for (const auto &it : edges_[curr_node_idx]) {
          // 松弛操作.
          if (it.weight_ < dist[it.head_] - dist[curr_node_idx]) {
            // 如果发现了一条更短的边.
            // 将这个节点连带边权加入pqueue.
            dist[it.head_] = it.weight_ + dist[curr_node_idx];
            prev[it.head_] = curr_node_idx;
            pqueue.emplace(it.head_, dist[it.head_]);
          }
        }
      }
    }
    return prev;
  }

  bool spfa(size_t start, std::vector<EdgeType> &dist,
            std::vector<size_t> &prev) {
    prev.resize(size_);
    for (size_t i = 0; i < size_; i++) {
      prev[i] = i;
    }
    dist.resize(size_, inf_);
    dist[start] = 0;
    std::queue<size_t> queue;
    std::vector<uint8_t> in_queue(size_, 0);
    queue.push(start);
    in_queue[start] = true;
    std::vector<size_t> visit_cnt(size_, 0);

    while (!queue.empty()) {
      auto curr = queue.front();
      queue.pop();
      in_queue[curr] = false;

      for (auto &it : edges_[curr]) {
        if ((it.weight_ < dist[it.head_]) &&
            (dist[curr] < dist[it.head_] - it.weight_)) {
          visit_cnt[it.head_] = visit_cnt[curr] + 1;
          dist[it.head_] = it.weight_ + dist[curr];
          prev[it.head_] = curr;

          if (visit_cnt[it.head_] >= size_) {
            return false;
          }

          if (!in_queue[it.head_]) {
            queue.push(it.head_);
            in_queue[it.head_] = true;
          }
        }
      }
    }
    return true;
  }

  DataType &operator[](size_t idx) { return nodes_[idx]; }
};

template <typename EdgeType, class DataType = char>
struct DirectedAdjMatrix {
  std::vector<std::vector<EdgeType>> edges_;
  std::vector<DataType> nodes_;
  EdgeType inf_;
  size_t size_;

  auto floyd(std::vector<std::vector<EdgeType>> &dist)
      -> std::vector<std::vector<size_t>> {
    dist.resize(size_, std::vector<EdgeType>(size_, inf_));
    std::vector<std::vector<size_t>> prev(size_, std::vector<size_t>(size_));
    for (size_t i = 0; i < size_; i++) {
      for (size_t j = 0; j < size_; j++) {
        dist[i][j] = edges_[i][j];
        if (edges_[i][j] != inf_) {
          prev[i][j] = i;
        } else {
          prev[i][j] = j;
        }
      }
    }

    for (size_t k = 0; k < size_; k++) {
      for (size_t i = 0; i < size_; i++) {
        for (size_t j = 0; j < size_; j++) {
          if (dist[i][j] > dist[i][k] && dist[k][j] < dist[i][j] - dist[i][k]) {
            prev[i][j] = prev[k][j];
            dist[i][j] = dist[i][k] + dist[k][j];
          }
        }
      }
    }

    return prev;
  }

  DirectedAdjMatrix(size_t size, const EdgeType &inf)
      : size_(size), inf_(inf) {}

  DirectedAdjMatrix(size_t size, EdgeType &&inf)
      : size_(size), inf_(std::move(inf)) {}

  void addEdge(size_t from, size_t to, const DataType &weight) {
    edges_[from][to] = weight;
  }

  void removeEdge(size_t from, size_t to) { edges_[from][to] = inf_; }

  DataType &operator[](size_t idx) { return nodes_[idx]; }
};

template <typename T>
struct Edge {
  size_t p1, p2;
  T weight;
};

template <typename T>
std::vector<Edge<T>> Kruskal(std::vector<Edge<T>> edges,
                             std::size_t max_node_idx) {
  std::sort(edges.begin(), edges.end(),
            [](const auto &a, const auto &b) { return a.weight < b.weight; });
  DSU visited(max_node_idx + 1);

  std::vector<Edge<T>> res;
  for (auto &it : edges) {
    if (!visited.inSameSet(it.p1, it.p2)) {
      res.emplace_back(it);
      visited.unite(it.p1, it.p2);
    }
  }

  return res;
}

}  // namespace graph
