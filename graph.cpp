#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

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
          if (res[node_to_proc] == inf || mat[node_to_proc][i] == inf ||
              dist < res[node_to_proc]) {
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

  CONSTEXPR23 auto add_edge(std::size_t from, std::size_t to, const T &weight)
      -> void {
    if (from >= node_num || to >= node_num) {
      throw std::invalid_argument(
          "Invalid argument: the starting or end point is not in the graph");
    }
    mat[from][to] = weight;
  }

  // To be realized
  auto spfa(std::size_t from, std::vector<T> &res) -> bool {}

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
      for (size_t i = 0; i < node_num; i++) {
        if (!visited[i] && mat[to_proc][i] != inf) {
          visited[i] = true;
          q.push(i);
        }
      }
    }
  }
};

int test_adj_dijkstra() {
  // 构造一个 6 个节点的有向图
  //
  // 图示（节点 0 为起点）：
  //    0
  //   / \
    //  1   2
  //   \ / \
    //    3   4
  //    |
  //    5
  //
  // 边及权重：
  // 0->1 (7), 0->2 (9), 0->5 (14)
  // 1->2 (10), 1->3 (15)
  // 2->3 (11), 2->5 (2)
  // 3->4 (6)
  // 4->5 (9)

  adjacent_matrix<int> g(6 /* node 数 */, /*inf=*/1000000);

  g.add_edge(0, 1, 7);
  g.add_edge(0, 2, 9);
  g.add_edge(0, 5, 14);
  g.add_edge(1, 2, 10);
  g.add_edge(1, 3, 15);
  g.add_edge(2, 3, 11);
  g.add_edge(2, 5, 2);
  g.add_edge(3, 4, 6);
  g.add_edge(4, 5, 9);

  // 从节点 0 出发跑 Dijkstra
  auto dist = g.dijkstra(0);

  // 预期最短路径距离（手算或参考经典例子）：
  // dist[0] = 0
  // dist[1] = 7
  // dist[2] = 9
  // dist[3] = 20   (0->2->3，9+11)
  // dist[4] = 26   (0->2->3->4，9+11+6)
  // dist[5] = 11   (0->2->5，9+2)
  std::vector<int> expected = {0, 7, 9, 20, 26, 11};

  // 检查结果
  for (std::size_t i = 0; i < dist.size(); ++i) {
    std::cout << "dist[" << i << "] = " << dist[i] << "\n";
    assert(dist[i] == expected[i] && "Dijkstra 结果与预期不符");
  }

  std::cout << "All tests passed!\n";
  return 0;
}

int test_adj_dfs() {
  // INF 定义
  const int INF = std::numeric_limits<int>::max();

  // 1. 构造一个 5 节点的无向图
  adjacent_matrix<int> g(5, INF);

  // 2. 添加边：0-1, 0-2, 1-3, 1-4
  g.add_edge(0, 1, 1);
  g.add_edge(0, 2, 1);
  g.add_edge(1, 3, 1);
  g.add_edge(1, 4, 1);

  // 3. 准备一个容器记录访问顺序
  std::vector<int> order;

  // 4. 定义回调：访问一个节点就 push_back 到 order
  auto record = [](std::size_t node, std::vector<int> &out) {
    out.push_back(static_cast<int>(node));
  };

  // 5. 调用 dfs，从节点 0 开始
  g.dfs(0, record, std::ref(order));

  // 6. 打印结果
  std::cout << "DFS 访问顺序：";
  for (int v : order) {
    std::cout << v << " ";
  }
  std::cout << std::endl;

  // （可选）断言顺序是否符合预期，比如 0 1 3 4 2
  std::vector<int> expected = {0, 1, 3, 4, 2};
  if (order == expected) {
    std::cout << "测试通过！" << std::endl;
  } else {
    std::cout << "测试失败，实际顺序：";
    for (int v : order) std::cout << v << " ";
    std::cout << std::endl;
  }

  return 0;
}

int test_adj_bfs() {
  // 1) 构造 5 个节点的图，默认 inf = 10000000
  adjacent_matrix<int> g(5);

  // 2) 增加有向边
  //    图的结构：
  //       0
  //     ↙  ↘
  //    1    2
  //    |    ↘
  //    3     4
  g.add_edge(0, 1, 1);
  g.add_edge(0, 2, 1);
  g.add_edge(1, 3, 1);
  g.add_edge(2, 4, 1);

  // 3) 用 BFS 记录访问顺序
  std::vector<int> order;
  g.bfs(
      0,
      // 回调：把每次访问到的节点编号 push 进 order
      [&order](std::size_t node) { order.push_back(static_cast<int>(node)); });

  // 4) 预期的 BFS 层次遍历顺序：0 -> 1,2 -> 3,4
  std::vector<int> expected = {0, 1, 2, 3, 4};

  // 打印并断言
  std::cout << "BFS order:";
  for (int v : order) {
    std::cout << ' ' << v;
  }
  std::cout << std::endl;

  assert(order == expected && "BFS 遍历顺序不符合预期！");

  std::cout << "BFS 测试通过！\n";
  return 0;
}

int main() {
  test_adj_dijkstra();
  test_adj_dfs();
  test_adj_bfs();
}
