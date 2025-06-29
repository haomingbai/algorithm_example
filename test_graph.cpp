#include <functional>
#include <iostream>
#include <limits>

#include "graph.cpp"

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

int test_adj_spfa() {
  {
    // —— 测试 1：无负环的有向图 ——
    // 节点 0→1→2→3，且 0→3 还有一条直达边
    // 结构：
    //   0 -5-> 1 -2-> 2 -1-> 3
    //   0 -9-> 3
    adjacent_matrix<int> g(4, /*inf=*/1000000);
    g.add_edge(0, 1, 5);
    g.add_edge(1, 2, 2);
    g.add_edge(2, 3, 1);
    g.add_edge(0, 3, 9);

    std::vector<int> dist;
    bool ok = g.spfa(0, dist);
    assert(ok && "SPFA on acyclic graph should succeed");

    // 预期最短距离：0→{0,1,2,3} = {0,5,7,8}
    std::vector<int> exp = {0, 5, 7, 8};
    std::cout << "No-cycle distances:";
    for (auto d : dist) std::cout << " " << d;
    std::cout << std::endl;

    assert(dist == exp && "Distances mismatch in no-cycle test");
  }

  {
    // —— 测试 2：带负环的图 ——
    // 0→1 (1), 1→2 (-2), 2→1 (1) 构成净权重 -1 的负环
    adjacent_matrix<int> g(3, /*inf=*/1000000);
    g.add_edge(0, 1, 1);
    g.add_edge(1, 2, -2);
    g.add_edge(2, 1, 1);

    std::vector<int> dist;
    bool ok = g.spfa(0, dist);
    assert(!ok && "SPFA should detect negative cycle");
    std::cout << "Negative cycle correctly detected." << std::endl;
  }

  std::cout << "All SPFA tests passed!\n";
  return 0;
}

int test_adj_dijkstra_persist() {
  constexpr std::size_t N = 5;
  // 构造一个 5 节点图，无边时默认 inf
  adjacent_matrix<int> g(N, /*inf=*/1000000);

  // 添加一些有向边：0->1 (10), 0->2 (3), 2->1 (1), 2->3 (2), 1->3 (2), 3->4 (1)
  g.add_edge(0, 1, 10);
  g.add_edge(0, 2, 3);
  g.add_edge(2, 1, 1);
  g.add_edge(2, 3, 2);
  g.add_edge(1, 3, 2);
  g.add_edge(3, 4, 1);

  // 运行 dijkstra 并获取 dist 与 prev
  std::vector<std::size_t> prev;
  auto dist = g.dijkstra(0, prev);

  // 1) 验证 prev 大小和值域
  assert(prev.size() == N);
  for (std::size_t v = 0; v < N; ++v) {
    // prev 值要么是 SIZE_MAX（起点或不可达），要么 < N
    assert(prev[v] == SIZE_MAX || prev[v] < N);
  }

  // 2) 起点 prev[0] 必须为 SIZE_MAX
  assert(prev[0] == SIZE_MAX);

  // 3) 验证各节点的最短前驱
  // 手工计算最短路径树：
  // 0: dist=0, prev=SIZE_MAX
  // 1: 最短路径 0->2->1, 所以前驱 prev[1] 应为 2
  // 2: 最短路径 0->2,       prev[2] == 0
  // 3: 最短路径 0->2->3,    prev[3] == 2
  // 4: 最短路径 0->2->3->4, prev[4] == 3
  std::vector<std::size_t> expected_prev = {
      SIZE_MAX,  // 0
      2,         // 1
      0,         // 2
      2,         // 3
      3          // 4
  };

  for (std::size_t v = 0; v < N; ++v) {
    std::cout << "prev[" << v << "] = ";
    if (prev[v] == SIZE_MAX)
      std::cout << "SIZE_MAX";
    else
      std::cout << prev[v];
    std::cout << std::endl;

    assert(prev[v] == expected_prev[v] && "前驱数组值与预期不符");
  }

  std::cout << "Dijkstra with prev 测试通过！" << std::endl;
  return 0;
}

int test_adj_spfa_persist() {
  constexpr std::size_t N = 5;
  const int INF = 1000000;

  // —— 测试 1：无负环场景 ——
  {
    // 构造 5 节点图
    adjacent_matrix<int> g(N, INF);
    // 添加边：0→1(4), 0→2(2), 2→1(1), 1→3(5), 2→3(8), 3→4(3)
    g.add_edge(0, 1, 4);
    g.add_edge(0, 2, 2);
    g.add_edge(2, 1, 1);
    g.add_edge(1, 3, 5);
    g.add_edge(2, 3, 8);
    g.add_edge(3, 4, 3);

    std::vector<int> dist;
    std::vector<std::size_t> prev;
    bool ok = g.spfa(0, dist, prev);
    assert(
        ok &&
        "SPFA 在无负环图上应返回 true");  // SPFA 成功返回 true
                                          // :contentReference[oaicite:0]{index=0}

    // 预期最短距离：{0, 3, 2, 8, 11}
    std::vector<int> exp_dist = {0, 3, 2, 8, 11};
    // 预期前驱： prev[0]=SIZE_MAX, prev[1]=2, prev[2]=0, prev[3]=1, prev[4]=3
    std::vector<std::size_t> exp_prev = {SIZE_MAX,  // 源点无前驱
                                         2, 0, 1, 3};

    for (std::size_t i = 0; i < N; ++i) {
      assert(dist[i] == exp_dist[i] && "距离计算错误");
      assert(prev[i] == exp_prev[i] &&
             "前驱记录错误");  // 前驱更新逻辑 O(1)
                               // :contentReference[oaicite:1]{index=1}
    }

    std::cout << "Test 1 (no negative cycle) passed.\n";
  }

  // —— 测试 2：带负环场景 ——
  {
    // 构造含 3 节点的图：0→1(1), 1→2(-2), 2→1(1) 形成净权 -1 的负环
    adjacent_matrix<int> g2(3, INF);
    g2.add_edge(0, 1, 1);
    g2.add_edge(1, 2, -2);
    g2.add_edge(2, 1, 1);

    std::vector<int> dist2;
    std::vector<std::size_t> prev2;
    bool ok2 = g2.spfa(0, dist2, prev2);
    assert(
        !ok2 &&
        "SPFA 应检测到负环并返回 false");  // 负环检测: cnt[i]>=N 时提前退出
                                           // :contentReference[oaicite:2]{index=2}

    // 对于负环，prev2 中未触达节点仍应为 SIZE_MAX（如节点 2 或其他）
    for (std::size_t i = 0; i < prev2.size(); ++i) {
      if (dist2[i] == INF) {
        assert(prev2[i] == SIZE_MAX && "不可达节点前驱应保持 SIZE_MAX");
      }
    }

    std::cout << "Test 2 (negative cycle) passed.\n";
  }

  std::cout << "All SPFA-with-prev tests passed!\n";
  return 0;
}

int test_adj_floyd() {
  constexpr std::size_t N = 4;
  const int INF = 1000000;

  // 1) 构造 4 节点图，默认边权为 INF
  adjacent_matrix<int> g(N, INF);

  // 2) 添加若干有向边
  //    图结构：
  //      0 → 1 (5)
  //      0 → 3 (10)
  //      1 → 2 (3)
  //      2 → 3 (1)
  g.add_edge(0, 1, 5);
  g.add_edge(0, 3, 10);
  g.add_edge(1, 2, 3);
  g.add_edge(2, 3, 1);

  // 3) 运行 floyd
  auto d = g.floyd();

  // 4) 预期距离矩阵
  //    从 i 到 j 的最短距离：
  //      0→0 = 0
  //      0→1 = 5
  //      0→2 = 5+3=8
  //      0→3 = min(10, 8+1)=9
  //      1→0 = INF (不可达)
  //      1→1 = 0
  //      1→2 = 3
  //      1→3 = 3+1=4
  //      2→0 = INF
  //      2→1 = INF
  //      2→2 = 0
  //      2→3 = 1
  //      3→* = INF except 3→3=0
  std::vector<std::vector<int>> exp = {
      {0, 5, 8, 9}, {INF, 0, 3, 4}, {INF, INF, 0, 1}, {INF, INF, INF, 0}};

  // 5) 验证每个条目
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      assert(d[i][j] == exp[i][j] && "Floyd 计算结果不符预期");
    }
  }

  std::cout << "Floyd 测试通过！\n";
  return 0;
}

int test_lst_dijkstra() {
  // 构造一个 5 节点图
  adjacent_list<int> g(5);

  // 添加边：0→1 (2), 0→2 (4)
  int d01 = 2, d02 = 4;
  g.add_edge(0, 1, d01);
  g.add_edge(0, 2, d02);

  // 添加更多边：1→2 (1), 1→3 (7)
  int d12 = 1, d13 = 7;
  g.add_edge(1, 2, d12);
  g.add_edge(1, 3, d13);

  // 2→4 (3), 3→4 (1)
  int d24 = 3, d34 = 1;
  g.add_edge(2, 4, d24);
  g.add_edge(3, 4, d34);

  // 运行 Dijkstra
  auto dist = g.dijkstra(0);

  // 预期距离：from 0 到 [0, 2, 3, 9, 6]
  std::vector<int> expected = {0, 2, 3, 9, 6};

  if (dist.size() != expected.size()) {
    std::cerr << "Dijkstra with heap: Size mismatch\n";
    return 1;
  }
  for (std::size_t i = 0; i < dist.size(); ++i) {
    if (dist[i] != expected[i]) {
      std::cerr << "Dijkstra with heap: Mismatch at node " << i << ": got "
                << dist[i] << ", expected " << expected[i] << "\n";
      return 2;
    }
  }

  std::cout << "Dijkstra with heap: All tests passed.\n";
  return 0;
}

int test_lst_spfa() {
  {
    // —— 测试 1：无负环的有向图 ——
    // 节点 0→1→2→3，且 0→3 还有一条直达边
    // 结构：
    //   0 -5-> 1 -2-> 2 -1-> 3
    //   0 -9-> 3
    adjacent_list<int> g(4, /*inf=*/1000000);
    g.add_edge(0, 1, 5);
    g.add_edge(1, 2, 2);
    g.add_edge(2, 3, 1);
    g.add_edge(0, 3, 9);

    std::vector<int> dist;
    bool ok = g.spfa(0, dist);
    assert(ok && "SPFA on acyclic graph should succeed");

    // 预期最短距离：0→{0,1,2,3} = {0,5,7,8}
    std::vector<int> exp = {0, 5, 7, 8};
    std::cout << "No-cycle distances:";
    for (auto d : dist) std::cout << " " << d;
    std::cout << std::endl;

    assert(dist == exp && "Distances mismatch in no-cycle test");
  }

  {
    // —— 测试 2：带负环的图 ——
    // 0→1 (1), 1→2 (-2), 2→1 (1) 构成净权重 -1 的负环
    adjacent_list<int> g(3, /*inf=*/1000000);
    g.add_edge(0, 1, 1);
    g.add_edge(1, 2, -2);
    g.add_edge(2, 1, 1);

    std::vector<int> dist;
    bool ok = g.spfa(0, dist);
    assert(!ok && "SPFA should detect negative cycle");
    std::cout << "Negative cycle correctly detected." << std::endl;
  }

  std::cout << "All SPFA tests passed!\n";
  return 0;
}

int test_lst_dijkstra_persist() {
  constexpr std::size_t N = 5;
  // 构造一个 5 节点图，无边时默认 inf
  adjacent_list<int> g(N, /*inf=*/1000000);

  // 添加一些有向边：0->1 (10), 0->2 (3), 2->1 (1), 2->3 (2), 1->3 (2), 3->4 (1)
  g.add_edge(0, 1, 10);
  g.add_edge(0, 2, 3);
  g.add_edge(2, 1, 1);
  g.add_edge(2, 3, 2);
  g.add_edge(1, 3, 2);
  g.add_edge(3, 4, 1);

  // 运行 dijkstra 并获取 dist 与 prev
  std::vector<std::size_t> prev;
  auto dist = g.dijkstra(0, prev);

  // 1) 验证 prev 大小和值域
  assert(prev.size() == N);
  for (std::size_t v = 0; v < N; ++v) {
    // prev 值要么是 SIZE_MAX（起点或不可达），要么 < N
    assert(prev[v] == SIZE_MAX || prev[v] < N);
  }

  // 2) 起点 prev[0] 必须为 SIZE_MAX
  assert(prev[0] == SIZE_MAX);

  // 3) 验证各节点的最短前驱
  // 手工计算最短路径树：
  // 0: dist=0, prev=SIZE_MAX
  // 1: 最短路径 0->2->1, 所以前驱 prev[1] 应为 2
  // 2: 最短路径 0->2,       prev[2] == 0
  // 3: 最短路径 0->2->3,    prev[3] == 2
  // 4: 最短路径 0->2->3->4, prev[4] == 3
  std::vector<std::size_t> expected_prev = {
      SIZE_MAX,  // 0
      2,         // 1
      0,         // 2
      2,         // 3
      3          // 4
  };

  for (std::size_t v = 0; v < N; ++v) {
    std::cout << "prev[" << v << "] = ";
    if (prev[v] == SIZE_MAX)
      std::cout << "SIZE_MAX";
    else
      std::cout << prev[v];
    std::cout << std::endl;

    assert(prev[v] == expected_prev[v] && "前驱数组值与预期不符");
  }

  std::cout << "Dijkstra with prev 测试通过！" << std::endl;
  return 0;
}

int test_lst_spfa_persist() {
  constexpr std::size_t N = 5;
  const int INF = 1000000;

  // —— 测试 1：无负环场景 ——
  {
    // 构造 5 节点图
    adjacent_list<int> g(N, INF);
    // 添加边：0→1(4), 0→2(2), 2→1(1), 1→3(5), 2→3(8), 3→4(3)
    g.add_edge(0, 1, 4);
    g.add_edge(0, 2, 2);
    g.add_edge(2, 1, 1);
    g.add_edge(1, 3, 5);
    g.add_edge(2, 3, 8);
    g.add_edge(3, 4, 3);

    std::vector<int> dist;
    std::vector<std::size_t> prev;
    bool ok = g.spfa(0, dist, prev);
    assert(
        ok &&
        "SPFA 在无负环图上应返回 true");  // SPFA 成功返回 true
                                          // :contentReference[oaicite:0]{index=0}

    // 预期最短距离：{0, 3, 2, 8, 11}
    std::vector<int> exp_dist = {0, 3, 2, 8, 11};
    // 预期前驱： prev[0]=SIZE_MAX, prev[1]=2, prev[2]=0, prev[3]=1, prev[4]=3
    std::vector<std::size_t> exp_prev = {SIZE_MAX,  // 源点无前驱
                                         2, 0, 1, 3};

    for (std::size_t i = 0; i < N; ++i) {
      assert(dist[i] == exp_dist[i] && "距离计算错误");
      assert(prev[i] == exp_prev[i] &&
             "前驱记录错误");  // 前驱更新逻辑 O(1)
                               // :contentReference[oaicite:1]{index=1}
    }

    std::cout << "Test 1 (no negative cycle) passed.\n";
  }

  // —— 测试 2：带负环场景 ——
  {
    // 构造含 3 节点的图：0→1(1), 1→2(-2), 2→1(1) 形成净权 -1 的负环
    adjacent_list<int> g2(3, INF);
    g2.add_edge(0, 1, 1);
    g2.add_edge(1, 2, -2);
    g2.add_edge(2, 1, 1);

    std::vector<int> dist2;
    std::vector<std::size_t> prev2;
    bool ok2 = g2.spfa(0, dist2, prev2);
    assert(
        !ok2 &&
        "SPFA 应检测到负环并返回 false");  // 负环检测: cnt[i]>=N 时提前退出
                                           // :contentReference[oaicite:2]{index=2}

    // 对于负环，prev2 中未触达节点仍应为 SIZE_MAX（如节点 2 或其他）
    for (std::size_t i = 0; i < prev2.size(); ++i) {
      if (dist2[i] == INF) {
        assert(prev2[i] == SIZE_MAX && "不可达节点前驱应保持 SIZE_MAX");
      }
    }

    std::cout << "Test 2 (negative cycle) passed.\n";
  }

  std::cout << "All SPFA-with-prev tests passed!\n";
  return 0;
}

int test_lst_dfs() {
  // INF 定义
  const int INF = std::numeric_limits<int>::max();

  // 1. 构造一个 5 节点的无向图
  adjacent_list<int> g(5, INF);

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

int test_lst_bfs() {
  // 1) 构造 5 个节点的图，默认 inf = 10000000
  adjacent_list<int> g(5);

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
  test_adj_spfa();
  test_adj_dijkstra_persist();
  test_adj_spfa_persist();
  test_adj_floyd();
  test_lst_dijkstra();
  test_lst_spfa();
  test_lst_dijkstra_persist();
  test_lst_spfa_persist();
  test_lst_dfs();
  test_lst_bfs();
}
