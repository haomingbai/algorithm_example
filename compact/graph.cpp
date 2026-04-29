#include <algorithm>
#include <cstddef>
#include <queue>
#include <vector>

struct DSU {
  std::vector<int> fa, sz;
  DSU(int n) : fa(n), sz(n, 1) { for (int i = 0; i < n; i++) fa[i] = i; }
  int find(int x) { return fa[x] == x ? x : fa[x] = find(fa[x]); }
  void unite(int a, int b) {
    a = find(a); b = find(b);
    if (a == b) return;
    if (sz[a] < sz[b]) std::swap(a, b);
    fa[b] = a; sz[a] += sz[b];
  }
  bool same(int a, int b) { return find(a) == find(b); }
};

struct Graph {
  struct Edge { int to; long long w; };
  int n;
  std::vector<std::vector<Edge>> adj;

  Graph(int n) : n(n), adj(n) {}
  void addEdge(int u, int v, long long w) { adj[u].push_back({v, w}); }

  std::vector<long long> dijkstra(int s) {
    const long long INF = 1e18;
    std::vector<long long> dist(n, INF);
    dist[s] = 0;
    std::priority_queue<std::pair<long long, int>, std::vector<std::pair<long long, int>>, std::greater<>> pq;
    pq.push({0, s});
    while (!pq.empty()) {
      auto [d, u] = pq.top(); pq.pop();
      if (d > dist[u]) continue;
      for (auto &e : adj[u]) {
        if (dist[u] + e.w < dist[e.to]) {
          dist[e.to] = dist[u] + e.w;
          pq.push({dist[e.to], e.to});
        }
      }
    }
    return dist;
  }
};

struct KruskalEdge {
  int u, v;
  long long w;
  bool operator<(const KruskalEdge &o) const { return w < o.w; }
};

std::vector<KruskalEdge> kruskal(std::vector<KruskalEdge> edges, int n) {
  std::sort(edges.begin(), edges.end());
  DSU dsu(n);
  std::vector<KruskalEdge> res;
  for (auto &e : edges) {
    if (!dsu.same(e.u, e.v)) {
      res.push_back(e);
      dsu.unite(e.u, e.v);
    }
  }
  return res;
}
