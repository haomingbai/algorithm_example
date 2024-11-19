#include <iostream>
#include <queue>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <array>

using namespace std;

/// @brief This struct is used to store the node with the shortest distance,
struct node_with_distance
{
    size_t node_index;
    double distance;
    node_with_distance(pair<int, double> &&node) : node_index(node.first), distance(node.second) {}
    node_with_distance(size_t idx, double dis) : node_index(idx), distance(dis) {}
};

bool operator<(const node_with_distance a, const node_with_distance b)
{
    return a.distance < b.distance;
}

/// @brief This to_edge indicates an edge to node destination.
struct to_edge
{
    size_t destination;
    double length;
    to_edge(size_t des, double len) : destination(des), length(len) {}
};

template <size_t node_number>
class graph_for_bellman_ford
{
private:
    array<size_t, node_number> head;
    vector<size_t> nxt;
    vector<to_edge> to;

public:
    graph_for_bellman_ford(size_t edge_number);
    graph_for_bellman_ford();
    auto add_edge(size_t start, size_t end, double length = 1) -> void
    {
        this->nxt.push_back(head[start]);
        this->head[start] = to.size();
        this->to.emplace_back(end, length);
    }
    bool bellman_ford(size_t from, array<double, node_number> &storage)
    {
        constexpr long long inf = 1e15;
        storage.fill(inf);
        storage[from] = 0;
        for (size_t i = 0; i < node_number; i++)
        {
            for (size_t j = 0; j < node_number; j++)
            {
                for (size_t k = head[j]; k != -1; k = nxt[k])
                {
                    storage[to[k].destination] = min(storage[to[k].destination], storage[j] + to[k].length);
                }
            }
        }
        for (size_t i = 0; i < node_number; i++)
        {
            for (size_t j = head[i]; j != -1; j = nxt[j])
            {
                if (storage[to[j].destination] > storage[i] + to[j].length)
                    return true;
            }
        } // We can use this behavior to decide whether there is negative loops because when we can optimize any path after n times of operations, there must be negative loops where we can make the path short with going along the loop.
        return false;
    }
};

template <size_t node_number>
graph_for_bellman_ford<node_number>::graph_for_bellman_ford(size_t edge_number)
{
    head.fill(-1);
    nxt.reserve(edge_number), to.reserve(edge_number);
}

template <size_t node_number>
graph_for_bellman_ford<node_number>::graph_for_bellman_ford()
{
    head.fill(-1);
    nxt.reserve(node_number), to.reserve(node_number);
}



auto main(int argc, char **argv) -> int
{
    auto p = new graph_for_bellman_ford<5>(5);
    p->add_edge(0, 2, 2);
    p->add_edge(2, 0, 2);
    p->add_edge(0, 3, 3);
    p->add_edge(3, 0, 3);
    p->add_edge(0, 4);
    p->add_edge(4, 0);
    p->add_edge(0, 1, 5);
    p->add_edge(1, 0, 5);
    // cout << "added" << endl;
    array<double,5> arr;
    auto y = p->bellman_ford(2, arr);
    for (auto it : arr)
    {
        cout << it << endl;
    }
    cout << y << endl;
}