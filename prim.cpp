#include <iostream>
#include <queue>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <array>
#include <list>

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

struct edge
{
    size_t start;
    size_t end;
    double length;
    edge(size_t start, size_t end, double length):start(start), end(end), length(length){}
};


/// @brief A graph used to perform dijkstra
/// @tparam node_number the number of nodes.
template <size_t node_number>
class graph_for_prim
{
private:
    array<size_t, node_number> head;
    vector<size_t> nxt;
    vector<to_edge> to;
public:
    graph_for_prim();
    ~graph_for_prim();
    auto prim(size_t from) -> vector<edge>&&
    {
        array<bool, node_number> visited(0);
        vector<edge> &selected = *new vector<edge>;
        size_t curr = from;
        priority_queue<edge> processing;
        for (auto i = head[curr]; i != -1; i = nxt[i])
        {
            processing.emplace(curr, to[i].destination, to[i].length);
        }
        size_t cnt(0);
        while (!processing.empty() && cnt < node_number)
        {
            while (processing.top())
            {
                /* code */
            }
            
        }
        
    }
};

template <size_t node_number>
graph_for_prim<node_number>::graph_for_prim()
{
    this->head.fill(-1);
    this->nxt.reserve(node_number);
    this->to.reserve(node_number);
}

template <size_t node_number>
graph_for_prim<node_number>::~graph_for_prim<node_number>()
{
}
