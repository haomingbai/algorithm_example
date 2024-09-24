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
    bool operator<(node_with_distance &a)
    {
        return distance < a.distance;
    }
    bool operator<(node_with_distance &&a)
    {
        return distance < a.distance;
    }
    bool operator>(node_with_distance &a)
    {
        return distance > a.distance;
    }
    bool operator>(node_with_distance &&a)
    {
        return distance > a.distance;
    }
};

/// @brief This to_edge indicates an edge to node destination.
struct to_edge
{
    size_t destination;
    double length;
};

/// @brief A graph used to perform dijkstra
/// @tparam node_number the number of nodes.
template <size_t node_number>
class graph_for_dijkstra
{
private:
    array<size_t, node_number> head;
    vector<size_t> nxt;
    vector<to_edge> to;

public:
    graph_for_dijkstra()
    {
        head.fill(-1);
        nxt.reserve(node_number);
        to.reserve(node_number);
    }
    void add_edge(size_t departure, size_t terminal, double length)
    {
        nxt.emplace_back(head[departure]);
        head[departure] = to.size(); // This will be the index of the edge to be added.
        // After the two steps, nxt[head[departure]] means the edge added last time before the edge mentioned in head[departure]. At this time, head[departure] means that the edge being added should be stored in to[head[departure]] while nxt[head[departure]] store the previous head[departure].
        to.emplace_back({terminal, length});
    }
    array<double, node_number> &&dijkstra(size_t departure)
    {
        priority_queue<node_with_distance> autosort_queue; // This data structure can automatically return the node with largest edge;
        std::array<bool, node_number> visited;
        std::array<double, node_number> distance;
        visited.fill(0);
        distance.fill(1e15);
        autosort_queue.push({departure, 0});
        while (!autosort_queue.empty())
        {
            auto node_to_process = autosort_queue.top();
            distance[node_to_process.node_index] = node_to_process.distance;
            visited[node_to_process.node_index] = 1;
            autosort_queue.pop();
            for (auto i = head[node_to_process.node_index]; i != -1; i = nxt[i])
            {
                if (!visited[to[i].destination] && distance[node_number.node_index] + to[i].length < distance[to[i].destination])
                {
                    distance[to[i].destination] = distance[node_number.node_index] + to[i].length;
                }
            }
            while (visited[autosort_queue.top().node_index])
            {
                autosort_queue.pop();
            }
        }
        return move(distance);
    }
};

auto main(int argc, char **argv) -> int
{
}