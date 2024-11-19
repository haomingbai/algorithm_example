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
    return a.distance > b.distance;
}

/// @brief This to_edge indicates an edge to node destination.
struct to_edge
{
    size_t destination;
    double length;
    to_edge(size_t des, double len) : destination(des), length(len) {}
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
    /// @brief Add an edge from depature to terminal to the graph
    /// @param departure The starting point of the edge, generally, we regard the edge as directional
    /// @param terminal The end point of the edge
    /// @param length The length of the edge.
    /// @return return nothing, if there is exception, you can just catch
    auto add_edge(size_t departure, size_t terminal, double length = 1) -> void
    {
        nxt.emplace_back(head[departure]);
        head[departure] = to.size(); // This will be the index of the edge to be added.
        // After the two steps, nxt[head[departure]] means the edge added last time before the edge mentioned in head[departure]. At this time, head[departure] means that the edge being added should be stored in to[head[departure]] while nxt[head[departure]] store the previous head[departure].
        to.emplace_back(terminal, length);
    }

    /// @brief Perform a dijkstra algorithm
    /// @param departure the starting point.
    /// @return an array, which contains the distance from the starting point to other end points.
    auto dijkstra(size_t departure) -> array<double, node_number> &&
    {
        priority_queue<node_with_distance> autosort_queue;                                // This data structure can automatically return the node with largest edge, which is always prrority_queue::top();
        std::array<bool, node_number> visited;                                            // visited should be used to mark whether the distance to a node can be made sure.
        std::array<double, node_number> &distance = *new std::array<double, node_number>; // This is used to store the distance from departure to the i_th node.
        visited.fill(0);                                                                  // All pts are unvisited.
        distance.fill(1e15);                                                              // 1e15 means the max value of a path, generally means infinity.
        autosort_queue.emplace(departure, 0.0);                                           // It is easy for us to know that the point can get to itself in distance 0, so we add this into queue.
        while (!autosort_queue.empty())                                                   // when the queue is not empty.
        {
            auto node_to_process = autosort_queue.top(); // The path with least distance, which we are going to process and mark as visited.
            distance[node_to_process.node_index] = node_to_process.distance; // The distance, though unnecessary, should be made sure again and again.
            visited[node_to_process.node_index] = 1; // At this step, we mark the node with shortest path as visited
            autosort_queue.pop(); // Remove this node now
            for (auto i = head[node_to_process.node_index]; i != -1; i = nxt[i])
            {
                if (!visited[to[i].destination] && distance[node_to_process.node_index] + to[i].length < distance[to[i].destination])
                {
                    distance[to[i].destination] = distance[node_to_process.node_index] + to[i].length;
                    autosort_queue.emplace(to[i].destination, distance[to[i].destination]);
                }
                // We compare the current path from start to end with a new path, which is the from start to node_to_process, then from node_to_process to end.
            }
            while (!autosort_queue.empty() && visited[autosort_queue.top().node_index])
            {
                autosort_queue.pop();
                // Remove the unused path.
            }
        }
        return move(distance);
    }
};

auto main(int argc, char **argv) -> int
{
    auto p = new graph_for_dijkstra<5>();
    p->add_edge(0, 2, 2);
    p->add_edge(2, 0, 2);
    p->add_edge(0, 3, 3);
    p->add_edge(3, 0, 3);
    p->add_edge(0, 4);
    p->add_edge(4, 0);
    p->add_edge(0, 1, 5);
    p->add_edge(1, 0, 5);
    // cout << "added" << endl;
    auto y = p->dijkstra(2);
    for (auto it : y)
    {
        cout << it << endl;
    }
}