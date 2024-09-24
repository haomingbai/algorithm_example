#include <cstddef>
#include <memory>
#include <queue>
#include <algorithm>
#include <iostream>
#include <utility>

#define inf 1e15

/// @brief Dijkstra algorithm, without any optimization.
/// @param graph This parameter is a graph which indicates the distance from i to j (graph[i][j]).
/// @param point_num The size of the graph, gneerally the number of points.
/// @param start The beginning point.
/// @return The distance from start to every point
double *dijkstra(double **graph, size_t point_num, size_t start)
{
    double *distance = new double[point_num]();
    bool *visited = new bool[point_num]();

    for (size_t i = 0; i < point_num; i++)
    {
        distance[i] = graph[start][i];
    }
    distance[start] = 0;
    visited[start] = true;

    for (size_t i = 0; i < point_num; i++)
    {
        // First, find the unvisited pt with the least distance
        size_t to_process = inf;
        double min_dis = inf;
        for (size_t j = 0; j < point_num; j++)
        {
            if (!visited[j] && min_dis > distance[j])
            {
                min_dis = distance[j];
                to_process = j;
            }
        }
        if (to_process == inf)
        {
            break;
        }

        // Then, mark this point to be visited
        visited[to_process] = true;

        // Finally, we process the point with points near to_process
        for (size_t j = 0; j < point_num; j++)
        {
            if (graph[to_process][j] != inf)
            {
                distance[j] = std::min(distance[j], graph[to_process][j] + distance[to_process]);
            }
        }
    }
    return distance;
}

int main()
{
    std::cout << "Hell world!" << std:: endl;
}