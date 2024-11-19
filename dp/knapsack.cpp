#include <iostream>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>

using namespace std;

typedef unsigned long long int ull;

auto knapsack(vector<pair<ull, ull>> &value_weight, ull max_weight) -> pair<ull, ull> &&
{
    if (value_weight.empty())
    {
        return make_pair(0, 0); // With a cost of zero, we can get the value of zero
    }
    vector<map<ull, ull>> sto;
    sto[0][0] = 0, sto[0][value_weight.front().second] = value_weight.front().first;
    size_t obj_num = value_weight.size();
    for (size_t i = 1; i < obj_num; i++)
    {
        // range from 0th to (obj_num - 1)th obj.
        for (auto &it : sto[i - 1])
        {
            // it.first indicates weight, second indicates value
            // "it" means the case with objs from 0 to i - 1
            sto[i][it.first] = max(sto[i][it.first], it.second); // No take the ith object, compare with another scheme, in which the ith obj was taken.
            if (it.first <= max_weight - value_weight[i].second) // no exceed the max weight
            {
                sto[i][it.first + value_weight[i].second] = it.second + value_weight[i].first; // take the ith obj
            }
        }
    }
    ull weight{0}, value{0};
    for (auto &&i : sto[obj_num - 1])
    {
        if (i.second > value)
        {
            weight = i.first, value = i.second;
        }
    }
    return make_pair(weight, value);
}

int main()
{
}