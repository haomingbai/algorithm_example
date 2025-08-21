/**
 * @file avl.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-07-20
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>

// 模板化 AVLTree：T 为元素类型，Compare 为比较器（默认 std::less<T>）。
// 要求：T 可比较（Compare 能对 T 做严格弱序比较），count/subtree_size 采用
// size_t 存储重复元素计数。
template <typename T, typename Compare = std::less<T>>
class AVLTree {
 public:
  // 方向枚举：左 / 右（便于阅读）
  enum direction { LEFT = 0, RIGHT = 1 };

  // 节点结构
  struct Node {
    T elem;                      // 节点值（泛型）
    Node* parent = nullptr;      // 父指针
    std::array<Node*, 2> child;  // child[LEFT], child[RIGHT]
    long height = 0;             // 节点高度（空子树高度记为 -1）
    size_t count = 1;            // 相同值的个数（处理多重集合）
    size_t subtree_size = 1;     // 以此节点为根的子树中元素总数（包含重复）

    Node(const T& e, Node* p = nullptr)
        : elem(e), parent(p), height(0), count(1), subtree_size(1) {
      child[LEFT] = child[RIGHT] = nullptr;
    }

    // 更新高度：max(左高, 右高) + 1（空子节点高度记为 -1）
    void updateHeight() {
      long lh = child[LEFT] ? child[LEFT]->height : -1;
      long rh = child[RIGHT] ? child[RIGHT]->height : -1;
      height = 1 + std::max(lh, rh);
    }

    // 更新子树大小（包含 count 自身重复数）
    void updateSubtreeSize() {
      size_t ls = child[LEFT] ? child[LEFT]->subtree_size : 0;
      size_t rs = child[RIGHT] ? child[RIGHT]->subtree_size : 0;
      subtree_size = ls + rs + count;
    }

    // 获取平衡因子：左高 - 右高
    long getBalanceFactor() const {
      long lh = child[LEFT] ? child[LEFT]->height : -1;
      long rh = child[RIGHT] ? child[RIGHT]->height : -1;
      return lh - rh;
    }
  };

  // 构造/析构
  AVLTree() : root(nullptr) {}
  ~AVLTree() { clear(root); }

  // 禁止拷贝（简单起见），可以按需实现拷贝构造/赋值
  AVLTree(const AVLTree&) = delete;
  AVLTree& operator=(const AVLTree&) = delete;

  // 插入元素（支持重复）
  void insert(const T& value) {
    if (!root) {
      root = new Node(value);
      return;
    }
    Node* cur = root;
    Node* inserted_parent = nullptr;
    while (true) {
      // 若 value 小于 cur->elem，走左子树
      if (comp(value, cur->elem)) {
        cur->subtree_size++;  // 路径上的子树大小都需要 +1
        if (!cur->child[LEFT]) {
          cur->child[LEFT] = new Node(value, cur);
          inserted_parent = cur;
          break;
        }
        cur = cur->child[LEFT];
      }
      // 若 cur->elem < value，走右子树
      else if (comp(cur->elem, value)) {
        cur->subtree_size++;
        if (!cur->child[RIGHT]) {
          cur->child[RIGHT] = new Node(value, cur);
          inserted_parent = cur;
          break;
        }
        cur = cur->child[RIGHT];
      }
      // 相等，则只增加计数与子树大小（不破坏结构）
      else {
        cur->count++;
        cur->subtree_size++;
        return;
      }
    }
    // 插入完成后从插入点的父节点向上修复平衡
    repairBalance(inserted_parent);
  }

  // 删除一个元素（如果存在）：若 count>1 则只减
  // count，否则真正移除节点并旋转修复
  void remove(const T& value) {
    Node* node = find(value);
    if (!node) return;

    // 先将从 node 到根路径的 subtree_size 全部 -1（移走一个元素）
    for (Node* p = node; p; p = p->parent) {
    }

    if (node->count > 1) {
      node->count--;
      return;
    }

    removeNode(node);
  }

  // 获取 value 的排名（比 value 小的元素个数 + 1）
  // 返回值为 size_t（从 1 开始）
  size_t getRank(const T& value) const {
    size_t rank = 1;
    Node* cur = root;
    while (cur) {
      if (comp(value, cur->elem)) {
        cur = cur->child[LEFT];
      } else if (comp(cur->elem, value)) {
        size_t left_size =
            cur->child[LEFT] ? cur->child[LEFT]->subtree_size : 0;
        rank += left_size + cur->count;
        cur = cur->child[RIGHT];
      } else {
        size_t left_size =
            cur->child[LEFT] ? cur->child[LEFT]->subtree_size : 0;
        rank += left_size;
        return rank;
      }
    }
    return rank;  // 如果没找到，rank 是插入该元素后应该有的排名
  }

  // 根据排名 k（1..N）查找元素（若 k 越界则抛出异常）
  T findByRank(size_t k) const {
    Node* cur = root;
    while (cur) {
      size_t left_size = cur->child[LEFT] ? cur->child[LEFT]->subtree_size : 0;
      if (k <= left_size) {
        cur = cur->child[LEFT];
      } else if (k > left_size + cur->count) {
        k -= (left_size + cur->count);
        cur = cur->child[RIGHT];
      } else {
        return cur->elem;
      }
    }
    throw std::out_of_range("findByRank: k out of range");
  }

  // 查询 value 的前驱（即 < value 的最大元素）。若不存在返回 std::nullopt。
  std::optional<T> getPredecessor(const T& value) const {
    Node* cur = root;
    std::optional<T> pred = std::nullopt;
    while (cur) {
      if (comp(cur->elem, value)) {  // cur->elem < value
        pred = cur->elem;
        cur = cur->child[RIGHT];
      } else {
        cur = cur->child[LEFT];
      }
    }
    return pred;
  }

  // 查询 value 的后继（即 > value 的最小元素）。若不存在返回 std::nullopt。
  std::optional<T> getSuccessor(const T& value) const {
    Node* cur = root;
    std::optional<T> succ = std::nullopt;
    while (cur) {
      if (comp(value, cur->elem)) {  // value < cur->elem
        succ = cur->elem;
        cur = cur->child[LEFT];
      } else {
        cur = cur->child[RIGHT];
      }
    }
    return succ;
  }

  // 返回树中元素总量（包含重复）
  size_t size() const { return root ? root->subtree_size : 0; }

  // 是否为空
  bool empty() const { return root == nullptr; }

 private:
  Node* root;
  Compare comp;  // 比较器实例

  // 递归删除全部节点（析构时使用）
  void clear(Node* node) {
    if (!node) return;
    clear(node->child[LEFT]);
    clear(node->child[RIGHT]);
    delete node;
  }

  // 查找值为 value 的节点（找到返回指针，否则返回 nullptr）
  Node* find(const T& value) const {
    Node* cur = root;
    while (cur) {
      if (comp(value, cur->elem)) {
        cur = cur->child[LEFT];
      } else if (comp(cur->elem, value)) {
        cur = cur->child[RIGHT];
      } else {
        return cur;
      }
    }
    return nullptr;
  }

  // 将 old_node 替换为 new_node（用于删除/移植）
  void transplant(Node* old_node, Node* new_node) {
    if (!old_node->parent) {
      root = new_node;
    } else if (old_node == old_node->parent->child[LEFT]) {
      old_node->parent->child[LEFT] = new_node;
    } else {
      old_node->parent->child[RIGHT] = new_node;
    }
    if (new_node) {
      new_node->parent = old_node->parent;
    }
  }

  // 删除具体节点（节点已确定 count == 1）
  void removeNode(Node* node) {
    Node* node_to_balance = nullptr;

    // 情形 1：至少有一个孩子为空（0 或 1 个孩子）
    if (!node->child[LEFT] || !node->child[RIGHT]) {
      node_to_balance = node->parent;
      Node* child = node->child[LEFT] ? node->child[LEFT] : node->child[RIGHT];
      transplant(node, child);
    } else {
      // 情形 2：左右孩子都存在 —— 找后继（右子树最左）
      Node* succ = node->child[RIGHT];
      while (succ->child[LEFT]) succ = succ->child[LEFT];

      // succ 的父节点（可能是 node 本身）
      node_to_balance = succ->parent;

      // 从 succ.parent 到 node（不含 node）的路径上，succ->count
      // 个元素已经移走， 因此需要把这些节点的 subtree_size 扣掉 succ->count
      for (Node* p = succ->parent; p && p != node; p = p->parent) {
        p->subtree_size -= succ->count;
      }

      if (succ->parent != node) {
        // 把 succ 用其右子替代
        transplant(succ, succ->child[RIGHT]);
        // 连接 succ 的右子为 node 的右子
        succ->child[RIGHT] = node->child[RIGHT];
        if (succ->child[RIGHT]) succ->child[RIGHT]->parent = succ;
      } else {
        node_to_balance = succ;
      }

      // 用 succ 替换 node
      transplant(node, succ);
      // 连接 succ 的左子为 node 的左子
      succ->child[LEFT] = node->child[LEFT];
      if (succ->child[LEFT]) succ->child[LEFT]->parent = succ;

      // 更新 succ 的元信息（count 保持 succ 原来的 count）
      succ->updateSubtreeSize();
      succ->updateHeight();
    }

    delete node;
    if (node_to_balance) repairBalance(node_to_balance);
  }

  // 旋转操作：以 v 为根进行一次单旋转（dir 参数语义与原实现保持一致）
  // 注意：我们沿用了原实现的 dir/ opposite_dir 表达，使得代码逻辑等价。
  Node* rotate(Node* v, direction dir) {
    direction opp = static_cast<direction>(1 - dir);
    Node* u = v->child[opp];  // 将 u 提到 v 的位置
    Node* parent = v->parent;

    // v 的 opp 子指向 u 的 dir 子
    v->child[opp] = u->child[dir];
    if (u->child[dir]) u->child[dir]->parent = v;

    // u 的 dir 子指向 v；更新父链
    u->child[dir] = v;
    v->parent = u;

    // 将 u 连接到原来 v 的父节点上
    u->parent = parent;
    if (parent) {
      if (v == parent->child[LEFT])
        parent->child[LEFT] = u;
      else
        parent->child[RIGHT] = u;
    } else {
      root = u;
    }

    // 先更新 v（现在是 u 的子），再更新 u
    v->updateHeight();
    v->updateSubtreeSize();
    u->updateHeight();
    u->updateSubtreeSize();

    return u;
  }

  // 修复从 node 向上直到根的平衡（维护高度与 subtree_size）
  void repairBalance(Node* node) {
    while (node) {
      node->updateHeight();
      node->updateSubtreeSize();
      long bf = node->getBalanceFactor();

      if (bf > 1) {
        // 左子树偏高，检查是否需要先做左子节点的左/右旋
        if (node->child[LEFT] && node->child[LEFT]->getBalanceFactor() < 0) {
          // LR 情形：先对左子做左旋（把左子->右子提升），等效原代码
          // rotate(node->child[LEFT], LEFT)
          rotate(node->child[LEFT], LEFT);
        }
        // 然后对 node 做右旋
        node = rotate(node, RIGHT);
      } else if (bf < -1) {
        // 右子树偏高，可能为 RL / RR
        if (node->child[RIGHT] && node->child[RIGHT]->getBalanceFactor() > 0) {
          // RL 情形：先对右子做右旋
          rotate(node->child[RIGHT], RIGHT);
        }
        // 然后对 node 做左旋
        node = rotate(node, LEFT);
      }
      // 向上继续
      node = node->parent;
    }
  }
};  // class AVLTree
