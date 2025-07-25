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
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <utility>

#include "./concepts.cpp"

template <FullyComparable DataType>
struct AVLTree {
  // 方向, 分为左和右.
  enum direction { LEFT = 0, RIGHT };

  struct AVLNode {
    DataType elem;              // 数据存放位置.
    AVLNode* parent = nullptr;  // 父节点指针

    // 子节点指针
    std::array<AVLNode*, 2> child = {nullptr, nullptr};
    long height = 0;  // 当前位置的子树高度.

    // 带有元素值和父节点的构造函数.
    AVLNode(const DataType& e, AVLNode* p = nullptr) : elem(e), parent(p) {}
    // 使用右值进行构造.
    AVLNode(DataType&& e, AVLNode* p = nullptr)
        : elem(std::move(e)), parent(p) {}

    // 更新当前节点的高度.
    // 高度定义为: 叶子节点高度为0, 空节点高度为-1.
    void updateHeight() {
      // 两个子树假设为空,
      // 初始化高度为-1.
      long left_height = -1;
      long right_height = -1;

      // 如果存在对应的叶子节点.
      // 此时更新高度.
      if (child[LEFT] != nullptr) {
        left_height = child[LEFT]->height;
      }
      if (child[RIGHT] != nullptr) {
        right_height = child[RIGHT]->height;
      }

      // 当前节点的高度是比较高的孩子节点高度还要大1
      height = 1 + std::max(left_height, right_height);
    }

    // 获取平衡因子 (左子树高度 - 右子树高度)
    long getBalanceFactor() const {
      // 获取左右子树的高度.
      long left_height = -1;
      long right_height = -1;

      // 如果两边子树存在
      if (child[LEFT] != nullptr) {
        left_height = child[LEFT]->height;
      }
      if (child[RIGHT] != nullptr) {
        right_height = child[RIGHT]->height;
      }

      // 平衡因子是左子树高度减去右子树高度.
      return left_height - right_height;
    }

    // 寻找中序遍历下的后继节点
    AVLNode* successor() {
      // 右子树不为空
      if (child[RIGHT] != nullptr) {
        // 后继在右子树
        AVLNode* curr = child[RIGHT];

        // 右子树中最小的就是后继
        while (curr->child[LEFT] != nullptr) {
          // 一直向左寻找
          curr = curr->child[LEFT];
        }
        return curr;
      }

      // 如果没有右子树，则向上查找
      // 获得父节点和当前节点
      AVLNode* parent = this->parent;
      AVLNode* curr = this;

      // 如果parent == nullptr,
      // 意味着已经不能向上找到后继了.
      // 那么这个时候待查找的节点就不存在后继
      // 如果存在这样的一个parent, 使得curr在parent的左侧,
      // 那么就找到了后继, 后继为parent.
      // 因为parent此时大于curr,
      // 而之前按顺序找到的第一个更大的节点就是parent.
      while (parent != nullptr && curr == parent->child[RIGHT]) {
        curr = parent;
        parent = parent->parent;
      }

      return parent;
    }

    // 寻找中序遍历下的前驱节点
    AVLNode* predcessor() {
      // 左子树不为空
      if (child[LEFT] != nullptr) {
        // 前驱在左子树
        AVLNode* curr = child[LEFT];

        // 左子树中最大的就是后继
        while (curr->child[RIGHT] != nullptr) {
          // 一直向右寻找
          curr = curr->child[RIGHT];
        }
        return curr;
      }

      // 如果没有左子树，则向上查找
      // 获得父节点和当前节点
      AVLNode* parent = this->parent;
      AVLNode* curr = this;

      // 如果parent == nullptr,
      // 意味着已经不能向上找到前驱了.
      // 那么这个时候待查找的节点就不存在前驱
      // 如果存在这样的一个parent, 使得curr在parent的有侧,
      // 那么就找到了前驱, 前驱为parent.
      // 因为parent此时大于curr,
      // 而之前按顺序找到的第一个更大的节点就是parent.
      while (parent != nullptr && curr == parent->child[LEFT]) {
        curr = parent;
        parent = parent->parent;
      }

      return parent;
    }
  };

  // 根节点
  AVLNode* root = nullptr;
  // 树大小
  std::size_t _size = 0;

  // 析构函数，释放所有节点
  ~AVLTree() { clear(root); }

  void clear(AVLNode* node) {
    if (node != nullptr) {
      clear(node->child[LEFT]);
      clear(node->child[RIGHT]);
      delete node;
    }
  }

  // 获取树大小.
  std::size_t size() const { return _size; }
  // 类标准库方法, 判断树是否为空.
  bool empty() const { return _size == 0; }

  // 左旋
  AVLNode* leftRotate(AVLNode* old_root) {
    assert(old_root != nullptr && old_root->child[RIGHT] != nullptr);
    auto new_root = old_root->child[RIGHT];
    auto parent = old_root->parent;

    // 步骤 1: 将 new_root 的左孩子过继给 old_root
    old_root->child[RIGHT] = new_root->child[LEFT];
    if (new_root->child[LEFT] != nullptr) {
      new_root->child[LEFT]->parent = old_root;
    }

    // 步骤 2: 将 new_root 连接到 parent
    new_root->parent = parent;
    if (parent == nullptr) {
      root = new_root;
    } else {
      if (old_root == parent->child[LEFT]) {
        parent->child[LEFT] = new_root;
      } else {
        parent->child[RIGHT] = new_root;
      }
    }

    // 步骤 3: 将 old_root 设置为 new_root 的左孩子
    new_root->child[LEFT] = old_root;
    old_root->parent = new_root;

    // 步骤 4: 更新高度 (必须先更新子节点，再更新父节点)
    old_root->updateHeight();
    new_root->updateHeight();

    return new_root;
  }

  // 右旋
  AVLNode* rightRotate(AVLNode* old_root) {
    assert(old_root != nullptr && old_root->child[LEFT] != nullptr);
    auto new_root = old_root->child[LEFT];
    auto parent = old_root->parent;

    // 步骤 1: 将 new_root 的右孩子过继给 old_root
    old_root->child[LEFT] = new_root->child[RIGHT];
    if (new_root->child[RIGHT] != nullptr) {
      new_root->child[RIGHT]->parent = old_root;
    }

    // 步骤 2: 将 new_root 连接到 parent
    new_root->parent = parent;
    if (parent == nullptr) {
      root = new_root;
    } else {
      if (old_root == parent->child[LEFT]) {
        parent->child[LEFT] = new_root;
      } else {
        parent->child[RIGHT] = new_root;
      }
    }

    // 步骤 3: 将 old_root 设置为 new_root 的右孩子
    new_root->child[RIGHT] = old_root;
    old_root->parent = new_root;

    // 步骤 4: 更新高度
    old_root->updateHeight();
    new_root->updateHeight();

    return new_root;
  }

  // 树的旋转.
  AVLNode* rotate(AVLNode node, direction dir) {
    switch (dir) {
      case LEFT:
        return leftRotate(node);
      case RIGHT:
        return rightRotate(node);
      default:
        assert(0);
        return nullptr;
    }
  }

  // 查找节点
  AVLNode* find(const DataType& key) {
    auto curr = root;

    // 如果curr是空指针, 那么就意味着没有找到key
    while (curr != nullptr && curr->elem != key) {
      if (key < curr->elem) {
        curr = curr->child[LEFT];
      } else {
        curr = curr->child[RIGHT];
      }
    }
    return curr;
  }

  // 插入元素
  bool insert(const DataType& elem) {
    // 空树的第一个元素.
    if (root == nullptr) {
      // try 防止构造失败
      try {
        root = new AVLNode(elem);
        _size++;
        return true;
      } catch (const std::exception& e) {
        root = nullptr;
        return false;
      }
    }

    // 从根节点开始查找
    AVLNode* curr = root;
    AVLNode* p = nullptr;

    // 当没有查找到空节点时
    while (curr != nullptr) {
      // 判断新节点应当在左子树还是右子树
      p = curr;

      // 如果待插入元素应当在左子树
      if (elem < curr->elem) {
        curr = curr->child[LEFT];
      }
      // 如果待插入元素更大, 应当在右子树
      else if (elem > curr->elem) {
        curr = curr->child[RIGHT];
      }
      // 相等意味着树中已经存在相应节点
      // 不应当重复插入.
      else {
        return false;  // 元素已存在
      }
    }
    // 如果curr变成空指针, 此时p是curr的父亲节点
    // 那么这意味着:
    // 如果elem < p->elem, 那么curr = p->child[LEFT]
    // p的左孩子已经留空,
    // elem > p->elem 时, 也是相同的原理,
    // p的右孩子已经留空.
    // 此时可以插入新节点.

    try {
      // 构造新节点.
      // 这里因为构造的时候已经传入p,
      // 所以新节点的父亲指针已经指向正确的位置(p).
      AVLNode* newNode = new AVLNode(elem, p);
      // 插入新节点
      if (elem < p->elem) {
        p->child[LEFT] = newNode;
      } else {
        assert(elem > p->elem);
        p->child[RIGHT] = newNode;
      }
      // 树的大小自增.
      _size++;

      // 修复树的平衡.
      repairBalance(p);  // 从新节点的父节点开始向上修复平衡
      return true;
    } catch (const std::exception& e) {
      return false;
    }
  }

  // 删除
  bool remove(const DataType& elem) {
    AVLNode* node_to_remove = find(elem);
    if (node_to_remove == nullptr) {
      return false;  // 节点不存在
    }

    AVLNode* node_to_rebalance_from = nullptr;  // 记录从哪里开始向上修复
    AVLNode* replacement_node = nullptr;  // 实际被删除或移动的节点的子节点

    // Case 1 & 2: 待删除节点有0个或1个孩子
    if (node_to_remove->child[LEFT] == nullptr ||
        node_to_remove->child[RIGHT] == nullptr) {
      // 这种时候, 应该从待删除节点的父亲节点进行平衡维护.
      // 因为移植过来的那个子节点的高度, 如果它存在, 就一定是0,
      // 也就是默认值.
      // 此时从待删除节点的父节点开始, 高度变小1, 可能需要更新高度,
      // 维护树的平衡.
      node_to_rebalance_from = node_to_remove->parent;

      // 获取待替换的节点.
      replacement_node = (node_to_remove->child[LEFT] != nullptr)
                             // 如果左子树不为空,
                             // 那么右子树一定为空.
                             // 把左子树移植上来即可.
                             ? node_to_remove->child[LEFT]
                             // 否则左子树为空
                             // 那么把右子树移植上来即可
                             : node_to_remove->child[RIGHT];
      // 两边都为空, 那么随便哪边移植上来都是空节点,
      // 那么移植一个空节点上来就很好.

      transplant(node_to_remove, replacement_node);
    }
    // Case 3: 待删除节点有2个孩子
    else {
      // 找到后继就可以
      AVLNode* successor = node_to_remove->successor();
      // 平衡应当从后继节点的父亲开始维护.
      // 因为后继被删除了, 所以最低的高度改变的子树就是它的父亲
      node_to_rebalance_from = successor->parent;
      // 要把这个节点移植上来.
      replacement_node = successor->child[RIGHT];

      // 如果后继节点不是待删除节点的直接子节点
      if (successor->parent != node_to_remove) {
        // 直接把后继的右孩子移植到位.
        transplant(successor, successor->child[RIGHT]);
        // 把后继节点移植到待删除节点的位置上.
        successor->child[RIGHT] = node_to_remove->child[RIGHT];
        // 把后继节点和它的右孩子节点连在一起
        node_to_remove->child[RIGHT]->parent = successor;
      } else {
        // 如果后继节点是待删除节点的右孩子，向上回溯的起点就是后继节点本身.
        // 这个时候, successor和待删除节点交换位置之后,
        // 右孩子孩子节点应该还是successor原来的右孩子.
        node_to_rebalance_from = successor;
      }

      // 把后继节点移植到待删除节点的位置.
      transplant(node_to_remove, successor);
      // 把后继节点和它的左孩子节点连接在一起.
      successor->child[LEFT] = node_to_remove->child[LEFT];
      node_to_remove->child[LEFT]->parent = successor;
    }

    // 释放内存, 树大小减少1
    delete node_to_remove;
    _size--;

    // 维护平衡
    if (node_to_rebalance_from != nullptr) {
      repairBalance(node_to_rebalance_from);
    } else if (root != nullptr) {
      // 这个分支我认为不会存在, 但是我具体也不知道.
      // 如果删除的是根节点且树不为空
      repairBalance(root);
    }

    return true;
  }

  // 向上回溯修复平衡
  void repairBalance(AVLNode* node) {
    while (node != nullptr) {
      node->updateHeight();
      // 平衡因子.
      long bf = node->getBalanceFactor();

      // 左重 (LL or LR)
      if (bf > 1) {
        // LL Case
        if (node->child[LEFT]->getBalanceFactor() >= 0) {
          node = rotate(node, RIGHT);
        }
        // LR Case
        else {
          rotate(node->child[LEFT], LEFT);
          node = rotate(node, RIGHT);
        }
      }
      // 右重 (RR or RL)
      else if (bf < -1) {
        // RR Case
        if (node->child[RIGHT]->getBalanceFactor() <= 0) {
          node = rotate(node, LEFT);
        }
        // RL Case
        else {
          rotate(node->child[RIGHT], RIGHT);
          node = rotate(node, LEFT);
        }
      }
      node = node->parent;  // 向上移动
    }
  }

  // 辅助函数：替换子树
  void transplant(AVLNode* old_node, AVLNode* new_node) {
    assert(old_node != nullptr);

    // 如果旧的节点是根节点, 那么应该替换根.
    if (old_node->parent == nullptr) {
      assert(old_node == root);
      root = new_node;
      return;
    }

    // 如果旧的节点不是根节点,
    // 那么应当连接旧节点的父亲和新节点.
    auto parent = old_node->parent;
    // 如果新节点不是空的,
    // 那么新节点的父指针指向旧节点的父亲.
    if (new_node != nullptr) {
      new_node->parent = parent;
    }

    // 将指向旧节点的孩子指针指向新节点.
    if (old_node == parent->child[LEFT]) {
      parent->child[LEFT] = new_node;
    } else {
      assert(old_node == parent->child[RIGHT]);
      parent->child[RIGHT] = new_node;
    }
  }
};
