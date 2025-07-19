/**
 * @file bplus_tree.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-07-19
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <algorithm>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

/**
 * @brief B+树的通用模板实现
 *
 * @tparam T 存储在树中的元素类型 (例如，一个结构体或 std::pair)。
 * @tparam Compare 一个比较函数对象 (Functor)，用于决定元素的排序。
 * 它必须提供 `bool operator()(const T& a, const T& b) const`，
 * 如果 a 在 b 前面，则返回 true。
 * @tparam M B+树的阶数，定义了一个节点可以拥有的最大子节点数。M 必须 >= 3。
 */
template <typename T, typename Compare, int M = 4>
class BPlusTree {
  static_assert(M >= 3, "B+ Tree order M must be at least 3");

 private:
  // 比较器实例，用于所有元素的比较
  Compare comp;

  // --- 节点结构定义 ---

  // B+树节点基类
  struct Node {
    bool is_leaf = false;
    Node* parent = nullptr;
    // 在内部节点中，elements 存储索引；在叶子节点中，存储实际数据。
    std::vector<T> elements;

    Node(bool leaf) : is_leaf(leaf) {}
    virtual ~Node() = default;

    // 使用比较器查找元素在节点中的位置（返回第一个不小于 element 的位置）
    int find_index(const T& element, const Compare& c) const {
      auto it = std::lower_bound(elements.begin(), elements.end(), element, c);
      return std::distance(elements.begin(), it);
    }

    // 检查节点是否已满
    bool is_full() const { return elements.size() == M - 1; }

    // 检查节点是否下溢 (用于删除操作)
    bool is_underflow() const {
      // M/2 的上取整是最小子节点数，所以 M/2 - 1 是最小键数
      return elements.size() < (M - 1) / 2;
    }
  };

  // 内部节点
  struct InternalNode : public Node {
    std::vector<Node*> children;

    InternalNode() : Node(false) {}
    ~InternalNode() override {
      for (Node* child : children) {
        delete child;
      }
    }
  };

  // 叶子节点
  struct LeafNode : public Node {
    LeafNode* next = nullptr;
    LeafNode* prev = nullptr;

    LeafNode() : Node(true) {}
    ~LeafNode() override = default;
  };

  Node* root = nullptr;

  // --- 私有辅助函数 ---

  /**
   * @brief 查找给定元素应该在的叶子节点。
   * @param element 一个包含用于查找的键的元素。
   * @return 指向目标叶子节点的指针。
   */
  LeafNode* find_leaf(const T& element) const {
    if (root == nullptr) return nullptr;
    Node* current = root;
    while (!current->is_leaf) {
      InternalNode* internal = static_cast<InternalNode*>(current);
      int index = internal->find_index(element, comp);
      if (index < internal->elements.size() &&
          !comp(element, internal->elements[index])) {
        current = internal->children[index + 1];
      } else {
        current = internal->children[index];
      }
    }
    return static_cast<LeafNode*>(current);
  }

  /**
   * @brief 在父节点中插入一个元素和子节点指针。这是节点分裂后调用的核心步骤。
   * @param left 分裂后的左子节点。
   * @param element 要插入到父节点的元素（作为索引）。
   * @param right 分裂后的右子节点。
   */
  void insert_in_parent(Node* left, const T& element, Node* right) {
    if (left == root) {
      InternalNode* new_root = new InternalNode();
      new_root->elements.push_back(element);
      new_root->children.push_back(left);
      new_root->children.push_back(right);
      root = new_root;
      left->parent = new_root;
      right->parent = new_root;
      return;
    }

    InternalNode* parent = static_cast<InternalNode*>(left->parent);
    int index = parent->find_index(element, comp);
    parent->elements.insert(parent->elements.begin() + index, element);
    parent->children.insert(parent->children.begin() + index + 1, right);
    right->parent = parent;

    if (parent->elements.size() >=
        M) {  // 内部节点键数量达到 M-1 是满，超过则分裂
      split_internal(parent);
    }
  }

  /**
   * @brief 当叶子节点满时，将其分裂成两个节点。
   * @param leaf 需要分裂的叶子节点。
   */
  void split_leaf(LeafNode* leaf) {
    LeafNode* new_leaf = new LeafNode();
    int mid_index = M / 2;

    new_leaf->elements.assign(leaf->elements.begin() + mid_index,
                              leaf->elements.end());
    leaf->elements.resize(mid_index);

    new_leaf->next = leaf->next;
    if (leaf->next) leaf->next->prev = new_leaf;
    leaf->next = new_leaf;
    new_leaf->prev = leaf;

    // 将新叶子的第一个元素复制到父节点作为索引
    insert_in_parent(leaf, new_leaf->elements.front(), new_leaf);
  }

  /**
   * @brief 当内部节点满时，将其分裂成两个节点。
   * @param node 需要分裂的内部节点。
   */
  void split_internal(InternalNode* node) {
    InternalNode* new_node = new InternalNode();
    int mid_index = (M - 1) / 2;
    T mid_element = node->elements[mid_index];

    new_node->elements.assign(node->elements.begin() + mid_index + 1,
                              node->elements.end());
    new_node->children.assign(node->children.begin() + mid_index + 1,
                              node->children.end());

    node->elements.resize(mid_index);
    node->children.resize(mid_index + 1);

    for (Node* child : new_node->children) child->parent = new_node;

    insert_in_parent(node, mid_element, new_node);
  }

  /**
   * @brief 处理节点下溢（键太少）。会尝试从兄弟节点借用或与兄弟节点合并。
   * @param node 发生下溢的节点。
   */
  void handle_underflow(Node* node) {
    if (node == root) {
      if (!root->is_leaf &&
          static_cast<InternalNode*>(root)->children.size() == 1) {
        Node* old_root = root;
        root = static_cast<InternalNode*>(root)->children[0];
        root->parent = nullptr;
        static_cast<InternalNode*>(old_root)->children.clear();
        delete old_root;
      }
      return;
    }

    InternalNode* parent = static_cast<InternalNode*>(node->parent);
    auto it = std::find(parent->children.begin(), parent->children.end(), node);
    int node_index = std::distance(parent->children.begin(), it);

    // 尝试从左兄弟借
    if (node_index > 0) {
      Node* left_sibling = parent->children[node_index - 1];
      if (left_sibling->elements.size() > (M - 1) / 2) {
        borrow_from_left(node, left_sibling, parent, node_index);
        return;
      }
    }

    // 尝试从右兄弟借
    if (node_index < parent->children.size() - 1) {
      Node* right_sibling = parent->children[node_index + 1];
      if (right_sibling->elements.size() > (M - 1) / 2) {
        borrow_from_right(node, right_sibling, parent, node_index);
        return;
      }
    }

    // 如果无法借，则合并
    if (node_index > 0) {
      merge_with_left(node, parent->children[node_index - 1], parent,
                      node_index);
    } else {
      merge_with_right(node, parent->children[node_index + 1], parent,
                       node_index);
    }
  }

  void borrow_from_left(Node* node, Node* left_sibling, InternalNode* parent,
                        int node_index) {
    if (node->is_leaf) {
      LeafNode* leaf = static_cast<LeafNode*>(node);
      LeafNode* left_leaf = static_cast<LeafNode*>(left_sibling);
      leaf->elements.insert(leaf->elements.begin(), left_leaf->elements.back());
      left_leaf->elements.pop_back();
      parent->elements[node_index - 1] = leaf->elements.front();
    } else {
      InternalNode* internal = static_cast<InternalNode*>(node);
      InternalNode* left_internal = static_cast<InternalNode*>(left_sibling);
      internal->elements.insert(internal->elements.begin(),
                                parent->elements[node_index - 1]);
      parent->elements[node_index - 1] = left_internal->elements.back();
      left_internal->elements.pop_back();
      internal->children.insert(internal->children.begin(),
                                left_internal->children.back());
      left_internal->children.back()->parent = internal;
      left_internal->children.pop_back();
    }
  }

  void borrow_from_right(Node* node, Node* right_sibling, InternalNode* parent,
                         int node_index) {
    if (node->is_leaf) {
      LeafNode* leaf = static_cast<LeafNode*>(node);
      LeafNode* right_leaf = static_cast<LeafNode*>(right_sibling);
      leaf->elements.push_back(right_leaf->elements.front());
      right_leaf->elements.erase(right_leaf->elements.begin());
      parent->elements[node_index] = right_leaf->elements.front();
    } else {
      InternalNode* internal = static_cast<InternalNode*>(node);
      InternalNode* right_internal = static_cast<InternalNode*>(right_sibling);
      internal->elements.push_back(parent->elements[node_index]);
      parent->elements[node_index] = right_internal->elements.front();
      right_internal->elements.erase(right_internal->elements.begin());
      internal->children.push_back(right_internal->children.front());
      right_internal->children.front()->parent = internal;
      right_internal->children.erase(right_internal->children.begin());
    }
  }

  void merge_with_left(Node* node, Node* left_sibling, InternalNode* parent,
                       int node_index) {
    if (node->is_leaf) {
      LeafNode* leaf = static_cast<LeafNode*>(node);
      LeafNode* left_leaf = static_cast<LeafNode*>(left_sibling);
      left_leaf->elements.insert(left_leaf->elements.end(),
                                 leaf->elements.begin(), leaf->elements.end());
      left_leaf->next = leaf->next;
      if (leaf->next) leaf->next->prev = left_leaf;
    } else {
      InternalNode* internal = static_cast<InternalNode*>(node);
      InternalNode* left_internal = static_cast<InternalNode*>(left_sibling);
      left_internal->elements.push_back(parent->elements[node_index - 1]);
      left_internal->elements.insert(left_internal->elements.end(),
                                     internal->elements.begin(),
                                     internal->elements.end());
      left_internal->children.insert(left_internal->children.end(),
                                     internal->children.begin(),
                                     internal->children.end());
      for (Node* child : internal->children) child->parent = left_internal;
      internal->children.clear();
    }
    parent->elements.erase(parent->elements.begin() + node_index - 1);
    parent->children.erase(parent->children.begin() + node_index);
    delete node;
    if (parent->is_underflow()) handle_underflow(parent);
  }

  void merge_with_right(Node* node, Node* right_sibling, InternalNode* parent,
                        int node_index) {
    merge_with_left(right_sibling, node, parent, node_index + 1);
  }

 public:
  BPlusTree() = default;
  ~BPlusTree() { delete root; }

  /**
   * @brief 搜索一个元素。
   * @param element 一个包含用于查找的键的元素。
   * @return 如果找到，返回指向树中完整元素的 const 指针；否则返回 nullptr。
   */
  const T* search(const T& element) const {
    LeafNode* leaf = find_leaf(element);
    if (leaf == nullptr) return nullptr;
    int index = leaf->find_index(element, comp);
    if (index < leaf->elements.size() &&
        !comp(leaf->elements[index], element) &&
        !comp(element, leaf->elements[index])) {
      return &leaf->elements[index];
    }
    return nullptr;
  }

  /**
   * @brief 向树中插入一个元素。如果键已存在，则更新它。
   * @param element 要插入的完整元素。
   */
  void insert(const T& element) {
    if (root == nullptr) root = new LeafNode();
    LeafNode* leaf = find_leaf(element);
    int index = leaf->find_index(element, comp);
    if (index < leaf->elements.size() &&
        !comp(leaf->elements[index], element) &&
        !comp(element, leaf->elements[index])) {
      leaf->elements[index] = element;
      return;
    }
    leaf->elements.insert(leaf->elements.begin() + index, element);
    if (leaf->elements.size() >= M) {
      split_leaf(leaf);
    }
  }

  /**
   * @brief 从树中删除一个元素。
   * @param element 一个包含用于查找的键的元素。
   */
  void remove(const T& element) {
    LeafNode* leaf = find_leaf(element);
    if (leaf == nullptr) return;
    int index = leaf->find_index(element, comp);
    if (index >= leaf->elements.size() ||
        comp(element, leaf->elements[index]) ||
        comp(leaf->elements[index], element)) {
      return;  // 元素不存在
    }
    leaf->elements.erase(leaf->elements.begin() + index);
    if (leaf->is_underflow()) {
      handle_underflow(leaf);
    }
  }

  /**
   * @brief 范围查找。查找 [start_element, end_element) 范围内的所有元素。
   * @param start_element 范围的起始（包含）。
   * @param end_element 范围的结束（不包含）。
   * @return 包含结果的 vector。
   */
  std::vector<T> search_range(const T& start_element,
                              const T& end_element) const {
    std::vector<T> result;
    LeafNode* leaf = find_leaf(start_element);
    if (leaf == nullptr) return result;
    int index = leaf->find_index(start_element, comp);
    while (leaf != nullptr) {
      for (size_t i = index; i < leaf->elements.size(); ++i) {
        if (!comp(leaf->elements[i], end_element)) return result;
        result.push_back(leaf->elements[i]);
      }
      leaf = leaf->next;
      index = 0;
    }
    return result;
  }

  /**
   * @brief 查找指定元素的前驱。
   * @param element 一个包含用于查找的键的元素。
   * @return 一个包含前驱元素的 optional，如果不存在则为空。
   */
  std::optional<T> find_predecessor(const T& element) const {
    LeafNode* leaf = find_leaf(element);
    if (!leaf) return std::nullopt;
    int index = leaf->find_index(element, comp);
    if (index > 0) return leaf->elements[index - 1];
    if (leaf->prev && !leaf->prev->elements.empty())
      return leaf->prev->elements.back();
    return std::nullopt;
  }

  /**
   * @brief 查找指定元素的后继。
   * @param element 一个包含用于查找的键的元素。
   * @return 一个包含后继元素的 optional，如果不存在则为空。
   */
  std::optional<T> find_successor(const T& element) const {
    LeafNode* leaf = find_leaf(element);
    if (!leaf) return std::nullopt;
    int index = leaf->find_index(element, comp);
    if (index < leaf->elements.size() &&
        !comp(leaf->elements[index], element) &&
        !comp(element, leaf->elements[index])) {
      index++;  // 如果找到完全匹配的，后继是下一个
    }
    if (index < leaf->elements.size()) return leaf->elements[index];
    if (leaf->next && !leaf->next->elements.empty())
      return leaf->next->elements.front();
    return std::nullopt;
  }

  /**
   * @brief 查找第一个不小于指定元素的元素 (等价于 std::lower_bound)。
   */
  std::optional<T> lower_bound(const T& element) const {
    LeafNode* leaf = find_leaf(element);
    if (!leaf) return std::nullopt;
    int index = leaf->find_index(element, comp);
    if (index < leaf->elements.size()) return leaf->elements[index];
    if (leaf->next && !leaf->next->elements.empty())
      return leaf->next->elements.front();
    return std::nullopt;
  }

  /**
   * @brief 查找第一个大于指定元素的元素 (等价于 std::upper_bound)。
   */
  std::optional<T> upper_bound(const T& element) const {
    LeafNode* leaf = find_leaf(element);
    if (!leaf) return std::nullopt;
    auto it = std::upper_bound(leaf->elements.begin(), leaf->elements.end(),
                               element, comp);
    int index = std::distance(leaf->elements.begin(), it);
    if (index < leaf->elements.size()) return leaf->elements[index];
    if (leaf->next && !leaf->next->elements.empty())
      return leaf->next->elements.front();
    return std::nullopt;
  }

  /**
   * @brief 按顺序遍历并打印所有元素。需要为类型 T 重载 << 操作符。
   */
  void traverse() const {
    if (root == nullptr) {
      std::cout << "Tree is empty." << std::endl;
      return;
    }
    Node* current = root;
    while (!current->is_leaf)
      current = static_cast<InternalNode*>(current)->children[0];
    LeafNode* leaf = static_cast<LeafNode*>(current);
    while (leaf != nullptr) {
      for (const auto& elem : leaf->elements) std::cout << elem << " ";
      leaf = leaf->next;
    }
    std::cout << std::endl;
  }
};

// --- 使用示例 ---

// 1. 定义我们想要存储的数据结构 (K-V 对)
struct UserRecord {
  int id;
  std::string name;
};

// 2. 为这个结构体定义一个比较器 (Functor)
//    这个比较器告诉B+树如何根据 id 来排序 UserRecord
struct CompareUserRecordById {
  bool operator()(const UserRecord& a, const UserRecord& b) const {
    return a.id < b.id;
  }
};

// 为了能方便地打印 UserRecord，我们重载 << 操作符
std::ostream& operator<<(std::ostream& os, const UserRecord& record) {
  os << "{" << record.id << ", " << record.name << "}";
  return os;
}

// 主函数，用于演示B+树的全部功能
int main() {
  std::cout << "--- B+树通用模板功能演示 (阶数 M=4) ---\n";
  BPlusTree<UserRecord, CompareUserRecordById, 4> user_db;

  std::cout << "\n--- 1. 插入操作 ---\n";
  user_db.insert({10, "Alice"});
  user_db.insert({20, "Bob"});
  user_db.insert({5, "Charlie"});
  user_db.insert({6, "David"});
  user_db.insert({12, "Eve"});
  user_db.insert({30, "Frank"});
  user_db.insert({7, "Grace"});
  user_db.insert({17, "Heidi"});
  std::cout << "初始数据: ";
  user_db.traverse();

  std::cout << "\n--- 2. 搜索操作 ---\n";
  const UserRecord* found = user_db.search({7, ""});
  if (found) std::cout << "找到 ID=7 的用户: " << found->name << std::endl;
  found = user_db.search({99, ""});
  if (!found) std::cout << "未找到 ID=99 的用户" << std::endl;

  std::cout << "\n--- 3. 范围查找 (ID从 7 到 20) ---\n";
  auto range_res = user_db.search_range({7, ""}, {20, ""});
  std::cout << "结果: ";
  for (const auto& r : range_res) std::cout << r << " ";
  std::cout << std::endl;

  std::cout << "\n--- 4. 前驱/后继/上下界查找 (以ID=12为例) ---\n";
  auto pred = user_db.find_predecessor({12, ""});
  if (pred) std::cout << "12 的前驱: " << *pred << std::endl;
  auto succ = user_db.find_successor({12, ""});
  if (succ) std::cout << "12 的后继: " << *succ << std::endl;
  auto lb = user_db.lower_bound({15, ""});
  if (lb) std::cout << "15 的最小上界: " << *lb << std::endl;
  auto ub = user_db.upper_bound({15, ""});
  if (ub) std::cout << "15 的最大下界: " << *ub << std::endl;

  std::cout << "\n--- 5. 删除操作 (测试合并与借用) ---\n";
  std::cout << "删除 6, 17, 7... (这将触发合并和借用)\n";
  user_db.remove({6, ""});
  user_db.remove({17, ""});
  user_db.remove({7, ""});
  std::cout << "删除后: ";
  user_db.traverse();

  std::cout << "\n--- 6. 清空树 ---\n";
  user_db.remove({5, ""});
  user_db.remove({10, ""});
  user_db.remove({12, ""});
  user_db.remove({20, ""});
  user_db.remove({30, ""});
  std::cout << "清空后: ";
  user_db.traverse();

  return 0;
}
