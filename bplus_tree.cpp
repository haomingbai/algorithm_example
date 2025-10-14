/**
 * @file bplus_tree.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-10-11
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

template <std::copyable Key, std::movable Value = Key, std::size_t T = 2,
          typename Compare = std::less<Key>>
struct BTree {
  static_assert(T >= 2, "BTree min degree T must be at least 2");
  // In order to avoid misunderstandings,
  // we don't use this consts unless when we are
  // declearing variables.
  static constexpr std::size_t kMinDegree = T;
  static constexpr std::size_t kMinElemNum = T - 1;
  static constexpr std::size_t kMaxDegree = 2 * T;
  static constexpr std::size_t kMaxElemNum = 2 * T - 1;

  Compare comp_ = Compare();

  bool Equal(const Key& k1, const Key& k2) const {
    return (!comp_(k1, k2)) && (!comp_(k2, k1));
  }

  struct BElem {
    Key key_;
    std::unique_ptr<Value> val_;
    std::size_t elem_cnt_ = 0;

    static BElem CreateBElem(Key key, Value val) {
      BElem elem;
      elem.key_ = key;
      elem.val_ = std::make_unique<Value>(std::move(val));
      elem.elem_cnt_ = 1;
      return elem;
    }

    static BElem CreateBElem(Key key) {
      BElem elem;
      elem.key_ = key;
      elem.val_ = nullptr;
      elem.elem_cnt_ = 0;
      return elem;
    }

    const Key& GetKey() const noexcept { return key_; }

    Value* GetValuePtr() const noexcept { return val_.get(); }

    std::size_t GetCount() const noexcept { return elem_cnt_; }

    void SetCount(std::size_t cnt = 1) noexcept { elem_cnt_ = cnt; }

    void IncreseCount(std::size_t cnt = 1) noexcept { elem_cnt_ += cnt; }

    void DecreaseCount(std::size_t cnt = 1) noexcept { elem_cnt_ -= cnt; }
  };

  struct BNode {
    std::array<BElem, kMaxElemNum + 1> elems_;
    std::array<std::unique_ptr<BNode>, kMaxDegree + 1> children_;
    std::size_t degree_;  // key_num + 1, same with leaf.
    std::size_t subtree_size_;
    BNode* next_;
    BTree* tree_;
    bool is_leaf_;

    // Factory of node ptr
    static std::unique_ptr<BNode> CreateBNode(BTree* tree, bool is_leaf) {
      auto node_ptr = std::make_unique<BNode>();
      node_ptr->tree_ = tree;
      node_ptr->is_leaf_ = is_leaf;
      node_ptr->next_ = nullptr;
      node_ptr->degree_ = 1;
      node_ptr->subtree_size_ = 0;
      return node_ptr;
    }

    // Is the node a leaf.
    bool IsLeaf() const noexcept { return is_leaf_; }

    bool IsEmpty() const noexcept { return degree_ <= 1; }

    // Get the element at the the index.
    const BElem& ElemAt(std::size_t index) const noexcept {
      assert(index + 1 < degree_);
      [[assume(index + 1 < degree_)]];
      return elems_[index];
    }

    // Get the elemtnt at the given index.
    BElem& ElemAt(std::size_t index) noexcept {
      assert(degree_ >= 1);
      assert(index + 1 < degree_);
      [[assume(degree_ >= 1)]];
      [[assume(index + 1 < degree_)]];
      return elems_[index];
    }

    std::size_t GetElemNum() const noexcept {
      assert(degree_ >= 1);
      [[assume(degree_ >= 1)]];
      return std::max<std::size_t>(degree_, 1) - 1;
    }

    std::size_t GetSubtreeSize() const noexcept { return subtree_size_; }

    bool IsFull() const noexcept { return degree_ == 2 * T; }

    bool IsAtMin() const noexcept { return degree_ == T; }

    void UpdateSubtreeSize() noexcept {
      std::size_t new_size = 0;
      // A leaf: the size of the current node.
      // Note: Since in B+ tree, the size of a tree is
      // only determined by the leaf node,
      // we should only include all sizes of all subtrees.
      if (IsLeaf()) {
        for (std::size_t i = 0; i < degree_ - 1; i++) {
          new_size += elems_[i].GetCount();
        }
      }
      // Not a leaf: The size of all subtrees.
      if (!IsLeaf()) {
        for (std::size_t i = 0; i < degree_; i++) {
          assert(children_[i] && "children_[i] must be non-null");
          new_size += children_[i]->GetSubtreeSize();
        }
      }
      subtree_size_ = new_size;
    }

    void SplitChild(std::size_t child_idx) {
      assert(!IsLeaf() &&
             "The leaf node cannot split its child since it has no child.");
      if (children_[child_idx]->IsLeaf()) {
        SplitLeafHelper(child_idx);
      } else {
        SplitInternalHelper(child_idx);
      }
    }

    void SplitLeafHelper(std::size_t child_idx) {
      // Pretest:
      // The child should be full.
      assert(children_[child_idx]->IsFull() && "The child should be full");
      // Preemptive split: the parent should not be full.
      assert(!IsFull() && "The parent should not be full");
      // Split child and uplift the element
      // at the middle of the child node.
      // Step 1:
      // Create a new node.
      auto left_child = children_[child_idx].get();
      auto right_child = CreateBNode(left_child->tree_, left_child->IsLeaf());
      // Step 2:
      // Move the elements at the right side of the left_child
      // to the left side of the right child.
      // Before:
      // [0, 1, ..., T - 2, T - 1, T, ..., 2 * T - 2]
      // After:
      // [0, 1, ..., T - 2], [0, 1, ..., T - 1]
      {
        constexpr std::size_t src = T - 1;
        constexpr std::size_t count = T;
        std::move(left_child->elems_.begin() + static_cast<std::ptrdiff_t>(src),
                  left_child->elems_.begin() +
                      static_cast<std::ptrdiff_t>(src + count),
                  right_child->elems_.begin());
      }
      // 2 * T - 1 -> T, T + 1
      left_child->degree_ = T;
      right_child->degree_ = T + 1;
      // clang-format off
      // Step 3:
      // Make room for the new index uplifted.
      // Before (the uppper is element, the lower one is children):
      // [0, 1, ..., child_idx - 1, child_idx, ..., degree_ - 2]
      // [0, 1, ..., child_idx, child_idx + 1, ..., degree_ - 1]
      // After:
      // [0, 1, ..., child_idx - 1, child_idx (empty), child_idx + 1, ..., degree_ - 1]
      // [0, 1, ..., child_idx, child_idx + 1 (empty), child_idx + 2, ..., degree_]
      // clang-format on
      {
        assert(degree_ >= 1);
        // Make sure there is space for move
        assert(!IsFull());
        // Move the child backward.
        auto child_first =
            children_.begin() + static_cast<std::ptrdiff_t>(child_idx + 1);
        auto child_last =
            children_.begin() + static_cast<std::ptrdiff_t>(degree_);
        std::move_backward(child_first, child_last, child_last + 1);
        // Move the elements backward.
        auto elem_first =
            elems_.begin() + static_cast<std::ptrdiff_t>(child_idx);
        auto elem_last =
            elems_.begin() + static_cast<std::ptrdiff_t>(degree_ - 1);
        std::move_backward(elem_first, elem_last, elem_last + 1);
      }
      // Step 4:
      // Insert the new element and children.
      auto new_elem_key = right_child->elems_[0].GetKey();
      elems_[child_idx] = BElem::CreateBElem(new_elem_key);
      children_[child_idx + 1] = std::move(right_child);
      // One more key and child.
      degree_++;
      // Step 5:
      // Maintain the next ptr of the children.
      children_[child_idx + 1]->next_ = left_child->next_;
      children_[child_idx]->next_ = children_[child_idx + 1].get();
      // Step 6:
      // Update the tree size of the subtrees.
      // Since the subtree size of parent is unchanged,
      // the parent node needn't update the size.
      children_[child_idx]->UpdateSubtreeSize();
      children_[child_idx + 1]->UpdateSubtreeSize();
      return;
    }

    void SplitInternalHelper(std::size_t child_idx) {
      // Preemptive split: the parent should not be full.
      assert(!IsFull() && "Preemptive split: the parent should not be full.");
      // Pretest: The child node is full.
      assert(children_[child_idx]->IsFull() && "The child node should be full");
      // Split the child and uplift the element
      // at the middle point of the child node.
      // Step 1: Get the pointer of the two nodes.
      auto left_child = children_[child_idx].get();
      auto right_child =
          BNode::CreateBNode(left_child->tree_, left_child->IsLeaf());
      // Step 2:
      // Move the elements of the child to the new node.
      // 2 * T - 1 -> T - 1, 1, T - 1
      // clang-format off
      // Before:
      // [0, 1, ..., T - 2, T - 1, T, T + 1, ..., 2 * T - 2]
      // After:
      // [0, 1, ..., T - 2], [(uplifted)], [0, 1, ..., T - 2]
      // clang-format on
      {
        constexpr std::size_t src = T;
        constexpr std::size_t move_count = T - 1;
        std::move(left_child->elems_.begin() + static_cast<std::ptrdiff_t>(src),
                  left_child->elems_.begin() +
                      static_cast<std::ptrdiff_t>(src + move_count),
                  right_child->elems_.begin());
      }
      // Step 3:
      // Move the children of the left child to the right child.
      // Before:
      // [0, 1, ..., T - 1, T, T + 1, ..., 2 * T - 1]
      // After:
      // [0, 1, ..., T - 1], [0, 1, ..., T - 1]
      {
        constexpr std::size_t move_count = T;
        const auto src = static_cast<std::size_t>(T);
        std::move(left_child->children_.begin() + src,
                  left_child->children_.begin() + src + move_count,
                  right_child->children_.begin());
      }
      // 2 * T - 1 -> T - 1, 1, T - 1
      left_child->degree_ = T;
      right_child->degree_ = T;
      // clang-format off
      // Step 4:
      // Make room for the new index uplifted.
      // Before (the uppper is element, the lower one is children):
      // [0, 1, ..., child_idx - 1, child_idx, ..., degree_ - 2]
      // [0, 1, ..., child_idx, child_idx + 1, ..., degree_ - 1]
      // After:
      // [0, 1, ..., child_idx - 1, child_idx (empty), child_idx + 1, ..., degree_ - 1]
      // [0, 1, ..., child_idx, child_idx + 1 (empty), child_idx + 2, ..., degree_]
      // clang-format on
      {
        assert(degree_ >= 1);
        assert(!IsFull());
        // Move the child backward.
        auto child_first =
            children_.begin() + static_cast<std::ptrdiff_t>(child_idx + 1);
        auto child_last =
            children_.begin() + static_cast<std::ptrdiff_t>(degree_);
        std::move_backward(child_first, child_last, child_last + 1);
        // Move the elements backward.
        auto elem_first =
            elems_.begin() + static_cast<std::ptrdiff_t>(child_idx);
        auto elem_last =
            elems_.begin() + static_cast<std::ptrdiff_t>(degree_ - 1);
        std::move_backward(elem_first, elem_last, elem_last + 1);
      }
      // Step 5:
      // Uplift the element at the index of T - 1.
      elems_[child_idx] = std::move(left_child->elems_[T - 1]);
      children_[child_idx + 1] = std::move(right_child);
      // One more key and child.
      degree_++;
      // Step 6:
      children_[child_idx]->UpdateSubtreeSize();
      children_[child_idx + 1]->UpdateSubtreeSize();
      return;
    }

    static std::unique_ptr<BNode> SplitRoot(std::unique_ptr<BNode> old_root) {
      assert(old_root->IsFull());
      auto new_root = CreateBNode(old_root->tree_, false);
      new_root->degree_ = 1;
      new_root->children_[0] = std::move(old_root);
      new_root->SplitChild(0);
      return new_root;
    }

    // Find the child index to descend for `key`.
    // Returns i in [0, GetElemNum()] such that the key belongs to children_[i].
    // Equivalent to: first i where !(elems_[i].key_ < key).
    std::size_t FindChildIndex(const Key& key) const noexcept {
      // number of keys in this node
      const std::size_t n = GetElemNum();
      std::size_t i = 0;
      // advance while elems_[i].key_ < key
      while (i < n && !tree_->comp_(key, elems_[i].GetKey())) {
        i++;
      }
      return i;
    }

    // Find the position inside this node (leaf or internal) to insert `key`.
    // Returns j in [0, GetElemNum()] where the new key should be placed so that
    // the elements remain sorted (stable with respect to existing equal keys:
    // insert after all strictly smaller keys).
    std::size_t FindElemPos(const Key& key) const noexcept {
      const std::size_t n = GetElemNum();
      std::size_t j = 0;
      while (j < n && tree_->comp_(elems_[j].GetKey(), key)) {
        j++;
      }
      return j;
    }

    // Insert a new element into this node at position `pos`, but if the key
    // already exists at that position (equal keys), increase the element count
    // and discard the provided `value`. This variant updates subtree_size_
    // appropriately and is intended for use when the tree must maintain
    // accurate subtree sizes.
    // Preconditions:
    //   pos <= GetElemNum().
    //   node must have capacity for one more element (caller must ensure not
    //   full).
    // Notes:
    // - For B+ tree usage, duplicate keys should only appear in leaves. We
    // assert
    //   that equality handling happens on leaves.
    // - After insertion or count-increase, this node's subtree_size_ is
    // updated.
    void InsertOneElemAt(Key key, Value value, std::size_t pos) noexcept {
      const std::size_t cur_keys = GetElemNum();
      assert(pos <= cur_keys);
      assert(IsLeaf());
      assert(!IsFull());
      // If pos points to an existing key equal to `key`, then we treat this as
      // duplicate insertion: increase the count and discard `value`.
      if (pos < cur_keys && tree_->Equal(elems_[pos].GetKey(), key)) {
        // In B+ tree semantics duplicate keys (counts) belong to leaves.
        // If this assertion fails, caller is misusing the API.
        assert(IsLeaf() && "Duplicate key encountered in non-leaf node");
        elems_[pos].IncreseCount(1);
        // Update this node's subtree_size_ to reflect the increased count.
        // (For a leaf this sums its counts; UpdateSubtreeSize is safe and
        // simple.)
        UpdateSubtreeSize();
        return;
      }
      // Otherwise we need to insert a fresh element at `pos`.
      // Shift elements [pos, cur_keys) right by one slot to make room.
      auto elem_first = elems_.begin() + static_cast<std::ptrdiff_t>(pos);
      auto elem_last = elems_.begin() + static_cast<std::ptrdiff_t>(cur_keys);
      std::move_backward(elem_first, elem_last, elem_last + 1);
      // Place the new element (leaf nodes store values; internal callers may
      // pass a dummy value if appropriate).
      elems_[pos] = BElem::CreateBElem(std::move(key), std::move(value));
      // One more key => degree_ increases by 1 (degree_ == key_num + 1).
      degree_ += 1;
      // Update subtree size for this node. For leaves this increases by 1;
      // for internal nodes UpdateSubtreeSize will recompute from children.
      UpdateSubtreeSize();
    }

    // Get the child's raw pointer at the given index.
    BNode* ChildAt(std::size_t index) {
      assert(index < degree_);
      return children_[index].get();
    }

    void MergeChild(std::size_t left_child_idx) {
      // Merge two child nodes at the min position.
      assert(!IsLeaf() &&
             "The leaf node cannot merge its child since it has no child");
      if (children_[left_child_idx]->IsLeaf()) {
        MergeLeafHelper(left_child_idx);
      } else {
        MergeInternalHelper(left_child_idx);
      }
    }

    // Make sure to merge node only when
    // borrow fails.
    bool TryMergeChild(std::size_t child_idx) {
      assert(children_[child_idx]->IsAtMin());
      if (child_idx == degree_ - 1) {
        if (!children_[child_idx - 1]->IsAtMin()) {
          return false;
        } else {
          MergeChild(child_idx - 1);
          return true;
        }
      } else {
        if (!children_[child_idx + 1]->IsAtMin()) {
          return false;
        } else {
          MergeChild(child_idx);
          return true;
        }
      }
    }

    void MergeLeafHelper(std::size_t left_child_idx) {
      // Pretest: The leaves to be merged should be at min.
      auto right_child_idx = left_child_idx + 1;
      assert(children_[left_child_idx]->IsAtMin());
      assert(children_[right_child_idx]->IsAtMin());
      auto left_child = ChildAt(left_child_idx);
      auto right_child = ChildAt(right_child_idx);
      // As we can see, when we apply the preemptive merge,
      // both children should be at min.
      // T - 1 + T - 1 -> 2 * T - 2
      // clang-format off
      // Step 1:
      // Move the elements at the right child to the left one.
      // Before:
      // [0, 1, ..., T - 2], [0, 1, ..., T - 2]
      // After:
      // [0, 1, ..., T - 2, T - 1, T, ..., 2 * T - 3]
      // clang-format on
      {
        std::size_t dest = T - 1;
        std::size_t move_count = T - 1;
        std::move(right_child->elems_.begin(),
                  right_child->elems_.begin() + move_count,
                  left_child->elems_.begin() + dest);
        // ElemNum = 2T - 2 -> Degree = 2T - 1
        left_child->degree_ = 2 * T - 1;
      }
      // Step 2:
      // Reset the next pointer of the left child.
      left_child->next_ = right_child->next_;
      // clang-format off
      // Step 3:
      // Move the elements at or after right_child_idx
      // and children after right_child_idx a step forward.
      // Before:
      // [0, 1, ..., left_child_idx, right_child_idx, ..., degree_ - 2]
      // [0, 1, ..., right_child_idx, right_child_idx + 1, degree_ - 1]
      // After:
      // [0, 1, ..., left_child_idx, ..., degree_ - 3]
      // [0, 1, ..., right_child_idx, ..., degree_ - 2]
      // clang-format on
      {
        assert(right_child_idx < degree_);
        auto move_count = degree_ - 1 - right_child_idx;
        auto elem_move_dest = elems_.begin() + left_child_idx;
        auto elem_move_begin = elems_.begin() + right_child_idx;
        std::move(elem_move_begin, elem_move_begin + move_count,
                  elem_move_dest);
        auto children_move_dest = children_.begin() + right_child_idx;
        auto children_move_begin = children_.begin() + right_child_idx + 1;
        std::move(children_move_begin, children_move_begin + move_count,
                  children_move_dest);
        // The number of element and children decrease by one.
        assert(degree_ > 1 && "parent degree_ must be > 1 before decrement");
        degree_--;
      }
      // Step 4:
      // Update the subtree size of the left child.
      // Note: Don't update the size of right child
      // since it is invalid now!
      left_child->UpdateSubtreeSize();
      return;
    }

    void MergeInternalHelper(std::size_t left_child_idx) {
      // Pretest: Both leaves should be at min.
      auto right_child_idx = left_child_idx + 1;
      auto left_child = ChildAt(left_child_idx);
      auto right_child = ChildAt(right_child_idx);
      assert(left_child->IsAtMin());
      assert(right_child->IsAtMin());
      auto separating_key_from_parent = std::move(elems_[left_child_idx]);
      // clang-format off
      // Step 1:
      // Move the elements on the right child to the left child.
      // Before:
      // [0, 1, ..., T - 2], [], [0, 1, ..., T - 2]
      // After:
      // [0, 1, ..., T - 2, T - 1, T, T + 1, ..., 2 * T - 2]
      // clang-format on
      {
        auto move_count = T - 1;
        auto move_dest = left_child->elems_.begin() + T;
        auto move_begin = right_child->elems_.begin();
        std::move(move_begin, move_begin + move_count, move_dest);
      }
      // clang-format off
      // Step 2:
      // Move the children on the right child to the left child.
      // Before:
      // [0, 1, ..., T - 1], [0, 1, ..., T - 1]
      // After:
      // [0, 1, ..., T - 1, T, T + 1, ..., 2 * T - 1]
      // clang-format on
      {
        auto move_count = T;
        auto move_dest = left_child->children_.begin() + T;
        auto move_begin = right_child->children_.begin();
        std::move(move_begin, move_begin + move_count, move_dest);
        left_child->degree_ = 2 * T;
      }
      // Step 3:
      // Move the node at parent[left_child_idx]
      // to the child node.
      left_child->elems_[T - 1] = std::move(separating_key_from_parent);
      // clang-format off
      // Step 4:
      // Move the elements at or after right_child_idx
      // and children after right_child_idx a step forward.
      // Before:
      // [0, 1, ..., left_child_idx, right_child_idx, ..., degree_ - 2]
      // [0, 1, ..., right_child_idx, right_child_idx + 1, degree_ - 1]
      // After:
      // [0, 1, ..., left_child_idx, ..., degree_ - 3]
      // [0, 1, ..., right_child_idx, ..., degree_ - 2]
      // clang-format on
      {
        assert(right_child_idx < degree_);
        auto move_count = degree_ - 1 - right_child_idx;
        auto elem_move_dest = elems_.begin() + left_child_idx;
        auto elem_move_begin = elems_.begin() + right_child_idx;
        std::move(elem_move_begin, elem_move_begin + move_count,
                  elem_move_dest);
        auto children_move_dest = children_.begin() + right_child_idx;
        auto children_move_begin = children_.begin() + right_child_idx + 1;
        std::move(children_move_begin, children_move_begin + move_count,
                  children_move_dest);
        // The number of element and children decrease by one.
        assert(degree_ > 1 && "parent degree_ must be > 1 before decrement");
        degree_--;
      }
      // Step 5:
      // Update the subtree size of the left child.
      left_child->UpdateSubtreeSize();
      return;
    }

    static std::unique_ptr<BNode> MergeRoot(std::unique_ptr<BNode> old_root) {
      assert(old_root->degree_ == 2);
      old_root->MergeChild(0);
      auto new_root = std::move(old_root->children_[0]);
      return new_root;
    }

    bool TryBorrow(std::size_t child_idx) {
      return TryBorrowLeft(child_idx) || TryBorrowRight(child_idx);
    }

    bool TryBorrowLeft(std::size_t child_idx) {
      // Pretest: children_[child_idx] should be at min.
      assert(children_[child_idx]->IsAtMin());
      // For the sake of safety and simplify the logic
      // of TryBorrow, we add a check here.
      if (child_idx == 0) {
        return false;
      }
      // If the left child cannot borrow extra nodes.
      if (children_[child_idx - 1]->IsAtMin()) {
        return false;
      }
      auto left_child = children_[child_idx - 1].get();
      auto right_child = children_[child_idx].get();
      // Make space for the new node.
      {
        auto elem_src_begin = right_child->elems_.begin();
        auto elem_move_cnt = right_child->GetElemNum();
        auto elem_dest_end = right_child->elems_.begin() + elem_move_cnt + 1;
        std::move_backward(elem_src_begin, elem_src_begin + elem_move_cnt,
                           elem_dest_end);
        // For un-leaf nodes, the children should also be moved.
        if (!right_child->IsLeaf()) {
          auto child_src_begin = right_child->children_.begin();
          auto child_move_cnt = right_child->degree_;
          auto child_dest_end =
              right_child->children_.begin() + child_move_cnt + 1;
          std::move_backward(child_src_begin, child_src_begin + child_move_cnt,
                             child_dest_end);
        }
        right_child->degree_++;
      }
      // Move the last element of the left_child to
      // the beginning of the right_child.
      if (left_child->IsLeaf()) {
        right_child->elems_[0] =
            std::move(left_child->elems_[left_child->GetElemNum() - 1]);
        assert(degree_ > 1 && "degree_ must be > 1 before decrement");
        // Update the index value.
        // Make a new index element.
        auto new_node = BElem::CreateBElem(right_child->elems_[0].GetKey());
        elems_[child_idx - 1] = std::move(new_node);
      } else {
        right_child->elems_[0] = std::move(elems_[child_idx - 1]);
        elems_[child_idx - 1] =
            std::move(left_child->elems_[left_child->GetElemNum() - 1]);
        right_child->children_[0] =
            std::move(left_child->children_[left_child->degree_ - 1]);
      }
      assert(left_child->degree_ > T);
      left_child->degree_--;
      // Update the size of the subtree of both nodes.
      left_child->UpdateSubtreeSize();
      right_child->UpdateSubtreeSize();
      // Move the new element to the coresponding position.
      return true;
    }

    bool TryBorrowRight(std::size_t child_idx) {
      // Pretest: children_[child_idx] should be at min.
      assert(children_[child_idx]->IsAtMin());
      // If we are the rightmost child or the right sibling is at min, we can't
      // borrow.
      if (child_idx == degree_ - 1 || children_[child_idx + 1]->IsAtMin()) {
        return false;
      }
      auto left_child = children_[child_idx].get();
      auto right_child = children_[child_idx + 1].get();
      if (right_child->IsLeaf()) {
        // --- Leaf Borrowing ---
        // Move the first element of the right leaf to the end of the left leaf.
        left_child->elems_[left_child->GetElemNum()] =
            std::move(right_child->elems_[0]);
        left_child->degree_++;
        // Shift all elements in the right child one step to the left.
        auto elem_dest_begin = right_child->elems_.begin();
        auto elem_src_begin = right_child->elems_.begin() + 1;
        auto elem_move_count = right_child->GetElemNum() - 1;
        std::move(elem_src_begin, elem_src_begin + elem_move_count,
                  elem_dest_begin);
        right_child->degree_--;
        // CRITICAL: Update the parent's separator key to match the new first
        // key of the right child. This logic belongs ONLY in the leaf case.
        elems_[child_idx] = BElem::CreateBElem(right_child->elems_[0].GetKey());
      } else {
        // --- Internal Node Borrowing ---
        // The parent's separator key moves down to the end of the left child.
        left_child->elems_[left_child->GetElemNum()] =
            std::move(elems_[child_idx]);
        // The leftmost child pointer of the right node moves to the end of the
        // left node.
        left_child->children_[left_child->degree_] =
            std::move(right_child->children_[0]);
        left_child->degree_++;
        // The leftmost key of the right node moves up to be the new parent
        // separator.
        elems_[child_idx] = std::move(right_child->elems_[0]);
        // Shift all elements in the right child one step to the left.
        auto elem_dest_begin = right_child->elems_.begin();
        auto elem_src_begin = right_child->elems_.begin() + 1;
        auto elem_move_count = right_child->GetElemNum() - 1;
        std::move(elem_src_begin, elem_src_begin + elem_move_count,
                  elem_dest_begin);
        // Shift all children in the right child one step to the left.
        auto child_dest_begin = right_child->children_.begin();
        auto child_src_begin = right_child->children_.begin() + 1;
        auto child_move_count = right_child->degree_ - 1;
        std::move(child_src_begin, child_src_begin + child_move_count,
                  child_dest_begin);
        right_child->degree_--;
      }
      // Update subtree sizes for both modified children.
      left_child->UpdateSubtreeSize();
      right_child->UpdateSubtreeSize();
      return true;
    }

    void RemoveOneElemAt(std::size_t pos) {
      const std::size_t cur_keys = GetElemNum();
      assert(pos < cur_keys);
      assert(IsLeaf());
      // If the pos points to a node with count
      // greater than 1, then we just decrease the count
      // of this element.
      if (elems_[pos].GetCount() > 1) {
        elems_[pos].elem_cnt_--;
        UpdateSubtreeSize();
        return;
      }
      // Remove the element by moving the elements backward
      // a step forward.
      auto src_begin = elems_.begin() + pos + 1;
      auto src_end = elems_.begin() + degree_ - 1;
      auto dest_begin = elems_.begin() + pos;
      std::move(src_begin, src_end, dest_begin);
      assert(degree_ > 1 && "parent degree_ must be > 1 before decrement");
      degree_--;
      UpdateSubtreeSize();
      return;
    }
  };

  struct BNodeIndexStack {
    static constexpr std::size_t kMaxCapacity = 64;
    // Each entry: (parent node pointer, the child index chosen at that parent
    // when descending)
    std::array<std::pair<BNode*, std::size_t>, kMaxCapacity> data_;
    std::size_t size_ = 0;

    bool Push(BNode* node, std::size_t child_index) {
      if (size_ < kMaxCapacity) {
        data_[size_++] = {node, child_index};
        return true;
      }
      return false;
    }

    void Pop() {
      assert(size_ > 0);
      size_--;
    }

    void Clear() { size_ = 0; }

    bool Empty() const { return size_ == 0; }

    std::size_t Size() const { return size_; }

    // Return top pair (parent, child_index). Undefined if stack empty.
    std::pair<BNode*, std::size_t> Top() const {
      assert(size_ > 0);
      return data_[size_ - 1];
    }

    // Return entry at given depth from top: 0 = top, 1 = one below top, ...
    std::pair<BNode*, std::size_t> AtFromTop(std::size_t from_top) const {
      assert(from_top < size_);
      return data_[size_ - 1 - from_top];
    }
  };

  std::unique_ptr<BNode> root_;

  BTree() : root_(BNode::CreateBNode(this, true)) {}

  // This method works when the key can fully reflect
  // the value.
  // When we find the same key at the leaf, we will
  // ignore the value given.
  // Under this kind of insertation will
  // the element count work.
  // Note: Never use this method with those not maintaining
  // the element count of the elements.
  void InsertOne(Key key, Value value) {
    // Detect whether the root needs spliting.
    if (root_->IsFull()) {
      auto new_root = BNode::SplitRoot(std::move(root_));
      root_ = std::move(new_root);
    }
    // Find the leaf node from parent.
    auto curr_node = root_.get();
    static thread_local BNodeIndexStack path;
    path.Clear();
    while (!curr_node->IsLeaf()) {
      auto child_idx = curr_node->FindChildIndex(key);
      auto child = curr_node->ChildAt(child_idx);
      // Preemptive split.
      if (child->IsFull()) {
        curr_node->SplitChild(child_idx);
      }
      // Find the index again.
      child_idx = curr_node->FindChildIndex(key);
      path.Push(curr_node, child_idx);
      curr_node = curr_node->ChildAt(child_idx);
    }
    auto pos_to_insert = curr_node->FindElemPos(key);
    curr_node->InsertOneElemAt(key, std::move(value), pos_to_insert);
    // Trace back to update the subtree size.
    while (!path.Empty()) {
      auto node_pair = path.Top();
      node_pair.first->UpdateSubtreeSize();
      path.Pop();
    }
    return;
  }

  // Under this kind of insertation will
  // the element count work.
  // Note: Never use this method with those not maintaining
  // the element count of the elements.
  void InsertOne(Key key)
    requires std::same_as<Key, Value>
  {
    InsertOne(key, key);
  }

  // Count: return the occurrence count of `key` (0 if not found).
  std::size_t Count(Key key) const noexcept {
    const BNode* curr = root_.get();
    while (!curr->IsLeaf()) {
      std::size_t child_idx = curr->FindChildIndex(key);
      // descend to child
      curr = curr->children_[child_idx].get();
    }
    // curr is leaf
    const std::size_t n = curr->GetElemNum();
    std::size_t pos = curr->FindElemPos(key);
    if (pos < n && Equal(curr->ElemAt(pos).GetKey(), key)) {
      return curr->ElemAt(pos).GetCount();
    }
    return 0;
  }

  void RemoveOne(Key key) {
    if (root_->IsLeaf() && root_->GetElemNum() == 0) {
      return;
    }
    if (!root_->IsLeaf() && root_->GetElemNum() == 1 &&
        root_->ChildAt(0)->IsAtMin() && root_->ChildAt(1)->IsAtMin()) {
      auto new_root = BNode::MergeRoot(std::move(root_));
      root_ = std::move(new_root);
    }
    static thread_local BNodeIndexStack path;
    path.Clear();
    // Find the leaf node.
    auto curr_node = root_.get();
    while (!curr_node->IsLeaf()) {
      auto child_idx = curr_node->FindChildIndex(key);
      if (curr_node->ChildAt(child_idx)->IsAtMin()) {
        if (!curr_node->TryBorrow(child_idx)) {
          auto merge_result = curr_node->TryMergeChild(child_idx);
          assert(merge_result);
        }
        child_idx = curr_node->FindChildIndex(key);
      }
      path.Push(curr_node, child_idx);
      assert(!curr_node->ChildAt(child_idx)->IsAtMin());
      curr_node = curr_node->ChildAt(child_idx);
    }
    auto elem_idx = curr_node->FindElemPos(key);
    if (elem_idx < curr_node->GetElemNum() &&
        Equal(curr_node->ElemAt(elem_idx).GetKey(), key)) {
      curr_node->RemoveOneElemAt(elem_idx);
    }
    while (!path.Empty()) {
      auto node = path.Top();
      node.first->UpdateSubtreeSize();
      path.Pop();
    }
    return;
  }
};
