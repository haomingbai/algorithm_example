#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __GNUC__
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

#define CONTAINER_OF(NODE_TYPE, MEMBER, NODE_PTR) \
  ((NODE_TYPE *)((char *)(NODE_PTR) - (offsetof(NODE_TYPE, MEMBER))))

typedef struct binary_node binary_node;
typedef binary_node binary_node;

enum direction { LEFT = 0, RIGHT = 1 };
enum color { RED = 0, BLACK = 1 };

struct binary_node {
  binary_node *parent;
  binary_node *child[2];
  enum color color;
};

enum traverse_order { PRE, IN, POST };

typedef struct binary_tree {
  binary_node *root;
  size_t size;
} binary_tree;

typedef binary_tree bTree;

int init_binary_tree(binary_tree *tree) {
  tree->size = 0, tree->root = NULL;
  return 0;
}

binary_tree *make_binary_tree() {
  bTree *tree = (bTree *)malloc(sizeof(bTree));
#ifdef __GNUC__
  if (unlikely(!tree)) {
#else
  if (!tree) {
#endif
    return NULL;
  }
  init_binary_tree(tree);
  return tree;
}

// Notice: this function will never update the size of the tree for it is not
// efficient enough to be integrated and in different cases there exists many of
// different occasions.
// Notice: make sure that the node orphan node.
binary_node *binary_tree_insert(binary_tree *tree, binary_node *node,
                                binary_node *new_node, enum direction dir) {
#ifdef __GNUC__
  if (unlikely(tree == NULL || new_node == NULL)) {
#else
  if (tree == NULL || new_node == NULL) {
#endif
    return NULL;
#ifdef __GNUC__
  } else if (unlikely(node == NULL)) {
#else
  } else if (node == NULL) {
#endif
    binary_node *old_node = tree->root;
    tree->root = new_node;
    new_node->parent = NULL, new_node->child[LEFT] = NULL,
    new_node->child[RIGHT] = NULL;
    return old_node;
  } else {
    binary_node *old_node = node->child[dir];
    node->child[dir] = new_node;
    new_node->child[LEFT] = NULL;
    new_node->child[RIGHT] = NULL;
    new_node->parent = node;
    return old_node;
  }
}

binary_node *binary_tree_transplant(binary_tree *tree, binary_node *old_node,
                                    binary_node *new_node) {
  binary_node *parent = old_node->parent;
  if (new_node != NULL) {
    new_node->parent = parent;
  }
  if (parent == NULL) {
    assert(tree->root == old_node);
    tree->root = new_node;
  } else {
    assert(tree->root != old_node);
    if (parent->child[LEFT] == old_node) {
      parent->child[LEFT] = new_node;
    } else if (parent->child[RIGHT] == old_node) {
      parent->child[RIGHT] = new_node;
    } else {
      assert(0);
    }
  }
  return old_node;
}

void *__alloc_with_offset(size_t size, size_t offset) {
  void *ptr = malloc(size);
#ifdef __GNUC__
  if (likely(ptr)) {
#else
  if (ptr) {
#endif
    return (void *)((char *)ptr + offset);
  } else {
    return NULL;
  }
}

#define MAKE_BINARY_NODE(NODE_TYPE, MEMBER)              \
  ((binary_node *)__alloc_with_offset(sizeof(NODE_TYPE), \
                                      offsetof(NODE_TYPE, MEMBER)))

#define FREE_BINARY_NODE(NODE_TYPE, MEMBER, NODE_PTR) \
  (free(CONTAINER_OF(NODE_TYPE, MEMBER, NODE_PTR)))

void binary_node_traverse(binary_node *node, void (*func)(binary_node *),
                          enum traverse_order order) {
  switch (order) {
    case PRE:
      func(node);
      if (node->child[LEFT] != NULL) {
        binary_node_traverse(node->child[LEFT], func, order);
      }
      if (node->child[RIGHT] != NULL) {
        binary_node_traverse(node->child[RIGHT], func, order);
      }
      break;
    case POST:
      if (node->child[LEFT] != NULL) {
        binary_node_traverse(node->child[LEFT], func, order);
      }
      if (node->child[RIGHT] != NULL) {
        binary_node_traverse(node->child[RIGHT], func, order);
      }
      func(node);
      break;
    case IN:
      if (node->child[LEFT] != NULL) {
        binary_node_traverse(node->child[LEFT], func, order);
      }
      func(node);
      if (node->child[RIGHT] != NULL) {
        binary_node_traverse(node->child[RIGHT], func, order);
      }
      break;
  }
}

void binary_tree_traverse(binary_tree *tree, void (*func)(binary_node *),
                          enum traverse_order order) {
  if (tree->root == NULL) {
    return;
  } else {
    binary_node_traverse(tree->root, func, order);
  }
}

void binary_node_switch_order(binary_node *node) {
  binary_node *tmp = node->child[LEFT];
  node->child[LEFT] = node->child[RIGHT];
  node->child[RIGHT] = tmp;
}

void binary_tree_reverse(binary_tree *tree) {
  binary_tree_traverse(tree, binary_node_switch_order, POST);
}

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

size_t binary_node_height(binary_node *node) {
  if (node == NULL) {
    return 0;
  } else {
    return MAX(binary_node_height(node->child[LEFT]),
               binary_node_height(node->child[RIGHT])) +
           1;
  }
}

size_t binary_tree_height(binary_tree *tree) {
  size_t res = binary_node_height(tree->root);
  return res;
}

size_t binary_tree_get_size(binary_tree *tree) { return tree->size; }

size_t binary_node_get_size(binary_node *root) {
  if (root == NULL) {
    return 0;
  }
  return 1 + binary_node_get_size(root->child[LEFT]) +
         binary_node_get_size(root->child[RIGHT]);
}

size_t binary_tree_update_size(binary_tree *tree) {
  size_t res = binary_node_get_size(tree->root);
  tree->size = res;
  return res;
}

size_t binary_tree_gen_size(binary_tree *tree) {
  size_t res = binary_node_get_size(tree->root);
  return res;
}

void __release_with_offset(void *ptr, size_t offset) {
  if (ptr == NULL) {
    return;
  }
  free((void *)((char *)ptr - offset));
}

void __free_binary_root_with_offset(binary_node *root, size_t offset) {
  if (root == NULL) {
    return;
  }
  __free_binary_root_with_offset(root->child[LEFT], offset);
  __free_binary_root_with_offset(root->child[RIGHT], offset);
  __release_with_offset(root, offset);
}

#define DESTROY_BINARY_TREE(NODE_TYPE, MEMBER, TREE_PTR) \
  __free_binary_root_with_offset(((TREE_PTR)->root),     \
                                 (offsetof(NODE_TYPE, MEMBER)))

int binary_tree_rotate(binary_tree *tree, binary_node *node,
                       enum direction dir) {
  if (node == NULL || tree == NULL) {
    return -1;
  }
  if (node->parent == NULL) {
    assert(node == tree->root);

    switch (dir) {
      case LEFT: {
        if (node->child[RIGHT] == NULL) {
          return -2;
        }

        binary_node *old_root = node, *new_root = node->child[RIGHT],
                    *child_to_swap = node->child[RIGHT]->child[LEFT];
        tree->root = new_root;

        new_root->child[LEFT] = old_root;
        old_root->child[RIGHT] = child_to_swap;

        if (child_to_swap != NULL) {
          child_to_swap->parent = old_root;
        }
        old_root->parent = new_root;
        new_root->parent = NULL;

        return 0;
      }
      case RIGHT: {
        if (node->child[LEFT] == NULL) {
          return -2;
        }

        binary_node *old_root = node, *new_root = node->child[LEFT],
                    *child_to_swap = node->child[LEFT]->child[RIGHT];
        tree->root = new_root;

        new_root->child[RIGHT] = old_root;
        old_root->child[LEFT] = child_to_swap;

        if (child_to_swap != NULL) {
          child_to_swap->parent = old_root;
        }
        old_root->parent = new_root;
        new_root->parent = NULL;

        return 0;
      }
      default: {
        return -3;
      }
    }
  }
  switch (dir) {
    case LEFT: {
      if (node->child[RIGHT] == NULL) {
        return -2;
      }
      binary_node *old_root = node, *new_root = node->child[RIGHT],
                  *child_to_swap = node->child[RIGHT]->child[LEFT];
      binary_node *parent = node->parent;

      assert(!(node == parent->child[LEFT] && node == parent->child[RIGHT]));
      if (node == parent->child[LEFT]) {
        parent->child[LEFT] = new_root;
      } else if (node == parent->child[RIGHT]) {
        parent->child[RIGHT] = new_root;
      } else {
        assert(0);
      }

      new_root->child[LEFT] = old_root;
      old_root->child[RIGHT] = child_to_swap;
      if (child_to_swap != NULL) {
        child_to_swap->parent = old_root;
      }
      old_root->parent = new_root;
      new_root->parent = parent;

      return 0;
    }
    case RIGHT: {
      if (node->child[LEFT] == NULL) {
        return -2;
      }
      binary_node *old_root = node, *new_root = node->child[LEFT],
                  *child_to_swap = node->child[LEFT]->child[RIGHT];
      binary_node *parent = node->parent;

      assert(!(node == parent->child[LEFT] && node == parent->child[RIGHT]));
      if (node == parent->child[LEFT]) {
        parent->child[LEFT] = new_root;
      } else if (node == parent->child[RIGHT]) {
        parent->child[RIGHT] = new_root;
      } else {
        assert(0);
      }

      new_root->child[RIGHT] = old_root;
      old_root->child[LEFT] = child_to_swap;
      if (child_to_swap != NULL) {
        child_to_swap->parent = old_root;
      }
      old_root->parent = new_root;
      new_root->parent = parent;

      return 0;
    }
    default: {
      return -3;
    }
  }
}

binary_node *binary_node_find_lowest_ancestor(binary_node *root,
                                              binary_node *node1,
                                              binary_node *node2) {
  if (root == NULL) {
    return NULL;
  } else if (root == node1 || root == node2) {
    return root;
  } else {
    binary_node *left = binary_node_find_lowest_ancestor(root->child[LEFT],
                                                         node1, node2),
                *right = binary_node_find_lowest_ancestor(root->child[RIGHT],
                                                          node1, node2);
    if (left && right) {
      return root;
    } else {
      if (left) {
        return left;
      } else if (right) {
        return right;
      } else {
        return NULL;
      }
    }
  }
}

binary_node *binary_tree_find_lowest_ancestor(binary_tree *tree,
                                              binary_node *node1,
                                              binary_node *node2) {
  assert(binary_tree_gen_size(tree) >= 2);
  return binary_node_find_lowest_ancestor(tree->root, node1, node2);
}

binary_node *construct_binary_tree(binary_node **node_list_mid,
                                   binary_node **node_list, size_t len,
                                   enum traverse_order seq) {
  switch (seq) {
    case PRE: {
      if (len < 1) {
        return NULL;
      }
      binary_node *root = node_list[0];
      if (len == 1) {
        root->child[LEFT] = NULL;
        root->child[RIGHT] = NULL;
        return root;
      }
      size_t pos;
      for (pos = 0; pos < len; pos++) {
        if (node_list_mid[pos] == root) {
          break;
        }
      }
      root->child[LEFT] =
          construct_binary_tree(node_list_mid, node_list + 1, pos, seq);
      root->child[RIGHT] = construct_binary_tree(
          node_list_mid + pos + 1, node_list + pos + 1, len - pos - 1, seq);
      return root;
    }
    case POST: {
      if (len < 1) {
        return NULL;
      }
      binary_node *root = node_list[len - 1];
      if (len == 1) {
        root->child[LEFT] = NULL;
        root->child[RIGHT] = NULL;
        return root;
      }
      size_t pos;
      for (pos = 0; pos < len; pos++) {
        if (node_list_mid[pos] == root) {
          break;
        }
      }
      root->child[LEFT] =
          construct_binary_tree(node_list_mid, node_list, pos, seq);
      root->child[RIGHT] = construct_binary_tree(
          node_list_mid + pos + 1, node_list + pos, len - pos - 1, seq);
      return root;
    }
    case IN: {
      assert(0);
    }
    default: {
      assert(0);
    }
  }
}

struct _internal_node_pair {
  binary_node *first, *second;
};

static inline struct _internal_node_pair _bst_small_big(
    binary_node *root, int (*cmp)(const binary_node *, const binary_node *)) {
  struct _internal_node_pair res = {NULL, NULL};
  if (root == NULL) {
    return (struct _internal_node_pair){.first = NULL, .second = NULL};
  }

  if (root->child[LEFT]) {
    struct _internal_node_pair left_pair =
        _bst_small_big(root->child[LEFT], cmp);
    if (left_pair.first == NULL || left_pair.second == NULL) {
      return (struct _internal_node_pair){.first = NULL, .second = NULL};
    } else if (cmp(left_pair.second, root) >= 0) {
      return (struct _internal_node_pair){NULL, NULL};
    } else {
      res.first = left_pair.first;
    }
  } else {
    res.first = root;
  }

  if (root->child[RIGHT]) {
    struct _internal_node_pair right_pair =
        _bst_small_big(root->child[RIGHT], cmp);
    if (right_pair.first == NULL || right_pair.second == NULL) {
      return (struct _internal_node_pair){NULL, NULL};
    } else if (cmp(root, right_pair.first) >= 0) {
      return (struct _internal_node_pair){NULL, NULL};
    } else {
      res.second = right_pair.second;
    }
  } else {
    res.second = root;
  }

  return res;
}

bool binary_node_is_bst_root(binary_node *root,
                             int (*cmp)(const binary_node *,
                                        const binary_node *)) {
  if (root == NULL) {
    return true;
  }

  struct _internal_node_pair node = _bst_small_big(root, cmp);
  if (node.first == NULL || node.second == NULL) {
    return false;
  }
  return true;
}

bool binary_tree_is_bst(binary_tree *tree,
                        int (*cmp)(const binary_node *, const binary_node *)) {
  return binary_node_is_bst_root(tree->root, cmp);
}

typedef binary_tree bst;
typedef binary_node bst_node;

int init_bst(bst *tree) { return init_binary_tree(tree); }

int bst_insert(bst *tree, bst_node *node,
               int (*cmp)(const bst_node *, const bst_node *)) {
  assert(node);
  if (binary_tree_get_size(tree) == 0) {
    binary_tree_insert(tree, NULL, node, LEFT);
    tree->size++;
    return 0;
  }

  bst_node *curr = tree->root;
  while (true) {
    assert(curr);
    int cmp_res = cmp(curr, node);
    if (!cmp_res) {
      return -1;
    }
    if (cmp_res > 0) {
      if (curr->child[LEFT] == NULL) {
        binary_tree_insert(tree, curr, node, LEFT);
        tree->size++;
        return 0;
      } else {
        curr = curr->child[LEFT];
      }
    } else {
      if (curr->child[RIGHT] == NULL) {
        binary_tree_insert(tree, curr, node, RIGHT);
        tree->size++;
        return 0;
      } else {
        curr = curr->child[RIGHT];
      }
    }
  }
}

bst_node *bst_find(bst *tree, bst_node *node,
                   int (*cmp)(const bst_node *, const bst_node *)) {
  if (!binary_tree_get_size(tree)) {
    return NULL;
  }
  if (!node) {
    return NULL;
  }

  bst_node *curr = tree->root;
  while (curr) {
    int cmp_res = cmp(curr, node);
    if (!cmp_res) {
      return curr;
    } else if (cmp_res < 0) {
      curr = curr->child[RIGHT];
    } else {
      curr = curr->child[LEFT];
    }
  }
  return curr;
}

// Be careful to use this func...
bst_node *bst_remove(bst *tree, bst_node *node,
                     int (*cmp)(const bst_node *, const bst_node *)) {
  bst_node *node_to_remove = bst_find(tree, node, cmp);
  if (node_to_remove == NULL) {
    return NULL;
  }
  tree->size--;
  // The parent == null need to be concerned!
  if (node_to_remove->child[LEFT] == NULL &&
      node_to_remove->child[RIGHT] == NULL) {
    bst_node *parent = node_to_remove->parent;
    if (parent == NULL) {
      tree->root = NULL;
      return node_to_remove;
    } else if (node_to_remove == parent->child[LEFT]) {
      parent->child[LEFT] = NULL;
      node_to_remove->parent = NULL;
      node_to_remove->child[LEFT] = NULL;
      node_to_remove->child[RIGHT] = NULL;
      return node_to_remove;
    } else if (node_to_remove == parent->child[RIGHT]) {
      parent->child[RIGHT] = NULL;
      node_to_remove->parent = NULL;
      node_to_remove->child[LEFT] = NULL;
      node_to_remove->child[RIGHT] = NULL;
      return node_to_remove;
    } else {
      assert(0);
    }
  } else if (node_to_remove->child[LEFT] == NULL) {
    bst_node *parent = node_to_remove->parent;
    if (parent == NULL) {
      tree->root = node_to_remove->child[RIGHT];
      node_to_remove->child[RIGHT]->parent = NULL;
      node_to_remove->child[RIGHT] = NULL;
      return node_to_remove;
    } else if (parent->child[LEFT] == node_to_remove) {
      parent->child[LEFT] = node_to_remove->child[RIGHT];
      node_to_remove->child[RIGHT]->parent = parent;
      node_to_remove->parent = NULL;
      node_to_remove->child[LEFT] = NULL;
      node_to_remove->child[RIGHT] = NULL;
      return node_to_remove;
    } else if (parent->child[RIGHT] == node_to_remove) {
      parent->child[RIGHT] = node_to_remove->child[RIGHT];
      node_to_remove->child[RIGHT]->parent = parent;
      node_to_remove->parent = NULL;
      node_to_remove->child[LEFT] = NULL;
      node_to_remove->child[RIGHT] = NULL;
      return node_to_remove;
    } else {
      assert(0);
    }
  } else if (node_to_remove->child[RIGHT] == NULL) {
    bst_node *parent = node_to_remove->parent;
    if (parent == NULL) {
      tree->root = node_to_remove->child[LEFT];
      node_to_remove->child[LEFT]->parent = NULL;
      node_to_remove->child[LEFT] = NULL;
      return node_to_remove;
    } else if (parent->child[LEFT] == node_to_remove) {
      parent->child[LEFT] = node_to_remove->child[LEFT];
      node_to_remove->child[LEFT]->parent = parent;
      node_to_remove->parent = NULL;
      node_to_remove->child[LEFT] = NULL;
      node_to_remove->child[RIGHT] = NULL;
      return node_to_remove;
    } else if (parent->child[RIGHT] == node_to_remove) {
      parent->child[RIGHT] = node_to_remove->child[LEFT];
      node_to_remove->child[LEFT]->parent = parent;
      node_to_remove->parent = NULL;
      node_to_remove->child[LEFT] = NULL;
      node_to_remove->child[RIGHT] = NULL;
      return node_to_remove;
    } else {
      assert(0);
    }
  } else {
    bst_node *nNode = node_to_remove->child[RIGHT];
    while (nNode->child[LEFT]) {
      nNode = nNode->child[LEFT];
    }
    if (nNode->child[LEFT] == NULL && nNode->child[RIGHT] == NULL) {
      bst_node *parent = nNode->parent;
      if (nNode == parent->child[LEFT]) {
        parent->child[LEFT] = NULL;
        nNode->parent = NULL;
      } else if (nNode == parent->child[RIGHT]) {
        parent->child[RIGHT] = NULL;
        nNode->parent = NULL;
      } else {
        assert(0);
      }
    } else {
      bst_node *parent = nNode->parent;
      bst_node *child = nNode->child[RIGHT];
      child->parent = parent;
      if (parent->child[LEFT] == nNode) {
        parent->child[LEFT] = nNode->child[RIGHT];
        nNode->parent = NULL;
      } else if (parent->child[RIGHT] == nNode) {
        parent->child[RIGHT] = nNode->child[RIGHT];
        nNode->parent = NULL;
      } else {
        assert(0);
      }
    }
    nNode->parent = node_to_remove->parent;
    nNode->child[LEFT] = node_to_remove->child[LEFT];
    nNode->child[RIGHT] = node_to_remove->child[RIGHT];
    nNode->child[LEFT]->parent = nNode;
    nNode->child[RIGHT]->parent = nNode;
    bst_node *parent = node_to_remove->parent;
    if (parent == NULL) {
      tree->root = nNode;
      nNode->parent = NULL;
    } else if (parent->child[LEFT] == node_to_remove) {
      parent->child[LEFT] = nNode;
    } else if (parent->child[RIGHT] == node_to_remove) {
      parent->child[RIGHT] = nNode;
    } else {
      assert(0);
    }
    node_to_remove->child[LEFT] = NULL;
    node_to_remove->child[RIGHT] = NULL;
    node_to_remove->parent = NULL;
    return node_to_remove;
  }
}

binary_node *bst_get_smallest(binary_tree *tree) {
  binary_node *it = tree->root;
  if (!it) {
    return it;
  }
  while (it->child[LEFT]) {
    it = it->child[LEFT];
  }
  return it;
}

binary_node *bst_get_greatest(binary_tree *tree) {
  binary_node *it = tree->root;
  if (!it) {
    return it;
  }
  while (it->child[RIGHT]) {
    it = it->child[RIGHT];
  }
  return it;
}

// RB Tree

typedef binary_tree rb_tree;

int init_rb_tree(rb_tree *tree) { return init_binary_tree(tree); }

static inline binary_node *_rbTree_get_grandpa(binary_node *node) {
  binary_node *parent = node->parent;
  if (parent == NULL) {
    return parent;
  }
  binary_node *grandpa = parent->parent;
  return grandpa;
}

static inline binary_node *_rbTree_get_uncle(binary_node *node) {
  binary_node *parent = node->parent;
  if (parent == NULL) {
    return parent;
  }
  binary_node *grandpa = parent->parent;
  if (grandpa == NULL) {
    return NULL;
  }
  if (parent == grandpa->child[LEFT]) {
    return grandpa->child[RIGHT];
  } else if (parent == grandpa->child[RIGHT]) {
    return grandpa->child[LEFT];
  } else {
    assert(0);
  }
}

static inline enum color _rbTree_color(binary_node *node) {
  if (node == NULL) {
    return BLACK;
  } else {
    return node->color;
  }
}

static inline void _rbTree_set_insert_balance(rb_tree *tree,
                                              binary_node *node) {
  // Maintain the balance.
  if (node == tree->root) {
    node->color = BLACK;
  } else if (_rbTree_color(node->parent) == RED &&
             _rbTree_color(_rbTree_get_uncle(node)) == RED) {
    binary_node *parent = node->parent, *uncle = _rbTree_get_uncle(node),
                *grandpa = _rbTree_get_grandpa(node);
    parent->color = BLACK, uncle->color = BLACK, grandpa->color = RED;
    _rbTree_set_insert_balance(tree, grandpa);
  } else if (_rbTree_color(node->parent) == RED &&
             _rbTree_color(_rbTree_get_uncle(node)) == BLACK) {
    binary_node *parent = node->parent, *grandpa = _rbTree_get_grandpa(node);
    if (parent == grandpa->child[LEFT] && node == parent->child[RIGHT]) {
      binary_tree_rotate(tree, parent, LEFT);
      binary_node *mid = node;
      node = parent;
      parent = mid;
    } else if (parent == grandpa->child[RIGHT] && node == parent->child[LEFT]) {
      binary_tree_rotate(tree, parent, RIGHT);
      binary_node *mid = node;
      node = parent;
      parent = mid;
    }

    if (parent == grandpa->child[LEFT] && node == parent->child[LEFT]) {
      binary_tree_rotate(tree, grandpa, RIGHT);
      // At this time, parent on the root, grandpa on the right parent and node
      // on the parent.
      parent->color = BLACK;
      node->color = RED;
      grandpa->color = RED;
    } else if (parent == grandpa->child[RIGHT] &&
               node == parent->child[RIGHT]) {
      binary_tree_rotate(tree, grandpa, LEFT);
      parent->color = BLACK;
      node->color = RED;
      grandpa->color = RED;
    }
  }

  if (tree->root) {
    tree->root->color = BLACK;
  }
}

int rb_tree_insert(rb_tree *tree, binary_node *node,
                   int (*cmp)(const binary_node *, const binary_node *)) {
  int insert_res = bst_insert(tree, node, cmp);
  if (insert_res) {
    return insert_res;
  }

  node->color = RED;
  _rbTree_set_insert_balance(tree, node);
  return 0;
}

static inline binary_node *_findSuccessor(binary_node *root) {
  binary_node *node = root->child[RIGHT];
  while (node && node->child[LEFT]) {
    node = node->child[LEFT];
  }
  return node;
}

binary_node *bst_find_successor(binary_tree *tree, binary_node *node) {
  binary_node *curr = node;

  if (!curr) {
    return NULL;
  }

  curr = curr->child[RIGHT];

  while (curr && curr->child[LEFT]) {
    curr = curr->child[LEFT];
  }

  return curr;
}

binary_node *bst_find_predcessor(binary_tree *tree, binary_node *node) {
  binary_node *curr = node;

  if (!curr) {
    return NULL;
  }

  curr = curr->child[LEFT];

  while (curr && curr->child[RIGHT]) {
    curr = curr->child[RIGHT];
  }

  return curr;
}

static inline binary_node *_rbTreeFindNode(rb_tree *tree, binary_node *node,
                                           int (*cmp)(const binary_node *,
                                                      const binary_node *)) {
  if (!tree->root) {
    return NULL;
  }

  binary_node *it = tree->root;
  int c = cmp(it, node);
  while (c) {
    if (c > 0) {
      it = it->child[LEFT];
    } else if (c < 0) {
      it = it->child[RIGHT];
    }
    if (!it) {
      break;
    }
    c = cmp(it, node);
  }
  return it;
}

binary_node *rb_tree_find_node(rb_tree *tree, binary_node *node,
                               int (*cmp)(const binary_node *,
                                          const binary_node *)) {
  return _rbTreeFindNode(tree, node, cmp);
}

static inline void _rbTree_switch_color(binary_node *node1,
                                        binary_node *node2) {
  enum color mid;
  if (node2 == node1) {
    return;
  }
  mid = node1->color;
  node1->color = node2->color;
  node2->color = mid;
}

// To be realized.
// Direction is the direction of the removed node.
static inline void _rbTreeRemoveMaintain(rb_tree *tree, binary_node *parent,
                                         enum direction dir) {
  // --- 开始修改: 添加 while 循环 ---
  while (parent) {
    binary_node *bro = parent->child[dir == LEFT ? RIGHT : LEFT];
    binary_node *close_nephew = NULL, *dist_nephew = NULL;
    if (bro) {
      close_nephew = bro->child[dir],
      dist_nephew = bro->child[dir == LEFT ? RIGHT : LEFT];
    }

    if (_rbTree_color(bro) == RED) {
      // Case 1: 兄弟节点是红色
      assert(parent->color == BLACK);
      binary_tree_rotate(tree, parent, dir);
      parent->color = RED;
      bro->color = BLACK;
      // 结构已改变，从相同的 parent 位置重新判断
      continue;
    } else if (_rbTree_color(bro) == BLACK &&
               _rbTree_color(close_nephew) == BLACK &&
               _rbTree_color(dist_nephew) == BLACK &&
               _rbTree_color(parent) == RED) {
      // Case 2: 兄弟和侄子都黑，父节点是红色
      bro->color = RED;
      parent->color = BLACK;
      break;  // 修复完成
    } else if (_rbTree_color(bro) == BLACK &&
               _rbTree_color(close_nephew) == BLACK &&
               _rbTree_color(dist_nephew) == BLACK &&
               _rbTree_color(parent) == BLACK) {
      // Case 3: 兄弟、侄子、父节点都黑
      bro->color = RED;
      // 问题向上传播
      if (parent->parent) {
        dir = (parent == parent->parent->child[LEFT] ? LEFT : RIGHT);
        parent = parent->parent;
        continue;  // 继续对新的父节点进行修复
      } else {
        // 到达根节点，结束
        break;
      }
    } else if (_rbTree_color(bro) == BLACK &&
               _rbTree_color(close_nephew) == RED &&
               _rbTree_color(dist_nephew) == BLACK) {
      // Case 4: 近侄子是红色
      enum direction rotate_dir = (dir == LEFT ? RIGHT : LEFT);
      binary_tree_rotate(tree, bro, rotate_dir);
      close_nephew->color = BLACK;
      bro->color = RED;
      dist_nephew = bro;
      bro = close_nephew;
      close_nephew = bro->child[dir];
      goto start_case_5;  // 维持原有逻辑，跳转到 Case 5
    } else if (_rbTree_color(bro) == BLACK &&
               _rbTree_color(dist_nephew) == RED) {
    // Case 5: 远侄子是红色
    start_case_5:
      binary_tree_rotate(tree, parent, dir);
      dist_nephew->color = BLACK;
      _rbTree_switch_color(parent, bro);
      break;  // 修复完成
    } else {
      // 无法处理的未知情况或已修复，退出
      break;
    }
  }  // --- 结束修改: while 循环 ---
}

binary_node *rb_tree_remove(rb_tree *tree, binary_node *node,
                            int (*cmp)(const binary_node *,
                                       const binary_node *)) {
  binary_node *to_remove = _rbTreeFindNode(tree, node, cmp);
  if (!to_remove) {
    return NULL;
  }

  // 4 cases
  tree->size--;
  // Case 0: node is root
  if (to_remove == tree->root && to_remove->child[LEFT] == NULL &&
      to_remove->child[RIGHT] == NULL) {
    tree->root = NULL;
  }
  // Case 1: node is leaf
  else if (!to_remove->child[LEFT] && !to_remove->child[RIGHT]) {
    binary_node *parent = to_remove->parent;
    enum direction dir;
    if (to_remove == parent->child[LEFT]) {
      parent->child[LEFT] = NULL;
      dir = LEFT;
    } else if (to_remove == parent->child[RIGHT]) {
      parent->child[RIGHT] = NULL;
      dir = RIGHT;
    }

    // Black leaf should be processed.
    if (to_remove->color == BLACK) {
      _rbTreeRemoveMaintain(tree, to_remove->parent, dir);
    }
  }
  // Case 2: node has child on both sides
  else if (to_remove->child[LEFT] && to_remove->child[RIGHT]) {
    binary_node *y = _findSuccessor(to_remove);
    enum color y_original_color = y->color;

    binary_node *fixup_parent;
    enum direction fixup_dir;

    if (y->parent != to_remove) {
      // y 不是 to_remove 的直接子节点
      fixup_parent = y->parent;
      fixup_dir = (y == fixup_parent->child[LEFT] ? LEFT : RIGHT);

      binary_tree_transplant(tree, y, y->child[RIGHT]);
      y->child[RIGHT] = to_remove->child[RIGHT];
      if (y->child[RIGHT]) y->child[RIGHT]->parent = y;
    } else {
      // y 是 to_remove 的直接子节点
      fixup_parent = y;
      fixup_dir = RIGHT;
    }

    binary_tree_transplant(tree, to_remove, y);
    y->child[LEFT] = to_remove->child[LEFT];
    if (y->child[LEFT]) y->child[LEFT]->parent = y;
    y->color = to_remove->color;

    if (y_original_color == BLACK) {
      // 使用预先保存的安全的父节点和方向进行修复
      _rbTreeRemoveMaintain(tree, fixup_parent, fixup_dir);
    }
  }
  // Case 3: node has child on the left
  else if (to_remove->child[LEFT]) {
    assert(!to_remove->child[RIGHT]);
    binary_node *child = to_remove->child[LEFT];
    binary_tree_transplant(tree, to_remove, child);
    child->color = BLACK;
    to_remove->child[LEFT] = NULL;
  }
  // Case 4: node has child on the right
  else if (to_remove->child[RIGHT]) {
    assert(!to_remove->child[LEFT]);
    binary_node *child = to_remove->child[RIGHT];
    binary_tree_transplant(tree, to_remove, child);
    assert(child->color == RED);
    child->color = BLACK;
    to_remove->child[RIGHT] = NULL;
  }
  if (tree->root) {
    tree->root->color = BLACK;
  }
  to_remove->parent = NULL;
  to_remove->child[LEFT] = NULL;
  to_remove->child[RIGHT] = NULL;
  return to_remove;
}
