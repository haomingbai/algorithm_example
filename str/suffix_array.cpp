/**
 * @file suffix_array.cpp
 * @brief
 * @author Haoming Bai <haomingbai@hotmail.com>
 * @date   2025-08-11
 *
 * Copyright © 2025 Haoming Bai
 * SPDX-License-Identifier: MIT
 *
 * @details
 */

#include <algorithm>
#include <cassert>
#include <iterator>
#include <numeric>
#include <vector>

#include "../concepts.cpp"

enum Type : bool { S_TYPE, L_TYPE };

struct _Bucket {
  long left, right;
};

// 这里按照理论上来说, 因为桶的左侧在诱导排序之前不会放元素,
// 所以左侧的初始化工作可以在函数里面做.
// 但是因为桶的右侧会先放LMS进去, 因此单独处理左侧并不清晰.
// 所以, 这里要求传入的桶数组必须做到左右两侧都已经处理完毕.
// 就是说, 这必须已经是当前适配SA数组的那个桶.
// str和type分别表示传入的串和字符的类型(S还是L),
// SA是经过部分处理, 放入了LMS的后缀数组半成品.
// prefix_sums是前缀和, 也就是从0到n的字母一共出现了几次.
// buckets是桶, 代表某个字母对应的下标应当在后缀数组中放置的位置边界.
// 这里字符i的桶未满的断言应该是:
// buckets[i].left <= buckets[i].right;
inline void _InducedSort(const std::vector<long> &str,
                         const std::vector<Type> &type, std::vector<long> &SA,
                         const std::vector<long> &prefix_sums,
                         std::vector<_Bucket> &buckets) {
  // 为了方便抄板子, 我这里先把一些变量提取出来.
  // 开销应该不大, 因为估计编译器一下就给消掉了.
  const long n = str.size(), max_val = buckets.size() - 1;
  // 从左向右扫描SA数组,
  // 这里的目标是从LMS进行L诱导.
  // 然后把L放到每个桶的头部.
  for (long i = 0; i < n; i++) {
    const auto curr_idx = SA[i];
    const auto prev_idx = curr_idx - 1;
    // 从当前的后缀下标诱导它的上一个字符的位置.
    if (prev_idx >= 0 && type[prev_idx] == L_TYPE) {
      // 将对应的下标放入相应的桶的头部,
      // 也就是桶的最左侧.
      const auto prev_char = str[prev_idx];
      auto &prev_bucket = buckets[prev_char];
      SA[prev_bucket.left] = prev_idx;
      prev_bucket.left++;
    }
  }
  // 将桶的底部重置, 这里的意思是删除那些LMS.
  // 删除掉LMS之后, 再根据之前放入的L的字符,
  // 诱导出所有的S的字符的位置.
  for (long i = 0; i <= max_val; i++) {
    // 这里桶的左右边界都是闭合的.
    // 比如buckets[c]代表[left, right]都是.
    // 那比如说字符0有a个, 那么[0, a - 1]都属于这个桶.
    // 当然如果a == 0, 那么就是[0, -1],
    // 也就意味着这个桶满了.
    buckets[i].right = prefix_sums[i] - 1;
  }
  // 从右往左扫描.
  // 这次扫描要把S类型的字符放进桶.
  for (long i = n - 1; i >= 0; i--) {
    const auto curr_idx = SA[i];
    const auto prev_idx = curr_idx - 1;
    // 从当前的后缀下标诱导它的上一个字符的位置.
    if (prev_idx >= 0 && type[prev_idx] == S_TYPE) {
      // 将对应的下标放入相应的桶的尾部,
      // 也就是桶的最右侧.
      const auto prev_char = str[prev_idx];
      auto &prev_bucket = buckets[prev_char];
      SA[prev_bucket.right] = prev_idx;
      prev_bucket.right--;
    }
  }
}
// str必须是已经被处理好的, 确认了最后的数字是全局唯一最小的哨兵的串.
// max_val可以给的稍微大一点也没关系.
inline std::vector<long> _SAIS(const std::vector<long> &str,
                               const long max_val) {
  // 第一步是获取字符串每个位置的类型,
  // 也就是说, 这个字符是S型还是L型.
  // 这个涉及这个字符比它正后方的那个字符是大还是小.
  // 我们见定义这两种类型为S_TYPE和L_TYPE.
  // 同时规定最后一个字符(它后面没有字符了)是S_TYPE.
  // 这里为了节省查询的时间, 就在注释里面写一下:
  // str[i] > str[i + 1], 记作L,
  // str[i] < str[i + 1], 记作S,
  // str[i] == str[i + 1], type[i] = type[i + 1].
  std::vector<Type> type(str.size(), S_TYPE);
  // 这里因为建立桶的需求,
  // 所以需要统计每个字符在这里都出现了几次.
  // 因为我们两次诱导排序,
  // 使用的是同一个串, 所以我们就不再排序的过程中扫描这个了哈.
  std::vector<long> cnt_occurance(max_val + 1, 0);
  // 尾部字符单独统计,
  // 因为下面扫描全字符串是从倒数第二个字符开始的.
  cnt_occurance[str.back()]++;
  // 收集所有LMS的下标.
  std::vector<long> lms_incidies;
  lms_incidies.reserve(str.size() / 2);
  // 逆序遍历字符串, 获取类型.
  // 这里逆序遍历的原因是, 如果
  // str[i] == str[i + 1], 那么则有:
  // type[i] = type[i + 1].
  // 我们在确定type[i]时, 必须知道type[i + 1].
  // 因此, 我们只能从后往前遍历.
  for (long i = str.size() - 2; i >= 0; i--) {
    if (str[i] > str[i + 1]) {
      type[i] = L_TYPE;
      // 在L_TYPE的右侧可能出现LMS
      // 如果L_TYPE的右侧是个S_TYPE,
      // 那么这个S_TYPE就是个LMS.
      if (type[i + 1] == S_TYPE) {
        lms_incidies.push_back(i + 1);
      }
    } else if (str[i] < str[i + 1]) {
      type[i] = S_TYPE;
    } else if (str[i] == str[i + 1]) {
      type[i] = type[i + 1];
    } else {
      assert(false);
    }
    // 记录出现次数.
    cnt_occurance[str[i]]++;
  }
  // 创建前缀和数组, 为建立桶和诱导排序做准备.
  std::vector<long> prefix_sums(max_val + 2);
  std::partial_sum(cnt_occurance.begin(), cnt_occurance.end(),
                   prefix_sums.begin());
  // 创建后缀数组, 这个数组在函数中要被使用很多次.
  std::vector<long> SA(str.size(), -1);
  // 创建桶, 这里先构造一个初始状态的桶, 不放入任何字符.
  std::vector<_Bucket> buckets(max_val + 1);
  buckets[0].left = 0, buckets[0].right = prefix_sums[0] - 1;
  for (long i = 1; i <= max_val; i++) {
    buckets[i].left = prefix_sums[i - 1];
    buckets[i].right = prefix_sums[i] - 1;
  }
  // 放入LMS.
  // 这里对于同一个字母, 入桶的顺序应该是倒序的.
  // 这个似乎和诱导排序的实现有关系.
  // 不管怎么样, 反正这里我们倒序入桶,
  // 确切来讲, 这里是顺序入桶底, 然后桶从右往左扩展.
  // 因此成倒序, 就像一个从右往左延伸的栈一样.
  for (auto it = lms_incidies.rbegin(); it != lms_incidies.rend(); it++) {
    auto curr_idx = *it;
    auto curr_char = str[curr_idx];
    auto &curr_bucket = buckets[curr_char];
    // 将对应的下标放入桶中.
    SA[curr_bucket.right] = curr_idx;
    curr_bucket.right--;
  }
  // 进行第一次诱导排序.
  _InducedSort(str, type, SA, prefix_sums, buckets);
  // 创建名字和下标的对应关系.
  // 这里用names数组表达对应位置的名字.
  std::vector<long> names(str.size(), -1);
  // 这两个变量分别记录了下发的名字的数量,
  // 和上一个被探测到的LMS的坐标.
  long name_cnt = 0;
  long last_lms_idx = -1;
  // 开始对LMS进行命名.
  for (const auto &it : SA) {
    // 判定一个字符是否是LMS.
    const auto is_lms = [&](const long idx) {
      // idx < 0 -> 下标不合法,
      // idx == 0 -> 当前字符左边没东西了,
      // 不可能左边有个L, 因此不可能是LMS.
      if (idx <= 0) {
        return false;
      }
      // 判断是否满足LMS条件.
      else if (type[idx] == S_TYPE && type[idx - 1] == L_TYPE) {
        return true;
      } else {
        return false;
      }
    };
    // 这里的这个函数是用来比较两个LMS子串是否相等的.
    const auto is_lms_eq = [&](const unsigned long idx1,
                               const unsigned long idx2) -> bool {
      // 如果下标相等就意味着两个LMS相等
      if (idx1 == idx2) {
        return true;
      }
      // 如果二者在下标不等的时候, 有一个是哨兵,
      // 那么另一个必然不是哨兵.
      // 那么二者不等.
      if (idx1 == str.size() - 1 || idx2 == str.size() - 1) {
        return false;
      }
      // 从偏移量为0开始比较
      long offset = 0;
      do {
        // 如果字符不相等, 就是不相等
        if (str[idx1 + offset] != str[idx2 + offset]) {
          return false;
        }
        // 字符相等, 类型不相等, 也是不相等.
        if (type[idx1 + offset] != type[idx2 + offset]) {
          return false;
        }
        // 手动更新偏移量
        offset++;
        // 循环条件: 两个待比较位置都没有来到下一个LMS
      } while (!is_lms(idx1 + offset) && !is_lms(idx2 + offset));
      // 如果有一个没有到达下一个LMS但是另外一个到达,
      // 那么二者一定不相等.
      if (!is_lms(idx1 + offset) || !is_lms(idx2 + offset)) {
        return false;
      }
      // 否则还是比较这两个LMS对应的字符.
      if (str[idx1 + offset] != str[idx2 + offset]) {
        return false;
      }
      if (type[idx1 + offset] != type[idx2 + offset]) {
        return false;
      }
      return true;
    };
    // 这个时候, it是当前正在处理的的lms下标
    if (is_lms(it)) {
      // 如果上一个LMS存在,
      // 需要检查名字是否相同.
      // 另外, 初始字符越大的, 名字一定越大,
      // 这里如果命名唯一的话,
      // 因为第二次诱导排序也是沿用的第一次的SA的顺序,
      // 所以因为没有递归, 所以可以省略第二次排序.
      // 这是通过了洛谷测试的 (虽然那个测试不咋靠谱),
      // 可以放心使用.
      if (last_lms_idx != -1) {
        // 如果二者不相等, 就要分配一个新名字.
        if (!is_lms_eq(it, last_lms_idx)) {
          name_cnt++;
        }
      } else {
        // 如果是第一个被处理的LMS,
        // 那么肯定要分配一个新名字.
        name_cnt++;
      }
      // 将名字下发下去.
      names[it] = name_cnt - 1;
      last_lms_idx = it;
    }
  }
  // 命名唯一, 无需递归,
  // 直接返回.
  if (static_cast<unsigned long>(name_cnt) == lms_incidies.size()) {
    return SA;
  } else {
    // 生成一个和长度和LMS个数相同的lms_str.
    std::vector<long> lms_str(lms_incidies.size());
    std::vector<long> lms_SA;
    // lms_incidies中的下标从小到大排序.
    // 也就是按照原串的顺序防止LMS下标.
    // 之前生成的时候顺序是相反的, 所以这里倒置一下.
    std::reverse(lms_incidies.begin(), lms_incidies.end());
    // 生成用于递归的LMS string.
    // LMS string中, 参与排序的是这些LMS子串的名字.
    for (unsigned long i = 0; i < lms_incidies.size(); i++) {
      auto curr_lms_idx = lms_incidies[i];
      lms_str[i] = names[curr_lms_idx];
    }
    // 最大的一个名字是最大的name_cnt - 1.
    lms_SA = _SAIS(lms_str, name_cnt - 1);
    // 生成一个新的桶并清空SA数组,
    // 进行第二次诱导排序.
    std::fill(SA.begin(), SA.end(), -1);
    buckets[0].left = 0, buckets[0].right = prefix_sums[0] - 1;
    for (long i = 1; i <= max_val; i++) {
      buckets[i].left = prefix_sums[i - 1];
      buckets[i].right = prefix_sums[i] - 1;
    }
    // 这里倒序遍历, 同时将遍历到的位置放在桶对应字母的右侧.
    // 因此可以保证lms_SA中靠右的下标会优先被处理后放入桶的右侧.
    // 因此对于同一个字母, lms_SA中靠右的在桶中也靠右.
    for (long i = lms_SA.size() - 1; i >= 0; i--) {
      // 这里套了两层的下标, 所以比较难以理解,
      // 这里的的意思是:
      // 在相同下标的位置, lms_str和lms_incidies是一一对应的.
      // 因此, lms_incidies[lms_SA[i]]就代表,
      // lms_str在lms_SA[i]这个下标的位置,
      // 存放的那个lms的名字对应的下标,
      // 就是lms_incidies[lms_SA[i]]中存放的那个下标.
      auto curr_lms_idx = lms_incidies[lms_SA[i]];
      // 同样地获取桶
      auto curr_char = str[curr_lms_idx];
      auto &curr_bucket = buckets[curr_char];
      // 将下标放在桶的右侧.
      SA[curr_bucket.right] = curr_lms_idx;
      curr_bucket.right--;
    }
    // 第二次诱导排序.
    _InducedSort(str, type, SA, prefix_sums, buckets);
    return SA;
  }
}

// 后缀数组计算.
template <typename E, RandomStdContainer<E> Container>
  requires std::is_convertible_v<E, unsigned long>
std::vector<unsigned long> BuildSuffixArray(const Container &str) {
  std::vector<long> processed(str.size() + 1);
  long max_val = 0;
  for (unsigned long i = 0; i < str.size(); i++) {
    // 强制为 unsigned 避免负值；+1 确保哨兵 (0) 比任何字符都小且唯一
    processed[i] = static_cast<unsigned char>(str[i]) + 1;
    if (processed[i] > max_val) max_val = processed[i];
  }
  processed.back() = 0;  // 哨兵，唯一且最小
  auto res = _SAIS(processed, max_val);
  // res[0] 对应哨兵的位置 (通常是 processed.size() - 1)
  std::vector<unsigned long> processed_res(std::next(res.begin()), res.end());
  return processed_res;
}

template <typename E, RandomStdContainer<E> Container>
  requires std::is_convertible_v<E, unsigned long>
std::vector<unsigned long> suffix_array(Container &&str) {
  std::vector<long> processed(str.size() + 1);
  long max_val = 0;
  for (unsigned long i = 0; i < str.size(); i++) {
    // 强制为 unsigned 避免负值；+1 确保哨兵 (0) 比任何字符都小且唯一
    processed[i] = static_cast<unsigned char>(str[i]) + 1;
    if (processed[i] > max_val) max_val = processed[i];
  }
  processed.back() = 0;  // 哨兵，唯一且最小
  auto res = _SAIS(processed, max_val);
  // res[0] 对应哨兵的位置 (通常是 processed.size() - 1)
  std::vector<unsigned long> processed_res(std::next(res.begin()), res.end());
  return processed_res;
}
