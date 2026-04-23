// dirty_code.cpp — deliberately contains 5 clang-tidy performance / modernize
// violations.  Your exercise:
//
//   clang-tidy dirty_code.cpp --warnings-as-errors=*
//
// Fix each warning in place and compare your result against clean_code.cpp.

#include <string>
#include <utility>
#include <vector>

// 1. performance-unnecessary-value-param — `s` is only read as const-ref.
int count_a(std::string s) {
  int n = 0;
  for (char c : s) n += (c == 'a');
  return n;
}

// 2. performance-unnecessary-copy-initialization — `copy` copies from a const ref.
int use_copy(const std::vector<std::string>& v) {
  const auto copy = v[0];   // v[0] on a const vector returns const T& → copy
  return int(copy.size());
}

// 3. performance-for-range-copy — `name` copies each string per iteration.
int total_len(const std::vector<std::string>& names) {
  int n = 0;
  for (auto name : names) n += int(name.size());
  return n;
}

// 4. performance-inefficient-vector-operation — no reserve() before the loop.
std::vector<std::string> build_names(int k) {
  std::vector<std::string> v;
  for (int i = 0; i < k; ++i) v.emplace_back("name_" + std::to_string(i));
  return v;
}

// 5. modernize-use-emplace — push_back with a temporary emplace could build.
void add_item(std::vector<std::pair<int, std::string>>& v, int k, const std::string& s) {
  v.push_back(std::make_pair(k, s));   // → v.emplace_back(k, s);
}
