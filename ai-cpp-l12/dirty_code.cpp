// dirty_code.cpp — deliberately contains 5 clang-tidy performance / modernize
// violations.  Your exercise:
//
//   clang-tidy dirty_code.cpp --warnings-as-errors=*
//
// Fix each warning in place and compare your result against clean_code.cpp.

#include <map>
#include <string>
#include <vector>

// 1. performance-unnecessary-value-param — `s` is only read as const-ref.
int count_a(std::string s) {
  int n = 0;
  for (char c : s) n += (c == 'a');
  return n;
}

// 2. performance-unnecessary-copy-initialization — `copy` copies a const ref.
int use_copy(const std::map<int, std::string>& m) {
  const auto copy = m.find(0)->second;  // should be: const auto& copy = ...
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

// 5. modernize-use-emplace — push_back with an rvalue that emplace can build.
void add_mapping(std::map<int, std::string>& m, int key, const std::string& val) {
  m.insert(std::pair<int, std::string>(key, val));   // → m.emplace(key, val);
}
