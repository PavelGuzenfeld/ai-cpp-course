// clean_code.cpp — the fixed version of dirty_code.cpp.  Every warning from
// clang-tidy has been silenced by actually fixing the bug, not by adding
// `NOLINT` comments.

#include <string>
#include <string_view>
#include <utility>
#include <vector>

// 1. Pass by string_view (or const std::string&) for read-only strings.
int count_a(std::string_view s) {
  int n = 0;
  for (const char c : s) n += (c == 'a');
  return n;
}

// 2. Bind by const reference — no copy.
int use_copy(const std::vector<std::string>& v) {
  const auto& copy = v[0];
  return int(copy.size());
}

// 3. Bind by const reference in the range loop.
int total_len(const std::vector<std::string>& names) {
  int n = 0;
  for (const auto& name : names) n += int(name.size());
  return n;
}

// 4. reserve() before the loop — one allocation instead of log2(k) reallocations.
std::vector<std::string> build_names(int k) {
  std::vector<std::string> v;
  v.reserve(k);
  for (int i = 0; i < k; ++i) v.emplace_back("name_" + std::to_string(i));
  return v;
}

// 5. emplace in-place rather than constructing a std::pair as a temporary.
void add_item(std::vector<std::pair<int, std::string>>& v, int k, const std::string& s) {
  v.emplace_back(k, s);
}
