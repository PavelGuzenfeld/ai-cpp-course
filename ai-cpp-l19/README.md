# Lesson 19: Linking Changes Semantics — Static vs Shared

## Goal

In Python, a module is imported once; there is one copy. In C++, the same
symbol can exist twice in one process, and which one you get changes
program behavior, not just binary size. This lesson makes that visible and
lets you break it on purpose.

## Build and Run

Inside the Docker container:

```bash
cd /workspace

# Static (default): each module gets its OWN copy of linking_registry's
# global state.
colcon build --packages-select nanobind-l19
source install/setup.bash
pytest ai-cpp-l19/ -v          # fails -- see "What You'll See" below

# Shared: rebuild with the registry as one shared library both modules
# link against.
rm -rf build/nanobind-l19 install/nanobind-l19
colcon build --packages-select nanobind-l19 --cmake-args -DLINKING_REGISTRY_SHARED=ON
source install/setup.bash
pytest ai-cpp-l19/ -v          # passes
```

## What You'll See

`registry.cpp` holds one process-global counter behind
`register_touch()`/`current_count()` — the minimal stand-in for
`gst-nvmm-cpp`'s process-global GType type cache. `module_a_native` and
`module_b_native` are two separate nanobind extension modules that both
link against it.

`test_linking.py` uses `threading.Barrier` so both modules touch the
registry at the same synchronized instant — an explicit barrier, not timing
luck, so the result below is deterministic on every run, not a rare flake:

```python
barrier = threading.Barrier(2)
# both threads call barrier.wait(), then register_touch() on their module
...
assert module_a_native.current_count() == baseline + 2
assert module_b_native.current_count() == baseline + 2
```

**Static build:** two touches happen, but `module_a_native.current_count()`
reads back `baseline + 1`, not `baseline + 2` — module A never sees module
B's touch. The test fails. This is exactly the shape of the real bug: two
plugins each believe they've fully registered a shared resource, and each is
wrong about the other's state.

**Shared build:** one counter, one increment sequence, both modules observe
every touch. The test passes.

## Why Static Gives You Two Copies

Each `add_library(linking_registry STATIC ...)` build produces an object
file that gets **copied whole** into every binary that links it. `module_a_native.so`
and `module_b_native.so` each end up with their own independent
`linking_demo::g_touch_count` — same source, same symbol name, two separate
storage locations, because static linking is a compile-time copy, not a
runtime reference.

A shared library is different: `liblinking_registry.so` is built once, and
`module_a_native.so`/`module_b_native.so` each carry an **undefined**
reference to `register_touch`/`current_count`, resolved to the *same*
loaded copy at runtime by the dynamic linker.

### The nm/ldd Walkthrough

Static build — neither module depends on a registry library at all, and
both define the symbol themselves, independently:

```
$ ldd module_a_native.so | grep registry
(nothing)

$ nm -D module_a_native.so | grep register_touch
0000000000049143 T _ZN12linking_demo14register_touchEv

$ nm -D module_b_native.so | grep register_touch
0000000000049143 T _ZN12linking_demo14register_touchEv
```

`T` means *defined here, in the text section* — both `.so` files, in full,
independently. The identical address is coincidental (both were built from
the same object layout); it does not mean they share memory at runtime.

Shared build — the dependency shows up, and the symbol moves:

```
$ ldd module_a_native.so | grep registry
liblinking_registry.so => /.../site-packages/./liblinking_registry.so (0x...)

$ nm -D module_a_native.so | grep register_touch
                 U _ZN12linking_demo14register_touchEv

$ nm -D liblinking_registry.so | grep register_touch
0000000000001139 T _ZN12linking_demo14register_touchEv
```

`U` means *undefined here — resolved elsewhere*. `module_a_native.so` no
longer carries its own copy; it asks the dynamic linker for one at load
time, and gets the same one `module_b_native.so` gets.

## RTLD_LOCAL: Why This Doesn't Even Need `-fvisibility=hidden`

A natural worry: if both static copies export `register_touch` at default
visibility, won't the dynamic linker's global symbol table quietly merge
them, accidentally "fixing" the bug through symbol interposition? Checked
directly against this interpreter:

```python
>>> import sys, os
>>> sys.getdlopenflags() & os.RTLD_GLOBAL
0
```

CPython's `dlopen` flags for extension modules do not set `RTLD_GLOBAL` —
each `.so`'s symbols stay local to that `.so` regardless of visibility
attributes. That's why the static-build divergence above reproduces
deterministically without any extra compiler flag.

`-fvisibility=hidden` is still worth using on real extension modules (mark
nanobind entry points and anything meant to be called from other C++
default-visible instead): it stops your module's internal helper names from
showing up in `nm -D` at all, which matters once a module has more than a
couple of internal functions and you don't want them polluting the dynamic
symbol table or colliding with an unrelated module that happens to pick the
same internal name. This lesson's two modules are small enough that the bug
above doesn't depend on it — but a larger codebase, or a build environment
that does set `RTLD_GLOBAL` (some embedders do, to let plugins call into
each other directly), would need it to keep the same guarantee.

## The ODR

Both modules `#include "registry.hpp"` and get a declaration of the same
functions. The One Definition Rule says there must be exactly one definition
of `linking_demo::g_touch_count` in the running program. Static linking
doesn't violate the ODR at compile time — each translation unit only sees
the declaration — but it does produce two definitions in the same *process*,
which is the ODR's actual concern, just not one the compiler or linker can
catch: nothing here is a compile error or a link error. It is a correct
build that produces the wrong behavior.

## The C++ ABI Boundary

`_GLIBCXX_USE_CXX11_ABI` (or equivalents on other standard libraries)
controls how `std::string`/`std::vector`/etc. are laid out. Two modules
built with different values of this macro, both linking the same static
library, can pass a `std::string` across their shared boundary and read
garbage — the object layouts don't match even though the source looks
identical. This is the same class of problem as the counter above, one
level more dangerous: it's a memory-layout mismatch, not just a
duplicated-state one. `readelf -p .comment module_a_native.so` shows the
compiler and flags a `.so` was built with, which is where you'd start
tracking this down in practice.

## Exercises

1. **Break `-fvisibility=hidden` on purpose**: add
   `set_target_properties(module_a_native PROPERTIES CXX_VISIBILITY_PRESET default)`
   and rebuild static. Does the divergence still reproduce? Why or why not,
   given what `sys.getdlopenflags()` showed above?

2. **Widen the barrier test**: change `test_integration_linking.py` to use 4
   threads (2 per module) instead of 2. Does the static build's divergence
   still land exactly on the expected per-module count?

3. **`readelf -p .comment`**: run it against both `module_a_native.so`
   builds (static and shared) and compare. What does it tell you about the
   compiler that produced each?

## What You Learned

- Static linking copies the object file into every binary that links it;
  shared linking resolves to one loaded copy at runtime — this changes
  program behavior for anything with global/static state, not just size
- `nm -D` (`T` = defined here, `U` = undefined, resolved elsewhere) and
  `ldd` are how you check which one you actually got
- CPython's extension-module `dlopen` uses `RTLD_LOCAL`, not `RTLD_GLOBAL`
  — verify this on your own interpreter before assuming it
- `-fvisibility=hidden` is good practice for extension modules regardless,
  to keep internal names out of the dynamic symbol table
- Two extension modules bundling different copies of the same static
  library is a live correctness bug, not a packaging inefficiency — L9's
  wheel matrix never had to say why bundling matters until now

## Lesson Files

| File | Description |
|------|-------------|
| [registry.hpp](registry.hpp) / [registry.cpp](registry.cpp) | The process-global counter, static or shared depending on the CMake option |
| [module_a_native.cpp](module_a_native.cpp) / [module_b_native.cpp](module_b_native.cpp) | Two identical nanobind modules, both linking `linking_registry` |
| [CMakeLists.txt](CMakeLists.txt) | `LINKING_REGISTRY_SHARED` option controlling static vs shared |
| [test_linking.py](test_linking.py) | Barrier-synchronized single-round divergence check |
| [test_integration_linking.py](test_integration_linking.py) | Same check across 5 rounds |
