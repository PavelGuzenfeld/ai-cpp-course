#pragma once

namespace linking_demo
{

// A process-global counter, the minimal stand-in for gst-nvmm-cpp's
// process-global GType type cache. Static linking gives every module that
// links this library its own copy of g_touch_count (see registry.cpp);
// shared linking gives every module the same one.
int register_touch();
int current_count();

} // namespace linking_demo
