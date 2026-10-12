#pragma once

namespace n2_detail {
// Size of the whole sink set when a kernel runs on one rank's slice of it (SinkSetScope, set by the distributed
// callers), 0: the slice is the set. The source-chunk count, the 256-thread tile switch and the host path of run_n2
// depend on it, so every sink's sum is the same whatever the number of ranks sharing the set (SCALING_DESIGN.md
// sec. 6.1: with the slice size they changed with the GPU count)
inline int& sink_set_size() { static int v = 0; return v; }
struct SinkSetScope {
    int prev;
    explicit SinkSetScope(int n) : prev(sink_set_size()) { sink_set_size() = n; }
    ~SinkSetScope() { sink_set_size() = prev; }
};
inline int set_size(int ni) { return sink_set_size() > 0 ? sink_set_size() : ni; }
}  // namespace n2_detail
