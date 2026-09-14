"""Cross-language interop verifier: reads values written by the C++ writer
and writes values for the C++ side to read back."""

import safe_shm_py as shm
import sys

# Test 1: Seqlock read
reader = shm.SeqlockReaderF64("xlang_seqlock")
val = reader.load()
assert abs(val - 3.14159265358979) < 1e-10, f"Seqlock: expected π, got {val}"

# Test 2: CyclicBuffer read
cb = shm.CyclicBufferReaderStampedF64("xlang_cyclic")
assert cb.available() == 20, f"CyclicBuffer: expected 20 entries, got {cb.available()}"
latest = cb.get_latest()
assert latest.sequence == 19, f"CyclicBuffer latest seq: expected 19, got {latest.sequence}"
assert abs(latest.data - 19 * 1.5) < 1e-10, f"CyclicBuffer latest data: expected {19*1.5}, got {latest.data}"

# Test 3: TimeSeries queries
ts = shm.TimeSeriesStampedF64("xlang_cyclic")
closest = ts.find_closest(7500)
assert closest is not None, "TimeSeries find_closest returned None"
assert closest.sequence in (7, 8), f"TimeSeries closest seq: expected 7 or 8, got {closest.sequence}"

interp = ts.find_interpolation_pair(7500)
assert interp is not None, "TimeSeries find_interpolation_pair returned None"
assert interp.before.sequence == 7, f"InterpPair before seq: expected 7, got {interp.before.sequence}"
assert interp.after.sequence == 8, f"InterpPair after seq: expected 8, got {interp.after.sequence}"
assert abs(interp.alpha - 0.5) < 1e-10, f"InterpPair alpha: expected 0.5, got {interp.alpha}"

# Test 4: Python writes, C++ reads (write to separate SHM)
writer = shm.SeqlockWriterF64("xlang_py_write")
writer.store(2.71828)

cb_writer = shm.CyclicBufferWriterStampedF64("xlang_py_cyclic")
for i in range(10):
    s = shm.StampedF64()
    s.timestamp_ns = (i + 1) * 500
    s.sequence = i
    s.data = float(i) * 2.0
    cb_writer.insert(s)

print("All Python assertions passed")
# Signal done
open("/tmp/xlang_test_done", "w").close()
