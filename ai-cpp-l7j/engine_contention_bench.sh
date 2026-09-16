#!/bin/bash
# Which engine should a 1080p->480p resize run on? Sweeps VIC against GPU
# while a neighbour loads the GPU, because the answer changes with the load.
#
#   nvcc -O2 gpu_load.cu -o /tmp/gpu_load   # the neighbour
#   ./engine_contention_bench.sh
#
# Needs the JetPack GStreamer stack (nvvidconv). The source ceiling is
# measured first on purpose: videotestsrc is CPU-bound and caps the pipeline,
# so a convert faster than the ceiling is not being measured, only bounded.
set -u
N=${N:-900}
LOAD_BIN=${LOAD_BIN:-/tmp/gpu_load}

run() {
    local hw=$1 t0 t1
    t0=$(date +%s.%N)
    gst-launch-1.0 -q \
        videotestsrc num-buffers="$N" ! \
        "video/x-raw,width=1920,height=1080,format=NV12,framerate=1000/1" ! \
        ${hw:+nvvidconv compute-hw=$hw !} \
        ${hw:+"video/x-raw(memory:NVMM),width=854,height=480" !} \
        fakesink sync=false >/dev/null 2>&1
    t1=$(date +%s.%N)
    echo "scale=2; $N / ($t1 - $t0)" | bc
}

printf 'frames=%s  1920x1080 NV12 -> 854x480\n\n' "$N"
printf 'source only (no convert): %s fps   <- pipeline ceiling\n\n' "$(run '')"
printf '%-18s %8s %8s\n' 'GPU neighbours' 'VIC fps' 'GPU fps'
for n in 0 1 3 6; do
    for _ in $(seq 1 "$n"); do "$LOAD_BIN" >/dev/null 2>&1 & done
    [ "$n" -gt 0 ] && sleep 3
    printf '%-18s %8s %8s\n' "$n" "$(run 2)" "$(run 1)"
    [ "$n" -gt 0 ] && pkill -f "$(basename "$LOAD_BIN")"
done
