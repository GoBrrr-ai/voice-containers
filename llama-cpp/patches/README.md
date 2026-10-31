Post-v0.8.0 patches vendored here and applied to both the C++ tree and the
llama-cpp-python vendored submodule during build (pinned by commit SHA).

0001-cuda-optimize-orin-single-column-mmvq-pr31268.patch
    ggml-org/llama.cpp PR #31268 "CUDA: optimize Orin single-column MMVQ"
    (1 commit, head ad9855654e2ee9079c9d6ff6840b1df3467f54e0, open PR).
    Successor of #30858 (closed unmerged) whose Orin Nano Super benchmarks
    measured tg128 decode gains of +38% (Q4_0), +31% (Q5_1), +26% (Q4_1),
    +23% (Q8_0), +18% (Q4_K_S), +15% (Q4_K_M) with bit-identical perplexity
    and 2445/2445 backend ops passing. Guarded by
    cc == GGML_CUDA_CC_ORIN (870) && single output column -> zero effect on
    Thor/sm_110 paths. Verified to apply cleanly to tag v0.8.0.

0002-cuda-check-uma-before-cudamemgetinfo-pr25384.patch  [NOT APPLIED]
    PR #25384 was evaluated and deliberately dropped: at v0.8.0 the
    device_get_memory() path already tolerates cudaMemGetInfo failure
    (warn + fall through to the /proc/meminfo UMA path, PR #17368 code), so
    the patch no longer applies and its remaining delta (setting *total from
    prop.totalGlobalMem) is cosmetic for us. We pass an explicit -ngl at
    runtime, which sidesteps the auto-heuristic entirely. Kept here only as
    a record; the build applies /tmp/patches/*.patch, so DO NOT name any
    file you don't want applied with the .patch suffix in this directory.
