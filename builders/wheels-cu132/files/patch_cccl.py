#!/usr/bin/env python3
"""Patch CCCL device_transform.cuh for GCC + CUDA 13.2.

CUDA 13.2's cccl/cub/device/device_transform.cuh declares a template
specialization of cuda::proclaims_copyable_arguments with a globally
qualified name (struct ::cuda::...) at global scope. GCC rejects that
("global qualification of class name is invalid"). Wrap the specialization
in a namespace block and drop the leading qualification, which is what the
upstream CCCL fix does.

Line-based and tolerant of template-parameter spelling (class/typename).
"""

import os
import sys

MARKER = "struct ::cuda::proclaims_copyable_arguments"

PATHS = [
    "/usr/local/cuda/targets/sbsa-linux/include/cccl/cub/device/device_transform.cuh",
    "/usr/local/cuda/targets/aarch64-linux/include/cccl/cub/device/device_transform.cuh",
    "/usr/local/cuda/include/cccl/cub/device/device_transform.cuh",
]
if os.environ.get("PATCH_CCCL_PATHS"):
    PATHS = os.environ["PATCH_CCCL_PATHS"].split(":")

patched = 0
for path in PATHS:
    try:
        lines = open(path).read().split("\n")
    except OSError:
        continue
    idx = next((i for i, l in enumerate(lines) if MARKER in l), None)
    if idx is None:
        print("no qualified specialization found (skipping)", path)
        continue
    # walk back to the nearest 'template' line
    start = idx
    while start > 0 and not lines[start].lstrip().startswith("template"):
        start -= 1
    # the specialization body must close on the line right after the struct
    # line (e.g. "{};"); refuse anything else rather than over-wrap
    end = idx
    while end < len(lines) - 1 and not lines[end].rstrip().endswith("};"):
        end += 1
    if end != idx + 1:
        print(f"unexpected specialization shape at line {idx + 1} (skipping)", path)
        continue
    body = [
        l.replace(
            "::cuda::proclaims_copyable_arguments", "proclaims_copyable_arguments"
        )
        for l in lines[start : end + 1]
    ]
    replacement = ["namespace cuda {"] + body + ["} // namespace cuda"]
    lines[start : end + 1] = replacement
    open(path, "w").write("\n".join(lines))
    print(f"patched {path} (lines {start + 1}-{end + 1})")
    patched += 1

if patched == 0:
    print("WARNING: no CCCL header patched")
sys.exit(0)
