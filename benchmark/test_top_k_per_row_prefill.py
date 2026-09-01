# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark for top_k_per_row_prefill (DeepSeek V4 sparse attention).

Shapes match DeepSeek V4 production config:
    - vocab_size=129280: DeepSeek V4 vocabulary size (full BPE vocab)
    - top_k=1024: number of KV cache slots selected per token in sparse attention
    - num_rows=1: single-token prefill/decode-like latency-sensitive path
    - num_rows=32: typical prefill micro-batch
    - num_rows=64: larger prefill batch
    - num_rows=2048: max prefill sequence length

Latency is measured via CUDA graph capture + replay (see
TopKPerRowPrefillBenchmark.get_latency): the kernel is captured once and the
graph is replayed under CUDA-event timing, which removes host launch overhead
and matches how the op runs inside vLLM (piecewise CUDA graphs).

The baseline uses vLLM's persistent_topk CUDA kernel when available,
falling back to FlagGems' non-TLE implementation so the benchmark can run
in plain Triton-TLE development environments.
"""

from importlib import import_module

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops import top_k_per_row_prefill

from . import base

device = flaggems_vllm.device

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required",
)

# --- vLLM CUDA baseline (preferred) with PyTorch fallback ---
try:
    import vllm._custom_ops  # noqa: F401 — loads torch.ops._C

    def _vllm_top_k_per_row_prefill(
        logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
    ):
        torch.ops._C.top_k_per_row_prefill(
            logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
        )

    HAS_VLLM = True
except (ImportError, AttributeError):
    HAS_VLLM = False
    _vllm_top_k_per_row_prefill = None


_top_k_per_row_prefill_module = import_module("flaggems_vllm.ops.top_k_per_row_prefill")


def _non_tle_top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    device = logits.device
    s_histogram_ptr = torch.empty(
        (num_rows, _top_k_per_row_prefill_module.NUM_BINS),
        device=device,
        dtype=torch.int32,
    )
    s_final_logits_ptr = torch.empty(
        (num_rows, _top_k_per_row_prefill_module.NUM_FILNAL_ITEMS),
        device=device,
        dtype=torch.float32,
    )
    s_final_cnt_ptr = torch.empty((num_rows,), device=device, dtype=torch.int32)
    s_threshold_bin_idx_ptr = torch.empty((num_rows,), device=device, dtype=torch.int32)
    s_final_bin_size_ptr = torch.empty((num_rows,), device=device, dtype=torch.int32)
    s_found_topk_values_ptr = torch.empty((num_rows,), device=device, dtype=torch.int32)
    block_size = _top_k_per_row_prefill_module.NUM_THREADS_PER_BLOCK

    _top_k_per_row_prefill_module.non_tle_top_k_per_row_prefill[(num_rows,)](
        logits,
        indices,
        row_starts,
        row_ends,
        stride0,
        stride1,
        logits.shape[1],
        s_histogram_ptr,
        s_final_logits_ptr,
        s_final_cnt_ptr,
        s_threshold_bin_idx_ptr,
        s_final_bin_size_ptr,
        s_found_topk_values_ptr,
        TOPK=top_k,
        BLOCK_SIZE=block_size,
        ROW_OFFSET=0,
        num_warps=block_size // 32,
    )


class TopKPerRowPrefillBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "num_rows, vocab_size, top_k, stride0, stride1"

    def get_latency(self, op, *args, **kwargs):
        """Measure latency via CUDA graph capture + replay.

        Warmup runs on a side stream so Triton JIT compilation / config
        selection completes before capture. Only device-side work is recorded
        in the graph, so replay timing excludes host launch overhead and
        matches how the op runs inside vLLM (piecewise CUDA graphs).
        """
        fn = lambda: op(*args, **kwargs)  # noqa: E731

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(base.Config.warm_up):
                fn()
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(base.Config.repetition):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        # average latency in ms
        return start.elapsed_time(end) / base.Config.repetition

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            # DeepSeek V4 full vocabulary
            (64, 129280, 1024, 129280, 1),
            # DeepSeek-V4-Flash
            (4, 8193, 512, 8456, 1),
            (16383, 4095, 512, 4352, 1),
            (4, 16385, 512, 16648, 1),
            (12961, 4100, 512, 4360, 1),
            (16380, 5115, 512, 5376, 1),
            (4100, 1025, 512, 1288, 1),
        ]

    def get_input_iter(self, dtype):
        for num_rows, vocab_size, top_k, stride0, stride1 in self.shapes:
            torch.manual_seed(42)
            buf = torch.randn(
                (num_rows - 1) * stride0 + (vocab_size - 1) * stride1 + 1,
                device=device,
                dtype=torch.float32,
            )
            logits = torch.as_strided(buf, (num_rows, vocab_size), (stride0, stride1))
            row_starts = torch.zeros(num_rows, dtype=torch.int32, device=device)
            row_ends = torch.full(
                (num_rows,), vocab_size, dtype=torch.int32, device=device
            )
            indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=device)

            yield logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k


@pytest.mark.top_k_per_row_prefill
def test_top_k_per_row_prefill():
    baseline_op = (
        _vllm_top_k_per_row_prefill if HAS_VLLM else _non_tle_top_k_per_row_prefill
    )
    bench = TopKPerRowPrefillBenchmark(
        op_name="top_k_per_row_prefill",
        torch_op=baseline_op,
        gems_op=top_k_per_row_prefill,
        dtypes=[torch.float32],
    )
    bench.run()
