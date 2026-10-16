# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-NPU Engram hash correctness check and latency benchmark.

Run from the repository root in the matching Ascend/vLLM environment:
    python tests/e2e/nightly/single_node/ops/singlecard_ops/benchmark_engram_hash.py

Edit the configuration below. No model weights, embedding tables, distributed
initialization, lookup, or CPU offload are involved. The model dimensions match
V4.1, but token_map/multipliers/primes are deterministic SYNTHETIC parameters;
these results are not checkpoint-exact or end-to-end serving measurements.

Measures the production kernels directly, with preallocated inputs/outputs:
  hash_only: request IDs are already known.
  request_and_hash: request-index kernel followed by hash kernel.
Eager Event intervals include host launch gaps. Graph captures multiple calls
and reports average replay time per call, not individual-kernel percentiles.
Repeated calls use the same buffers (warm-cache microbenchmark).
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import random
import statistics
from pathlib import Path

# ------------------------ Edit configuration here -------------------------
DEVICE = "npu:0"
BATCH_SIZES = (1, 4, 8, 16, 32, 33, 64)
TOKENS_PER_REQUEST = (1, 6)  # 1: ordinary decode; 6: ACTUAL verification length
MODES = ("eager", "graph")
SCOPES = ("hash_only", "request_and_hash")
WARMUP_CALLS = 30
TRIALS = 30
CALLS_PER_SAMPLE = 100
SEED = 2026
OUTPUT_PATH = "benchmark_results/engram_hash.json"

# V4.1 structural dimensions; synthetic hash parameters are generated below.
NUM_LAYERS = 2
MAX_NGRAM = 4
NUM_HEADS = 8
VOCAB_SIZE = 129280
COMPRESSED_VOCAB_SIZE = 99092
HEAD_TABLE_ROWS = 16000000
PAD_ID = 2
DEAD_ID = -1  # only a synthetic sentinel for this direct-kernel benchmark
HISTORY_TOKENS = 128  # >= MAX_NGRAM-1 gives complete history in timed cases
BLOCK_T = 32
NUM_WARPS = 4
# -------------------------------------------------------------------------


def make_hash_parameters():
    """Generate positive, non-overflowing synthetic int64 hash parameters."""
    def is_prime(value):
        return value > 1 and all(value % divisor for divisor in range(2, math.isqrt(value) + 1))

    rng = random.Random(SEED)
    token_map = [index % COMPRESSED_VOCAB_SIZE for index in range(VOCAB_SIZE)]
    rng.shuffle(token_map)
    # Keep products below 2**63: this oracle does not exercise signed overflow.
    multipliers = [[rng.randrange(1 << 30, 1 << 40) | 1 for _ in range(MAX_NGRAM)] for _ in range(NUM_LAYERS)]
    primes, offsets = [], []
    candidate = HEAD_TABLE_ROWS + 1
    for _ in range(NUM_LAYERS):
        layer_primes, layer_offsets = [], []
        offset = 0
        for _ in range(MAX_NGRAM - 1):
            order_primes, order_offsets = [], []
            for _ in range(NUM_HEADS):
                while not is_prime(candidate):
                    candidate += 1
                order_primes.append(candidate)
                order_offsets.append(offset)
                offset += candidate
                candidate += 1
            layer_primes.append(order_primes)
            layer_offsets.append(order_offsets)
        primes.append(layer_primes)
        offsets.append(layer_offsets)
    assert max(max(row) for row in multipliers) * max(token_map) < 1 << 63
    assert max(value for layer in offsets for order in layer for value in order) < 1 << 31
    return dict(token_map=token_map, multipliers=multipliers, primes=primes, offsets=offsets)


def make_inputs(batch, tokens_per_request, *, boundary=False):
    """Uniform request lengths, nearest-first lookback; no graph padding."""
    rng = random.Random(SEED + batch * 17 + tokens_per_request)
    total = batch * tokens_per_request
    starts = [index % MAX_NGRAM if boundary else HISTORY_TOKENS for index in range(batch)]
    lookback = [
        [rng.randrange(VOCAB_SIZE) if start > col else -1 for col in range(MAX_NGRAM - 1)] for start in starts
    ]
    return dict(
        input_ids=[rng.randrange(VOCAB_SIZE) for _ in range(total)],
        positions=[start + step for start in starts for step in range(tokens_per_request)],
        query_start_loc=[index * tokens_per_request for index in range(batch + 1)],
        req_ids=[req for req in range(batch) for _ in range(tokens_per_request)],
        lookback_token_ids=lookback,
        dead_mask=[boundary and index % 5 == 0 for index in range(total)],
        lookback_dead_mask=[
            [boundary and req % 3 == 0 and col == 0 for col in range(MAX_NGRAM - 1)] for req in range(batch)
        ],
    )


def reference_hash(inputs, parameters):
    """Scalar CPU oracle: independently address each request's token history."""
    result = []
    for token, req in enumerate(inputs["req_ids"]):
        chunk_begin = inputs["query_start_loc"][req]
        chunk_position = inputs["positions"][chunk_begin]
        position = inputs["positions"][token]
        token_result = []
        for layer in range(NUM_LAYERS):
            rolling, blocked, ids = 0, False, []
            for shift in range(MAX_NGRAM):
                history_position = position - shift
                if history_position >= chunk_position:
                    source_index = token - shift
                    raw_id = inputs["input_ids"][source_index]
                    dead = inputs["dead_mask"][source_index]
                else:
                    col = chunk_position - 1 - history_position
                    raw_id = inputs["lookback_token_ids"][req][col]
                    dead = raw_id >= 0 and inputs["lookback_dead_mask"][req][col]
                source = parameters["token_map"][raw_id] if raw_id >= 0 else PAD_ID
                if dead:
                    source = DEAD_ID
                blocked = blocked or history_position < 0 or source == DEAD_ID
                value = PAD_ID if blocked else source
                rolling ^= value * parameters["multipliers"][layer][shift]
                if shift:
                    ids.extend(
                        rolling % prime + offset
                        for prime, offset in zip(
                            parameters["primes"][layer][shift - 1], parameters["offsets"][layer][shift - 1]
                        )
                    )
            token_result.append(ids)
        result.append(token_result)
    return result


def prepare_case(torch, kernels, inputs, parameters):
    """All allocation/transfers happen before correctness checks and timing."""
    tensors = {}
    for name, values in {**inputs, **parameters}.items():
        dtype = torch.bool if name in ("dead_mask", "lookback_dead_mask") else torch.int64
        if name in ("input_ids", "lookback_token_ids", "query_start_loc", "req_ids"):
            dtype = torch.int32
        tensors[name] = torch.tensor(values, dtype=dtype, device=DEVICE)
    total = len(inputs["input_ids"])
    output = torch.empty((total, NUM_LAYERS, (MAX_NGRAM - 1) * NUM_HEADS), dtype=torch.int32, device=DEVICE)
    window = tensors["lookback_token_ids"]
    window_mask = tensors["lookback_dead_mask"]
    launch_args = dict(
        input_stride=1,
        mask_stride=1,
        position_stride=1,
        table_stride=0,
        table_col_stride=0,
        query_stride=1,
        num_table_rows=0,
        max_blocks=0,
        cache_block_size=1,
        MAX_NGRAM=MAX_NGRAM,
        num_heads=NUM_HEADS,
        BLOCK_T=BLOCK_T,
        BLOCK_H=1 << (NUM_HEADS - 1).bit_length(),
        dead_id=DEAD_ID,
        lookback_depth=window.shape[1],
        lookback_row_stride=window.stride(0),
        lookback_col_stride=window.stride(1),
        lookback_mask_row_stride=window_mask.stride(0),
        lookback_mask_col_stride=window_mask.stride(1),
        num_warps=NUM_WARPS,
    )

    def hash_only():
        kernels._hash_ids_kernel[(math.ceil(total / BLOCK_T), NUM_LAYERS)](
            tensors["input_ids"],
            tensors["token_map"],
            tensors["dead_mask"],
            tensors["positions"],
            None,
            tensors["query_start_loc"],
            tensors["req_ids"],
            tensors["multipliers"],
            tensors["primes"],
            tensors["offsets"],
            None,
            window,
            window_mask,
            output,
            total,
            0,
            PAD_ID,
            **launch_args,
        )

    def request_and_hash():
        kernels._engram_req_index_kernel[(math.ceil(total / BLOCK_T),)](
            tensors["query_start_loc"],
            tensors["req_ids"],
            total,
            len(inputs["query_start_loc"]) - 1,
            tensors["query_start_loc"].stride(0),
            BLOCK_T=BLOCK_T,
        )
        hash_only()

    return dict(hash_only=hash_only, request_and_hash=request_and_hash), tensors, output


def check_case(torch, kernels, inputs, parameters):
    paths, tensors, output = prepare_case(torch, kernels, inputs, parameters)
    expected = torch.tensor(reference_hash(inputs, parameters), dtype=torch.int32)
    for name, run in paths.items():
        if name == "request_and_hash":
            tensors["req_ids"].fill_(-1)  # ensure request search really writes IDs
        output.fill_(-1)
        run()
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(tensors["req_ids"].cpu(), torch.tensor(inputs["req_ids"], dtype=torch.int32))
    return paths, tensors, output


def measure(torch, run, mode):
    for _ in range(WARMUP_CALLS):
        run()
    torch.npu.synchronize()
    graph = None
    if mode == "graph":
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(CALLS_PER_SAMPLE):
                run()
        for _ in range(WARMUP_CALLS):
            graph.replay()
        torch.npu.synchronize()
    samples = []
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    for _ in range(TRIALS):
        begin.record()
        if graph is not None:
            graph.replay()
        else:
            for _ in range(CALLS_PER_SAMPLE):
                run()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000 / CALLS_PER_SAMPLE)
    return dict(
        median_us=statistics.median(samples), min_us=min(samples), max_us=max(samples), samples_us=samples
    )


def package_versions():
    versions = {}
    for name in ("torch", "torch-npu", "vllm", "vllm-ascend", "triton", "triton-ascend"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-only", action="store_true", help="Check CPU oracle without importing NPU packages")
    args = parser.parse_args()
    assert BATCH_SIZES and all(batch > 0 for batch in BATCH_SIZES)
    assert TOKENS_PER_REQUEST and all(count > 0 for count in TOKENS_PER_REQUEST)
    assert TRIALS > 0 and CALLS_PER_SAMPLE > 0 and WARMUP_CALLS > 0
    assert MAX_NGRAM >= 2 and NUM_LAYERS > 0 and NUM_HEADS > 0
    assert VOCAB_SIZE >= COMPRESSED_VOCAB_SIZE > PAD_ID >= 0
    assert HISTORY_TOKENS >= MAX_NGRAM - 1
    assert set(MODES) <= {"eager", "graph"} and set(SCOPES) <= {"hash_only", "request_and_hash"}
    parameters = make_hash_parameters()
    # One fixed-vector check with independently calculable 2-gram output.
    smoke = make_inputs(1, 1)
    expected = reference_hash(smoke, parameters)
    current = parameters["token_map"][smoke["input_ids"][0]]
    previous = parameters["token_map"][smoke["lookback_token_ids"][0][0]]
    rolling = (current * parameters["multipliers"][0][0]) ^ (previous * parameters["multipliers"][0][1])
    assert expected[0][0][0] == rolling % parameters["primes"][0][0][0]
    if args.reference_only:
        for count in TOKENS_PER_REQUEST:
            for boundary in (False, True):
                result = reference_hash(make_inputs(33, count, boundary=boundary), parameters)
                assert len(result) == 33 * count
                assert all(
                    len(row) == NUM_LAYERS and all(len(ids) == (MAX_NGRAM - 1) * NUM_HEADS for ids in row)
                    for row in result
                )
        print("CPU reference smoke checks passed; NPU correctness and timing were NOT run.")
        return
    # Lazy imports let --help/--reference-only run on a machine without torch.
    torch = importlib.import_module("torch")
    importlib.import_module("torch_npu")
    if not torch.npu.is_available():
        raise RuntimeError("This benchmark requires an available Ascend NPU")
    torch.npu.set_device(DEVICE)
    kernels = importlib.import_module("vllm_ascend.ops.triton.engram_hash")
    results = dict(
        configuration={name: value for name, value in globals().items() if name.isupper()},
        environment=dict(
            python=platform.python_version(),
            device=torch.npu.get_device_name(),
            packages=package_versions(),
            kernel_file=kernels.__file__,
            kernel_sha256=hashlib.sha256(Path(kernels.__file__).read_bytes()).hexdigest(),
        ),
        parameter_kind="synthetic_positive_int64_no_overflow",
        records=[],
    )
    for count in TOKENS_PER_REQUEST:
        check_case(torch, kernels, make_inputs(33, count, boundary=True), parameters)
        for batch in BATCH_SIZES:
            inputs = make_inputs(batch, count)
            paths, tensors, output = check_case(torch, kernels, inputs, parameters)
            expected = torch.tensor(reference_hash(inputs, parameters), dtype=torch.int32)
            cases = [(mode, scope) for mode in MODES for scope in SCOPES]
            random.Random(SEED + batch + count).shuffle(cases)
            for mode, scope in cases:
                measured = measure(torch, paths[scope], mode)
                # Also check captured graph output; transfers stay outside timing.
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
                record = dict(
                    batch=batch,
                    tokens_per_request=count,
                    total_tokens=batch * count,
                    output_shape=list(output.shape),
                    mode=mode,
                    scope=scope,
                    correctness="passed",
                    **measured,
                )
                results["records"].append(record)
                print(f"B={batch:3d} S={count} {mode:5s} {scope:16s} {measured['median_us']:.3f} us", flush=True)
            # Retain all tensors through graph destruction and stream completion.
            torch.npu.synchronize()
            del paths, tensors, output
    destination = Path(OUTPUT_PATH)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {destination.resolve()}")


if __name__ == "__main__":
    main()
