# PCIe collective preparation

`AllReduce.plan` declares the eligible transports for an established channel.
`PreparationSession` races them jointly across the communicator and retains the
selected native execution state before graph capture. Each callsite submits its
actual shape, dtype, layout, arithmetic options, and aliasing contract. Equivalent
declarations coalesce; request names and layer numbers do not create separate
tuning keys. Every caller still binds its own serving operands.

## Candidate operations

| Declaration | Candidates |
| --- | --- |
| Staged `AllReduce.all_reduce`, direct peer groups | Native pull; LL and Simple peer push; BF16 payload-ready full reduction and reduce-scatter/all-gather; provisioned BF16 two-shot |
| Staged fused residual addition and RMSNorm | Native staged transports compatible with declared input, residual, normalization, grouping, and output arithmetic |
| DCP `all_gather_heads` and `lse_reduce_scatter`, TP2/4/8 | Thread count and block limit; equivalent effective grids are measured once |
| DCP `all_gather_heads`, TP4/8/16 | Ring on TP4/8/16; phased and recursive profiles on TP8/16, on SM120 devices with at least 64 SMs |
| DCP `lse_reduce_scatter`, TP4 | Direct and XOR pairwise push schedules |
| BF16 two-shot push, TP4 | Direct and XOR pairwise payload schedules |
| IslandRS `all_reduce`, TP16 | Thread and active-CTA counts; four SM transports combining local pull or push with cross-island full reduction or reduce-scatter/all-gather; hybrid local SM writes and cross-island copy-engine transfers |

Payload-ready transports require their reserved incoming slots, BF16 support,
and sufficient SMs to keep their grid resident. Two-shot retains its native
capacity and alignment constraints. Pairwise schedules require SM120 with at
least 64 SMs and a resident eight-CTA cluster of 128-thread blocks; explicit
thread or block overrides exclude them.
Unsupported alternatives are absent, rather than failing over during execution.
Plain tuning admits FP16/BF16 tensors with positive 16-byte-packed payloads
within the prepared staging capacity. LL requires two capacity shards per remote
source; Simple additionally keeps payload bytes disjoint from its 36 CTA flag
records. Full reduction and reduce-scatter/all-gather use separate BF16 payload
shards. Widths, row counts, and divisibility by rank count do not restrict this
measured candidate catalog.
The peer-push transports require the prepared alternating graph slots. Thread counts
32, 64, 128, 256, and 512 are measured over grids up to 36 CTAs, including the
payload-covering grid. Pull remains a measured alternative. Optimized TP2 LL and
Simple implementations remain separate entrants alongside generic peer push.
Two-shot measures warp-aligned thread counts through 512 and block limits through
64, coalescing limits that produce the same effective grid.

Native peer-push kernels support rank counts 2, 3, 4, 6, 8, and 10 with fully
mapped peers. Single-channel TP10 auto preparation provisions the actual direct
pool and retains it on success. Only a coordinated peer-mapping rejection with
complete rollback selects hierarchy; unrelated setup or cleanup failures remain
fatal. Explicit hierarchical choices and multichannel TP10 declarations retain
hierarchical routing. Pairwise reachability alone does not establish simultaneous
full-peer mapping capacity. Inputs and outputs must be 16-byte aligned, use
compatible physical traversal, and be either exact aliases or disjoint.

Tuning races the native alternatives over their launch geometry; explicitly
disabled tuning retains the static dispatch policy. Explicit
`CollectiveConfig(implementation=..., threads=..., blocks=...)` pins a legal
configuration. Native low-level plans still use `PcieConfig`.

Gather tuning retains ordinary DCP as its default. Selected alternatives
require disjoint contiguous, 16-byte-aligned rows. The phased profile requires a
compiled kernel that can keep its complete eight-CTA cluster active; its
unpinned thread counts are 128, 256, and 512. The ring profile admits 128 or 512
threads; recursive profiles use 128 threads. Neither uses a cluster.
All selected profiles have eight physical and active CTAs. Existing DCP
thread and block overrides constrain every implementation. A block override
other than eight excludes the selected profiles. Ring excludes thread overrides
outside 128 and 512; recursive profiles exclude thread overrides other than 128.
`PcieConfig(backend="native")` retains the fixed native declaration. Each exact
message shape races normally
without enabling a runtime ladder or changing the all-reduce channel.
The loaded phased kernel must pass its occupancy check before capture; failure
stops preparation. Isolated races do not measure cluster-launch waits behind
overlapping compute. The selection key does not describe PCIe links; retune
after re-cabling a GPU group.

IslandRS assumes 16 ranks arranged as four contiguous four-rank islands.
Preparation races five transports over 128, 256, 512, 768, and 1024 threads and
the active-CTA counts admitted by the channel's physical grid, capped at 64 CTAs
and one CTA per SM. Explicit launch overrides retain their declared
participation. All choices share peer mappings, ordered FP32 sums, and BF16
conversion points. Selection records retain the exact shape and input/output
alias pattern.

The SM transports combine local pull or push with cross-island full reduction
or reduce-scatter/all-gather. They launch the fixed physical grid. Every physical
CTA advances the data-buffer generation; only participating CTAs advance their
communication epoch. Separate counters preserve bank parity when inactive CTAs
rejoin. Local push places input and completed reduction quarters in recipients'
staging planes. Cross-island reduce-scatter/all-gather assigns each stride
position to one island, then gathers completed BF16 values. Distinct partial
and final readiness tags use signed modular ordering to support generation wrap.

The hybrid transport, `island_hybrid_rsag`, uses eight transfer batches. Local
SM writes overlap cross-island copies on a dedicated native CUDA stream.
Packet scratch and rendezvous flags occupy a separate region of the channel
slab. At a 14 MiB BF16 input capacity, this region adds 22.75 MiB plus 256 bytes
per rank. The channel owns the stream and events until queued work drains during
close. The transport advances the SM protocol's buffer generations and
communication epochs so serialized calls can alternate transports.

Default-effort tuning groups compatible FP16 and BF16 plain all-reduce
payloads and contiguous BF16 IslandRS payloads by transfer size. It uses
sixteen intervals per power-of-two size range, with a minimum interval width of
16 bytes. Within a preparation batch, the largest compatible payload in each
interval represents that interval. The smallest representative and the largest
representative in each power-of-four size band race the complete eligible
catalog. Other representatives race
the union of the best eight configurations from each bracketing full search.
Every remaining shape adopts its interval representative's complete winning
configuration, including the transport and launch geometry. On a cache miss,
it binds that configuration and validates every benchmark input producer on
all ranks before saving the selection, with no timed race.
The complete catalog remains available for program discovery and rank agreement.

Calibration shortlists are saved with their representative selections. Exact
selections include the calibration's program-aware cache identity and retained
choices, so a changed calibration cannot reuse an incompatible selection. Fine
interval records include their bracketing or interval representative's decisions.
Request labels do not affect cache identity. Calibration groups retain the ordered ranks,
GPU identities, eligible configurations, layouts, aliasing, and owner geometry.
Audit effort measures the unrestricted catalog. Explicit pins and disabled
tuning bypass calibration. Caller-supplied benchmark contexts and dependent
requests do not share built-in calibration. Requests retaining measured finalists
preserve their complete family catalogs. Fused arithmetic and DCP operations
retain their shape-specific searches. Check performance within each bucket
against audit races when changing transport kernels or bucket boundaries.

Hierarchical collectives, DMA rings, FP8 sequence-parallel transports, paired gathers,
and owner/top-k exchanges retain their native fixed declarations. Transport
codecs with different arithmetic are not interchangeable tuning candidates.

`PCIeTwoShotBF16.all_reduce` accepts a disjoint output or the exact input
storage span. Reduce-scatter consumes the input into a fixed private shard
before same-stream all-gather writes the output; peers consume IPC copies.
Partial overlaps remain rejected. Standalone reduce-scatter and all-gather
require disjoint outputs. All operations and graph replays sharing one runtime
must remain serialized.

On 16 ranks, static gather tuning also measures acknowledged recursive doubling with
the higher rank bits first (XOR distances 4, 8, 1, 2). When each group of four
consecutive ranks shares a physical switch, this order sends three shard-equivalents
across switches per rank. The lower-bit-first order sends twelve. Both orders
return the same rank-major rows and share the ordinary per-peer generations.
The tuning trial chooses the order for its installed rank placement.

For four rows on sixteen ranks, `all_gather_pair_kimi_topk` accepts caller-owned
`out_logits` in addition to the latent, selected weights and expert IDs. The
prepared plan measures the paired exchange plus standalone selection against a
single row CTA that fetches both fields and selects the experts. Every rank
participates in the same configuration. Both candidates write all gathered
logits and preserve deterministic lower-ID ties.

### Four-row reduction in a Triton consumer

An SM120 tensor-parallel group can reduce four contiguous BF16 rows of width
3,584 or 7,168 before running a row-local Triton epilogue in the same CTA.
`island_row.prepared_rows(plan, rows=4, width=width, threads=threads)`
returns a layout for an eligible prepared IslandRS owner. The layout borrows
the local slab and the six remote mappings that the owner already holds.
Every participating rank must enter its channel serially in the same order.

The enclosing Triton kernel launches `layout.grid`. It calls
`island_row.reduce_rows` on every physical CTA, passing the aligned input and
output tensors, the tuple from `layout.arguments(source, destination)`, and the
rank and width specialization. CTA indices 0–3 write their complete reduced
row before entering the local epilogue. The other CTAs still advance their bank
counters. Both reduction stages add four FP32 operands from positive zero
in rank order and round once to BF16. Interleaved ordinary and fused calls keep
the native generation and bank protocol. Preparation can measure this combined
operation against the selected ordinary reduction and the same epilogue.

## Benchmark and synchronization contract

The `_benchmark.py` fixtures allocate private operands, generate and reset inputs,
capture candidate calls, and check arithmetic before timing. They preserve
declared tensor layouts and residual aliasing. Gather fixtures change arbitrary
byte patterns, poison their destinations, and compare exact output and input
bytes, including NaN payloads and FP32 values transported through BF16 views.
A caller can supply an explicit `benchmark_call` for a different representative
execution context.

All ranks compile and bind a candidate before any rank primes it. Each benchmark
producer completes validation on every rank before the next producer runs. A
validation error or cancellation stops the group at that boundary. Each rank
queues one sample behind a stream gate; the coordinator releases it after every
rank reports readiness. An untimed replay precedes input production and the
timed replay to absorb rank-release skew. Each sample drains before submission
of the next sample. Corresponding event samples reduce with **maximum across
ranks** before repeat sizing, elimination, and winner selection. The tournament
retains a champion and one challenger, bounding candidate operand residency to
two fixtures plus graph/event and cache-eviction storage.

`PreparationProgress.ready_collective_tuning` carries
`CollectiveTuningRequirement(key, ranks, step, payload)`. An embedding coordinator
exchanges these contributions in communicator rank order and resumes
`PreparationJob.advance(collective_tuning=contributions)`. All members must report
the same key, ranks, and step. Query signatures reject inconsistent shapes or
arithmetic before candidate entry. Cancellation is agreed at drained group
boundaries. Coordinators must propagate preparation exceptions to every member.

Collective candidates execute on the whole group; they are not divided among
the individual-kernel tuner's candidate shards. Independent TP replicas tune
their own communication group. The selection key includes the ordered GPU UUIDs,
rank-local query/program identities, and measurement policy. A valid cached
record from any member is adopted by the group. Cache-only mode fails if none
has a record. Interrupted collective races restart; completed selections reuse
the ordinary selection cache.

The selection key does not describe PCIe links or peer-access policy. Use a
fresh tuning-cache directory after changing those properties on the same
ordered GPUs. The IslandRS peer schedule remains fixed to its four-island rank
layout; launch tuning does not discover arbitrary topologies.

## Validation

Run the standalone group test with two or four visible peer-accessible SM120
GPUs:

```sh
B12X_RUN_COLLECTIVE_TUNING_TEST=1 python -m pytest -q \
  tests/experimental/b12x/comm/test_collective_tuning_gpu.py
```

The test exercises duplicate declarations, cache reuse with a missing member
cache, one-rank cancellation, inconsistent shapes, cache-only misses, and graph
replay. `benchmarks/experimental/b12x/benchmark_pcie_allreduce_transports.py`
compares explicit transports and the static dispatch policy with CUDA graphs.

With sixteen visible SM120 GPUs, run
`tests/experimental/b12x/comm/test_island_rs_tuning_gpu.py` with the same test
opt-in. It checks interval selection, recovery from a missing member cache,
validation-failure propagation, and allocation-stable mixed-shape graph replay
with changing inputs.

## Specialized column gathers

Phased gathers send the complete local input to one destination
at a time. They reuse the DCP channel's two staging banks and signals, and
launch one eight-CTA cluster. Every thread fences its phase writes before the
cluster advances, including threads with no remaining input. The
`selected_phased_barrier` transport also synchronizes peer ranks after each
destination; `selected_phased_push` uses one peer barrier after all sends.
Both assemble exact token-major output from local staging after peer completion.

These transports require `block_limit=8`, `active_blocks=8`, and
`submode="barrier"`. Materialization checks the compiled kernel's cluster
residency before capture. Other active/grid combinations are rejected; the
ordinary gather defaults do not change. For an established `DcpAllToAll`
channel and matching contiguous input/output tensors, declare a phased plan
with explicit geometry:

```python
from dataclasses import replace
from b12x.comm import pcie
from b12x.preparation import FrozenMapping

query = pcie.query_from_runtime(
    runtime,
    surface="DcpAllToAll.all_gather_heads",
    call=dict(local_input=source, out=output, threads=128, block_limit=8),
)
query = replace(query, call=FrozenMapping(dict(
    query.call,
    gather_transport="selected_phased_barrier",
    submode="barrier",
    active_blocks=8,
)))
plan = pcie.plan(query, runtime=runtime)
```

Submit this plan through the ordinary collective preparation session. Selected
barrier profiles and ordinary captured DCP calls share the channel's bank origin
and device epoch. All ranks must use the same
profile and serialize calls sharing a channel.

`all_gather_varlen` gathers live prefixes of contiguous uint8 rows at DCP2/4.
Its device int64 counts must be identical on every rank, multiples of 16, and
within the row capacity; invalid counts trap before communication and invalidate
the CUDA context. Inconsistent counts can hang peers. The prepared
bank must be a multiple of 16 KiB and fit at or below `exchange_offset` when
word-exchange packets are reserved, otherwise within `slot_bytes`. Calls on one
owner must be serialized: byte gathers share epochs and staging banks with its
other DCP calls.

The `selected_ring` profile uses one fixed outgoing neighbor and
one incoming neighbor. The eight CTAs divide a flattened sequence of 16-byte
input packs, so a warp may span row boundaries.
Each CTA retains the same pack ownership for every rank's shard. It sends the
local shard once, then loads each received pack once to write both its token-major
output position and the outgoing neighbor's staging slot. The final received
shard only writes output. Ready and acknowledgment records reuse the channel's
per-CTA, per-peer counters; ordinary calls remain pairwise-compatible with those
counters. Forwarded packs occupy distinct source slots. Each CTA releases output
and forwarding stores before acknowledging received storage. At most two ready
notifications remain outstanding. Before a ready generation reuses a parity slot,
the control thread waits for and acquires the downstream acknowledgment from two
generations earlier. The final downstream acknowledgment drains the operation
before its channel epoch advances. Staging banks and operation epochs retain the
same layout.

Ring declarations admit TP4/8/16; recursive declarations require DCP8 or TP16.
Both require `block_limit=8`, `active_blocks=8`, and `submode="barrier"` (or
16 CTAs for DCP16 ring); they use no cluster. Ring accepts `threads=128` or
`threads=512`; recursive profiles require `threads=128`. Set
`gather_transport="selected_ring"` in the declaration above to select the ring.
The `selected_recursive` profile exchanges one, two, four, then eight rank
segments with XOR partners. Each CTA forwards its own completed output segments
and acknowledges their consumption. The `selected_recursive_ready` profile
omits these acknowledgments. Its two staging banks protect received values
until the intervening serialized collective requires every rank to finish the
preceding call before the same bank can be reused.
