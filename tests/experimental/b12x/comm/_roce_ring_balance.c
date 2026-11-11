// Per-directed-link traffic of one collective on a closed four-rank ring. Every
// one-shot payload, reduce-scatter shard, all-gather shard and variable-length
// contribution is delivered per destination, so the four kinds share one route table
// and one count. Each rank posts its own writes; a packet whose target is two hops
// away lands in the next hop's mailbox and that hop forwards it, crossing a second
// directed link. All four ranks run in one process against the production proxy with
// a fake verbs provider, and the payload writes are summed per directed link.
//
// With the tie broken by rank parity both cables carry two shards per collective;
// before that change the clockwise links carried three and the counter-clockwise
// links one, which the report mode still prints.
#define _GNU_SOURCE
#include <infiniband/verbs.h>
#include <stdint.h>
#include ROCE_PROXY_SOURCE

#define WORLD 4
#define REGION_BYTES (16u << 20)
#define SLOT_BYTES (1u << 20)
#define SHARD_CAP 32768u
#define SEND_OFF ((8u << 20) + 512u * 1024u)
#define PIPE_SEND_OFF (9u << 20)
#define PIPE_RECV_OFF (10u << 20)
#define PIPE_FLAG_OFF (11u << 20)
#define CTRL_OFF (12u << 20)
#define VARLEN_RECV_OFF (13u << 20)
#define VARLEN_FLAG_OFF (14u << 20)
#define VARLEN_BYTES 32768u
#define ONESHOT_BYTES 65536u
#define TOTAL_PACKS 2048u
#define CHUNK_PACKS 512u
#define CQ_SIZE 4096
// Ops run back to back on one set of lanes, so each seq is distinct and rises.
#define SEQ_ONESHOT 1u
#define SEQ_RS 2u
#define SEQ_AG 3u
#define SEQ_VL 4u
#define MAX_PASSES 64

enum { OP_ONESHOT, OP_REDUCE_SCATTER, OP_ALL_GATHER, OP_VARLEN, OP_COUNT };
static const char* const op_names[OP_COUNT] = {"one-shot", "reduce-scatter", "all-gather",
                                               "varlen"};

// One device, queue pair and completion queue per rank, so a completion only ever
// retires a write its own rank posted.
static struct ibv_context device[WORLD];
static struct ibv_mr mr;
static struct ibv_cq cq[WORLD];
static struct ibv_wc completions[WORLD][CQ_SIZE];
static struct ibv_qp qps[WORLD][WORLD];
static uint8_t regions[WORLD][REGION_BYTES];
static uint8_t* ring_areas[WORLD];
static roce_ctx_t ctx[WORLD];
static unsigned cq_head[WORLD], cq_tail[WORLD];
static uint64_t posts;
// Payload writes per (posting rank, receiving rank), summed over the four ranks.
static long link[WORLD][WORLD];
static int failures;

static void require(int condition, const char* what) {
  if (condition) return;
  fprintf(stderr, "failed: %s\n", what);
  failures++;
}

// The queue pair names both endpoints: the rank that posts it and the peer it reaches.
// The fake provider copies each write where a NIC would, so mailbox descriptors and
// forwarded payloads survive for the next hop's relay.
static int qp_ends(struct ibv_qp* qp, int* from, int* to) {
  for (int r = 0; r < WORLD; r++)
    for (int p = 0; p < WORLD; p++)
      if (qp == &qps[r][p]) {
        *from = r;
        *to = p;
        return 1;
      }
  return 0;
}

static int post_send(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad) {
  (void)bad;
  int from = -1, to = -1;
  if (!qp_ends(qp, &from, &to)) abort();
  for (; wr; wr = wr->next) {
    uint64_t dst = (uint64_t)wr->wr.rdma.remote_addr;
    int payload = 0;
    for (int i = 0; i < wr->num_sge; i++) {
      uint32_t len = wr->sg_list[i].length;
      // A descriptor body (120 bytes) opens a mailbox packet; the packs behind it are
      // the shard. Flags (4 bytes) and commit serials (8) are not transfers.
      if (len > sizeof(ring_descriptor_t) - 8) payload = 1;
      memcpy((void*)(uintptr_t)dst, (const void*)(uintptr_t)wr->sg_list[i].addr, len);
      dst += len;
    }
    if (payload) link[from][to]++;
    posts++;
    if (wr->send_flags & IBV_SEND_SIGNALED) {
      if (cq_tail[from] - cq_head[from] == CQ_SIZE) abort();
      completions[from][cq_tail[from]++ % CQ_SIZE] =
          (struct ibv_wc){.wr_id = wr->wr_id, .status = IBV_WC_SUCCESS};
    }
  }
  return 0;
}

static int poll_cq(struct ibv_cq* ibcq, int count, struct ibv_wc* wc) {
  int r = (int)((const struct ibv_context*)ibcq->context - device);
  int n = 0;
  while (n < count && cq_head[r] != cq_tail[r]) wc[n++] = completions[r][cq_head[r]++ % CQ_SIZE];
  return n;
}

static int any_pending(void) {
  for (int r = 0; r < WORLD; r++)
    for (int o = 0; o < WORLD; o++)
      for (int rail = 0; rail < ctx[r].n_rail; rail++)
        for (int slot = 0; slot < RING_SLOT_COUNT(WORLD); slot++)
          if (ctx[r].ring_pending[o][rail][slot]) return 1;
  return 0;
}

// Service every rank until no completion, forward or acknowledgment is left.
static int pump(void) {
  for (int pass = 0; pass < MAX_PASSES; pass++) {
    uint64_t before = posts;
    for (int r = 0; r < WORLD; r++) {
      if (drain_all(&ctx[r])) {
        fprintf(stderr, "rank %d: %s\n", r, ctx[r].err);
        failures++;
        return -1;
      }
    }
    if (posts == before && !any_pending()) return 0;
  }
  fprintf(stderr, "the ring did not settle in %d passes\n", MAX_PASSES);
  failures++;
  return -1;
}

// Every rank keeps a region and a mailbox block of its own; each rank's peers are the
// other three, addressed by region base and by ring base.
static void setup(void) {
  for (int r = 0; r < WORLD; r++) {
    device[r].ops.post_send = post_send;
    device[r].ops.poll_cq = poll_cq;
    cq[r].context = &device[r];
    ring_areas[r] = calloc(1, roce_ring_bytes(WORLD));
    if (ring_areas[r] == NULL) abort();
  }
  for (int r = 0; r < WORLD; r++) {
    roce_ctx_t* c = &ctx[r];
    memset(c, 0, sizeof(*c));
    c->world = WORLD;
    c->rank = r;
    c->n_rail = 1;
    c->n_hca = 1;
    c->n_lane = 1;
    c->ring_open = 0;
    c->region = regions[r];
    c->region_bytes = REGION_BYTES;
    c->slot_bytes = SLOT_BYTES;
    c->recv_off = 0;
    c->flag_off = 8u << 20;
    c->send_off = SEND_OFF;
    c->pipe_bytes = 1u << 20;
    c->pipe_send_off = PIPE_SEND_OFF;
    c->pipe_recv_off = PIPE_RECV_OFF;
    c->pipe_flag_off = PIPE_FLAG_OFF;
    c->shard_cap = SHARD_CAP;
    c->ctrl_off = CTRL_OFF;
    c->varlen_bytes = VARLEN_BYTES;
    c->varlen_recv_off = VARLEN_RECV_OFF;
    c->varlen_flag_off = VARLEN_FLAG_OFF;
    c->ring = ring_areas[r];
    c->ring_sources = ring_source_mask(c);
    for (int origin = 0; origin < WORLD; origin++)
      c->ring_destinations[origin] = ring_destination_mask(c, origin);
    c->hca[0].cq = &cq[r];
    c->hca[0].mr = &mr;
    c->ring_mr[0] = &mr;
    // The relay never runs out of room in this trace, so start it as running.
    atomic_store(&c->running, 1);
    for (int p = 0; p < WORLD; p++) {
      c->peer_addr[p] = (uintptr_t)regions[p];
      c->ring_addr[p] = (uintptr_t)ring_areas[p];
      c->rail_hca[p][0] = 0;
      if (p == r) continue;
      qps[r][p].context = &device[r];
      c->hca[0].qp[p][0] = &qps[r][p];
    }
  }
}

// Ring service rejects a packet whose previous hop is not exactly one step farther from
// the target than the rank serving it (ring_distance). Walk every route the tie rule
// chooses on every world size and check each hop is one cable step that shortens that
// distance by exactly one.
static void require_shortest_routes(void) {
  roce_ctx_t probe = {0};
  for (probe.world = 2; probe.world <= ROCE_MAX_PEERS; probe.world++)
    for (int origin = 0; origin < probe.world; origin++)
      for (int target = 0; target < probe.world; target++) {
        if (target == origin) continue;
        int hop = origin;
        for (int step = 0; step < probe.world && hop != target; step++) {
          int next = ring_next_at(&probe, hop, target);
          require(next == (hop + 1) % probe.world || next == (hop + probe.world - 1) % probe.world,
                  "a hop is one cable step");
          require(ring_distance(&probe, hop, target) == ring_distance(&probe, next, target) + 1,
                  "a hop shortens the route by exactly one");
          hop = next;
        }
        require(hop == target, "every route reaches its target");
      }
}

static void post_op_kind(int op) {
  // Fill every region first: a rank posts into its peers' regions, so refilling one
  // between posts would erase what an earlier rank delivered.
  for (int r = 0; r < WORLD; r++) memset(ctx[r].region, 0x5a + r, REGION_BYTES);
  for (int r = 0; r < WORLD; r++) {
    roce_ctx_t* c = &ctx[r];
    int rc;
    switch (op) {
      case OP_ONESHOT:
        rc = post_op(c, SEQ_ONESHOT, ONESHOT_BYTES);
        break;
      case OP_REDUCE_SCATTER:
        // One chunk covers peer p's whole shard, so one post per destination.
        rc = post_pipe_chunk(c, SEQ_RS, 0, 0, TOTAL_PACKS, CHUNK_PACKS);
        break;
      case OP_ALL_GATHER:
        rc = post_pipe_chunk(c, SEQ_AG, 1, 0, TOTAL_PACKS, CHUNK_PACKS);
        break;
      default: {
        // The kernel staged chunk 0 of this rank's contribution.
        volatile uint32_t* ctrl = (volatile uint32_t*)(c->region + c->ctrl_off);
        __atomic_store_n(&ctrl[CTRL_READY_VL], SEQ_VL, __ATOMIC_RELEASE);
        rc = run_varlen_op(c, SEQ_VL, VARLEN_BYTES, VARLEN_BYTES / 16, 1);
        break;
      }
    }
    if (rc != 0) {
      fprintf(stderr, "rank %d %s: %s\n", r, op_names[op], c->err);
      failures++;
      return;
    }
  }
  pump();
}

int main(int argc, char** argv) {
  int report = argc > 1 && strcmp(argv[1], "report") == 0;
  require_shortest_routes();
  setup();
  for (int op = 0; op < OP_COUNT; op++) {
    memset(link, 0, sizeof link);
    post_op_kind(op);
    long total = 0;
    printf("%s:", op_names[op]);
    for (int from = 0; from < WORLD; from++)
      for (int to = 0; to < WORLD; to++) {
        if (link[from][to] == 0) continue;
        printf(" %d->%d=%ld", from, to, link[from][to]);
        total += link[from][to];
      }
    printf(" total=%ld\n", total);
    if (report) continue;
    require(total == 16, "a collective crosses sixteen directed links");
    for (int from = 0; from < WORLD; from++)
      for (int to = 0; to < WORLD; to++)
        if ((from + 1) % WORLD == to || (to + 1) % WORLD == from)
          require(link[from][to] == 2, "every cable direction carries two shards per collective");
  }
  for (int r = 0; r < WORLD; r++) free(ring_areas[r]);
  if (report) return 0;
  if (!failures) puts("ring link balance: ok");
  return failures != 0;
}
