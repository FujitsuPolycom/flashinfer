// The all-gather relay on a closed four-rank ring. Rank r posts its reduced shard to
// its two neighbors only; r + 1 forwards rank r's chunk to r + 2 from the row it
// received. With two chunks the neighbors split the work by chunk parity, so both
// cables carry the same traffic. The four ranks run against the production proxy with
// a fake verbs provider, every payload write is counted by the region it lands in (a
// peer's row versus a relay mailbox), and a far rank must finish with the origin's
// bytes and ready flag. The plan is also required to be refused on every ring shape
// whose forwards would not be direct writes, and the production pipeline loop runs on
// one rank at a time with an odd chunk count, where one neighbor finishes first.
#define _GNU_SOURCE
#include <infiniband/verbs.h>
#include <stdint.h>
#include ROCE_PROXY_SOURCE

#define WORLD 4
#define REGION_BYTES (16u << 20)
#define SLOT_BYTES (1u << 20)
#define SHARD_CAP 32768u
#define PIPE_SEND_OFF (9u << 20)
#define PIPE_RECV_OFF (10u << 20)
#define PIPE_FLAG_OFF (11u << 20)
#define CTRL_OFF (12u << 20)
#define CTRL_BYTES 16384u
#define ONESHOT_PACKS 1024u
#define CQ_SIZE 4096
#define SEQ_ONE_CHUNK 1u
#define SEQ_TWO_CHUNKS 2u
#define PIPE_CHUNKS 3u
// Three chunks of 96 packs cover the 256-pack shard of four ranks: no chunk is empty
// and the third is shorter than the other two.
#define PIPE_CHUNK_PACKS (ONESHOT_PACKS * 3u / (WORLD * 8u))
#define SEQ_PIPE_BASE 32u

static struct ibv_context device[WORLD];
static struct ibv_mr mr;
static struct ibv_cq cq[WORLD];
static struct ibv_wc completions[WORLD][CQ_SIZE];
static struct ibv_qp qps[WORLD][WORLD];
static uint8_t regions[WORLD][REGION_BYTES];
static uint8_t* ring_areas[WORLD];
static roce_ctx_t ctx[WORLD];
static unsigned cq_head[WORLD], cq_tail[WORLD];
// Payload writes per (posting rank, receiving rank), split by where they landed.
static long row[WORLD][WORLD];
static long mailbox[WORLD][WORLD];
// Payload writes each rank posted, and the same count taken before any forward.
static long own[WORLD], posted[WORLD];
static int failures;

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

// A mailbox write leaves its descriptor and payload in the receiving rank's ring block;
// every other payload write lands in a receive row.
static int lands_in_mailbox(uint64_t dst, int* peer) {
  for (int p = 0; p < WORLD; p++)
    if (dst >= (uint64_t)(uintptr_t)ring_areas[p] &&
        dst < (uint64_t)(uintptr_t)ring_areas[p] + roce_ring_bytes(WORLD)) {
      *peer = p;
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
      if (len > sizeof(ring_descriptor_t) - 8) payload = 1;
      memcpy((void*)(uintptr_t)dst, (const void*)(uintptr_t)wr->sg_list[i].addr, len);
      dst += len;
    }
    if (payload) {
      int peer = -1;
      if (lands_in_mailbox((uint64_t)wr->wr.rdma.remote_addr, &peer))
        mailbox[from][peer]++;
      else
        row[from][to]++;
      own[from]++;
    }
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

static void require(int condition, const char* what) {
  if (condition) return;
  fprintf(stderr, "failed: %s\n", what);
  failures++;
}

// The plan a four-rank ring derives: the far peer goes through the clockwise neighbor,
// and each neighbor forwards the other one's chunks.
static void ring_plan(int rank, int* plan) {
  for (int i = 0; i < 2 * WORLD; i++) plan[i] = -1;
  plan[(rank + 2) % WORLD] = (rank + 1) % WORLD;
  plan[WORLD + (rank + WORLD - 1) % WORLD] = (rank + 1) % WORLD;
  plan[WORLD + (rank + 1) % WORLD] = (rank + WORLD - 1) % WORLD;
}

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
    c->send_off = 8u << 20;
    c->pipe_bytes = ONESHOT_PACKS * 16u;
    c->pipe_send_off = PIPE_SEND_OFF;
    c->pipe_recv_off = PIPE_RECV_OFF;
    c->pipe_flag_off = PIPE_FLAG_OFF;
    c->shard_cap = SHARD_CAP;
    c->ctrl_off = CTRL_OFF;
    c->ring = ring_areas[r];
    c->ring_sources = ring_source_mask(c);
    for (int origin = 0; origin < WORLD; origin++)
      c->ring_destinations[origin] = ring_destination_mask(c, origin);
    c->hca[0].cq = &cq[r];
    c->hca[0].mr = &mr;
    c->ring_mr[0] = &mr;
    atomic_store(&c->running, 1);
    for (int p = 0; p < WORLD; p++) {
      c->peer_addr[p] = (uintptr_t)regions[p];
      c->ring_addr[p] = (uintptr_t)ring_areas[p];
      c->rail_hca[p][0] = 0;
      if (p == r) continue;
      qps[r][p].context = &device[r];
      c->hca[0].qp[p][0] = &qps[r][p];
    }
    int plan[2 * WORLD];
    ring_plan(r, plan);
    if (roce_set_relay(c, plan, 2 * WORLD) != 0) {
      fprintf(stderr, "rank %d relay: %s\n", r, c->err);
      abort();
    }
  }
}

static void clear_counts(void) {
  memset(row, 0, sizeof row);
  memset(mailbox, 0, sizeof mailbox);
  memset(own, 0, sizeof own);
  memset(posted, 0, sizeof posted);
}

static long sum_of(const long m[WORLD][WORLD]) {
  long n = 0;
  for (int a = 0; a < WORLD; a++)
    for (int b = 0; b < WORLD; b++) n += m[a][b];
  return n;
}

// Every rank stages its shard, posts each chunk once, and forwards the chunks it
// carries for its neighbors, the way run_pipe_op does.
static void all_gather(uint32_t seq, uint32_t total_packs, uint32_t chunk_packs, uint32_t chunks) {
  // Fill every region first: a rank posts into its peers' regions, so refilling one
  // between posts would erase what an earlier rank delivered.
  for (int r = 0; r < WORLD; r++) memset(ctx[r].region, 0x5a + r, REGION_BYTES);
  for (int r = 0; r < WORLD; r++)
    for (uint32_t chunk = 0; chunk < chunks; chunk++)
      if (post_pipe_chunk(&ctx[r], seq, 1, chunk, total_packs, chunk_packs) != 0) {
        fprintf(stderr, "rank %d post: %s\n", r, ctx[r].err);
        failures++;
        return;
      }
  for (int r = 0; r < WORLD; r++) {
    posted[r] = own[r];
    roce_ctx_t* c = &ctx[r];
    for (int s = 0; s < WORLD; s++) {
      if (c->relay_to[s] < 0) continue;
      for (uint32_t chunk = 0; chunk < chunks; chunk++) {
        if (!relay_forwards_chunk(c, s, chunk)) continue;
        if (!pipe_chunk_arrived(c, seq, s, chunk)) {
          fprintf(stderr, "rank %d: source %d chunk %u never arrived\n", r, s, chunk);
          failures++;
        } else if (forward_pipe_chunk(c, seq, s, chunk, total_packs, chunk_packs) != 0) {
          fprintf(stderr, "rank %d forward: %s\n", r, c->err);
          failures++;
        }
      }
    }
  }
}

// The far rank's row for origin `origin` holds the origin's shard, and its ready flag
// for that chunk is set.
static void require_far_rank_has_chunk(uint32_t seq, uint32_t origin, uint32_t chunk,
                                       uint32_t total_packs, uint32_t chunk_packs) {
  int far = (origin + 2) % WORLD;
  uint32_t first = (uint32_t)(((uint64_t)origin * total_packs) / WORLD);
  uint32_t lo = first + chunk * chunk_packs;
  uint32_t packs = chunk_packs;
  if (lo + packs > first + (total_packs / WORLD)) packs = first + (total_packs / WORLD) - lo;
  const uint8_t* sent = ctx[origin].region + PIPE_SEND_OFF + (uint64_t)(lo - first) * 16;
  const uint8_t* got = ctx[far].region + PIPE_RECV_OFF + ((uint64_t)origin * 2 + 1) * SHARD_CAP +
                       (uint64_t)(lo - first) * 16;
  require(memcmp(got, sent, (size_t)packs * 16) == 0 && got[0] == (uint8_t)(0x5a + origin),
          "the far rank holds that origin's chunk");
  require(pipe_chunk_arrived(&ctx[far], seq, origin, chunk), "the far rank's ready flag is set");
}

// Every forward the relay posts is a direct write to a cable neighbor, so the plan only
// describes the mesh and a closed ring of four ranks. A line and every other ring size
// reach the far peer through the ring mailbox; each must be refused with the relay left
// off, not accepted with ring_fanout silently off.
static void refuse_other_ring_shapes(void) {
  roce_ctx_t* c = &ctx[0];
  int plan[2 * WORLD];
  ring_plan(0, plan);
  const int was_open = c->ring_open, was_relay = c->relay;
  c->relay = 0;
  c->ring_open = 1;
  require(roce_set_relay(c, plan, 2 * WORLD) != 0, "a relay plan is refused on an open line");
  require(!c->relay && c->err[0] != '\0', "the refusal names a reason and leaves the relay off");
  c->ring_open = was_open;

  // The shape check runs before any plan entry is read, so the eight-rank context only
  // has to exist: no masks or routes are built for it.
  uint8_t* area = calloc(1, roce_ring_bytes(8));
  if (area == NULL) abort();
  roce_ctx_t eight = *c;
  eight.world = 8;
  eight.ring = area;
  eight.ring_open = 0;
  eight.relay = 0;
  int plan8[16];
  for (int i = 0; i < 16; i++) plan8[i] = -1;
  plan8[2] = plan8[8 + 7] = 1;
  plan8[8 + 1] = 7;
  require(roce_set_relay(&eight, plan8, 16) != 0, "a relay plan is refused on a ring of eight");
  require(!eight.relay, "a refused plan leaves the relay off on a ring of eight");
  free(area);

  // The four-rank closed ring still accepts the same plan, so the refusal is about the
  // shape.
  c->relay = 0;
  require(roce_set_relay(c, plan, 2 * WORLD) == 0, "the four-rank ring still accepts its plan");
  require(c->relay, "the accepted plan turns the relay on");
  c->relay = was_relay;
}

// run_pipe_op on one rank, three chunks. Both neighbors have already posted their
// chunks, so the rank under test only posts its own and forwards what arrived. Three is
// odd: the counter-clockwise neighbor's chunks reach it as 0 and 2 and the clockwise
// neighbor's as 1, so the two sources finish forwarding at different times, and the
// accounting has to drop a source on a chunk the other neighbor carries as well as on a
// completed forward.
static void pipeline_three_chunks(void) {
  const uint32_t total = ONESHOT_PACKS, chunk_packs = PIPE_CHUNK_PACKS;
  for (int r = 0; r < WORLD; r++) {
    const uint32_t seq = SEQ_PIPE_BASE + (uint32_t)r;
    for (int i = 0; i < WORLD; i++) {
      memset(ctx[i].region, 0x5a + i, REGION_BYTES);
      memset(ctx[i].region + CTRL_OFF, 0, CTRL_BYTES);
    }
    // Each neighbor stages its shard and posts every chunk, the way its kernel does.
    for (int s = 0; s < WORLD; s++) {
      if (s == r) continue;
      for (uint32_t chunk = 0; chunk < PIPE_CHUNKS; chunk++)
        if (post_pipe_chunk(&ctx[s], seq, 1, chunk, total, chunk_packs) != 0) {
          fprintf(stderr, "rank %d source %d post: %s\n", r, s, ctx[s].err);
          failures++;
        }
    }
    clear_counts();
    ctx[r].relay_forwarded = 0;
    volatile uint32_t* ctrl = (volatile uint32_t*)(ctx[r].region + CTRL_OFF);
    for (uint32_t chunk = 0; chunk < PIPE_CHUNKS; chunk++) ctrl[CTRL_READY_AG + chunk] = seq;
    if (run_pipe_op(&ctx[r], seq, total * 16, chunk_packs, PIPE_CHUNKS, PIPE_PHASE_AG, 0) != 0) {
      fprintf(stderr, "rank %d pipeline: %s\n", r, ctx[r].err);
      failures++;
      continue;
    }
    const int clockwise = (r + 1) % WORLD, counter = (r + WORLD - 1) % WORLD;
    require(own[r] == 9, "a rank posts three chunks to two neighbors and three forwards");
    require(row[r][clockwise] == 3 + 2,
            "the clockwise link carries its chunks and even far chunks");
    require(row[r][counter] == 3 + 1,
            "the counter-clockwise link carries its chunks and the odd one");
    require(sum_of(row) == 9 && sum_of(mailbox) == 0,
            "nine transfers leave one rank and no chunk takes a mailbox");
    require(ctx[r].relay_forwarded == 3, "the rank forwards three chunks");
    for (int s = 0; s < WORLD; s++) {
      if (ctx[r].relay_to[s] < 0) continue;
      for (uint32_t chunk = 0; chunk < PIPE_CHUNKS; chunk++)
        if (relay_forwards_chunk(&ctx[r], s, chunk))
          require_far_rank_has_chunk(seq, (uint32_t)s, chunk, total, chunk_packs);
    }
  }
}

int main(int argc, char** argv) {
  setup();
  // No argument runs every check; a section name runs that one so a test can name it.
  const char* only = argc > 1 ? argv[1] : NULL;
  const int all = only == NULL, chunks = all || strcmp(only, "chunks") == 0;
  const int refusals = all || strcmp(only, "refusals") == 0;
  const int pipeline = all || strcmp(only, "pipeline") == 0;
  if (!chunks && !refusals && !pipeline) {
    fprintf(stderr, "unknown check %s\n", only);
    return 2;
  }
  if (!chunks) goto other_checks;

  // One chunk: two direct writes per rank, one forward each, twelve transfers, and no
  // mailbox packet at all.
  clear_counts();
  all_gather(SEQ_ONE_CHUNK, ONESHOT_PACKS, ONESHOT_PACKS / WORLD, 1);
  for (int r = 0; r < WORLD; r++) {
    require(posted[r] == 2, "a rank posts two direct writes");
    require(row[r][(r + 1) % WORLD] == 2, "the clockwise link carries the shard and the forward");
    require(row[r][(r + WORLD - 1) % WORLD] == 1, "the counter-clockwise link carries the shard");
    require(row[r][(r + 2) % WORLD] == 0, "a rank writes no far row itself");
  }
  require(sum_of(row) == 12 && sum_of(mailbox) == 0,
          "twelve all-gather transfers, not sixteen, and no mailbox");
  for (int origin = 0; origin < WORLD; origin++)
    require_far_rank_has_chunk(SEQ_ONE_CHUNK, origin, 0, ONESHOT_PACKS, ONESHOT_PACKS / WORLD);

  // Two chunks: the clockwise neighbor carries the even chunk and the counter-clockwise
  // one the odd chunk, so each direction carries half.
  clear_counts();
  all_gather(SEQ_TWO_CHUNKS, ONESHOT_PACKS, ONESHOT_PACKS / WORLD / 2, 2);
  for (int r = 0; r < WORLD; r++) {
    roce_ctx_t* c = &ctx[r];
    int clockwise = (r + 1) % WORLD, counter = (r + WORLD - 1) % WORLD;
    require(relay_forwards_chunk(c, counter, 0), "the even chunk of the far side is forwarded");
    require(!relay_forwards_chunk(c, counter, 1), "an odd chunk is left to the other neighbor");
    require(!relay_forwards_chunk(c, clockwise, 0), "the clockwise neighbor's chunk is not ours");
    require(relay_forwards_chunk(c, clockwise, 1), "the odd chunk of the near side is forwarded");
    require(posted[r] == 4, "a rank posts two direct writes per chunk");
    require(row[r][clockwise] == 3 && row[r][counter] == 3,
            "each cable direction carries three transfers over two chunks");
  }
  require(sum_of(row) == 24 && sum_of(mailbox) == 0,
          "two chunks cross twenty-four directed links and no mailbox");
  for (int origin = 0; origin < WORLD; origin++)
    for (uint32_t chunk = 0; chunk < 2; chunk++)
      require_far_rank_has_chunk(SEQ_TWO_CHUNKS, origin, chunk, ONESHOT_PACKS,
                                 ONESHOT_PACKS / WORLD / 2);

other_checks:
  if (refusals) refuse_other_ring_shapes();
  if (pipeline) pipeline_three_chunks();

  for (int r = 0; r < WORLD; r++) free(ring_areas[r]);
  if (!failures) puts("ring partner relay: ok");
  return failures != 0;
}
