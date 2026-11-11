"""Both cables of a four-rank closed ring carry the same traffic.

Every collective delivers one shard per destination, so the reduce-scatter,
all-gather, one-shot and variable-length routes all follow ``ring_next_at``. The
driver runs four ranks in one process against the production proxy source with a
fake verbs provider and counts the payload writes each rank posts per directed
link, including the hop a two-hop packet is forwarded over.
"""

import re
import subprocess

LINK = re.compile(r"(\d)->(\d)=(\d+)")


def test_every_ring_link_carries_the_same_traffic(proxy_driver):
    result = subprocess.run(
        [str(proxy_driver)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[-1] == "ring link balance: ok"
    routes = lines[:-1]
    assert len(routes) == 4, result.stdout
    for line in routes:
        counts = {(int(a), int(b)): int(n) for a, b, n in LINK.findall(line)}
        # Four ranks, two hops each way: every one of the eight cable directions
        # carries two shards, and nothing else carries any.
        assert len(counts) == 8, line
        assert set(counts.values()) == {2}, line
        assert sum(counts.values()) == 16, line
