"""All-gather on a four-rank ring forwards the far chunk through a neighbor.

The driver builds the four ranks in one process against the production proxy
source with a fake verbs provider, sets the ring relay plan on every rank, and
counts each payload write by where it lands: a receive row of a peer, or a relay
mailbox. A rank posts its shard to its two neighbors only and the neighbor
forwards it, so twelve transfers cross the ring instead of sixteen and no
all-gather chunk touches a mailbox.
"""

import subprocess

import pytest


@pytest.mark.parametrize("section", [None, "refusals", "pipeline"])
def test_ring_all_gather_forwards_through_a_neighbor(proxy_driver, section):
    args = [str(proxy_driver)] + ([section] if section else [])
    result = subprocess.run(args, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "ring partner relay: ok"
