"""Search frontiers behind one interface: FIFO queue, LIFO stack, priority queue.

Every frontier supports add, pop, membership by board, and len. The priority
frontier also supports decrease_key.
"""

from __future__ import annotations

import heapq
import itertools
from collections import deque


class QueueFrontier:
    """FIFO frontier for breadth-first search."""

    def __init__(self):
        self._items = deque()
        self._boards = set()

    def add(self, node, priority=None):
        self._items.append(node)
        self._boards.add(node.board)

    def pop(self):
        node = self._items.popleft()
        self._boards.discard(node.board)
        return node

    def __contains__(self, board):
        return board in self._boards

    def __len__(self):
        return len(self._items)


class StackFrontier(QueueFrontier):
    """LIFO frontier for depth-first search."""

    def pop(self):
        node = self._items.pop()
        self._boards.discard(node.board)
        return node


class PriorityFrontier:
    """Min-priority frontier with decrease-key, for UCS, greedy, and A*.

    heapq has no decrease-key, so this uses lazy deletion: decrease_key pushes
    a new heap entry and marks the old one stale, and pop skips stale entries.
    `_live` maps each board in the frontier to its current entry, so len()
    counts live states only. Without it, a heap holding only stale entries
    would look non-empty and pop would fail.

    Equal priorities pop in insertion order, which preserves UDLR order.
    """

    _STALE = object()

    def __init__(self):
        self._heap = []
        self._live = {}
        self._counter = itertools.count()

    def add(self, node, priority):
        entry = [priority, next(self._counter), node]
        self._live[node.board] = entry
        heapq.heappush(self._heap, entry)

    def priority(self, board):
        return self._live[board][0]

    def decrease_key(self, node, priority):
        """Replace the frontier entry for node.board if priority is lower.

        Returns True if the entry was replaced.
        """
        old = self._live[node.board]
        if priority >= old[0]:
            return False
        old[2] = self._STALE
        self.add(node, priority)
        return True

    def pop(self):
        while self._heap:
            _, _, node = heapq.heappop(self._heap)
            if node is not self._STALE:
                del self._live[node.board]
                return node
        raise IndexError("pop from empty frontier")

    def __contains__(self, board):
        return board in self._live

    def __len__(self):
        return len(self._live)
