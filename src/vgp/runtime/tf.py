"""Transform tree with time buffering -- our small tf2.

WHAT PROBLEM THIS SOLVES
------------------------
The camera sees the object. The gripper must reach it. Between those two facts
sit five transforms, each produced by a different component at a different rate:

    base_link -> tool0          from joint encoders,   ~500 Hz
    tool0     -> camera_link    from hand-eye calib,   static
    camera_link -> camera_optical  a fixed convention,  static
    camera_optical -> obj/cube  from PnP,               ~15 Hz

Without a tree, every consumer that needs base->object hand-composes those four,
and each one re-derives the chain, picks its own direction convention, and gets
to make its own off-by-one-inverse mistake. With a tree, there is ONE answer and
everyone asks for it by name:

    tf.lookup(BASE, "obj/cube", stamp)

WHY A TREE AND NOT A GENERAL GRAPH
----------------------------------
In a tree there is exactly one path between any two frames, so the answer is
unique. A general graph permits two paths that disagree -- which is not a
theoretical worry: it is precisely what happens when you have both a hand-eye
calibration and a fiducial-based estimate of the same camera pose, and they
differ by your calibration error. A tree forces you to decide which one is
authoritative instead of silently getting whichever path the search found first.
`set_transform` enforces the single-parent rule for this reason.

THE TIME PROBLEM, AND WHY WE INTERPOLATE
----------------------------------------
Encoders tick at 500 Hz, PnP produces a pose at 15 Hz, and the PnP result is
stamped when its IMAGE was captured -- typically 30-80 ms in the past. To place
that observation correctly we need where the arm was AT THAT INSTANT, not where
it is now. On a moving arm those differ by centimetres.

So the buffer keeps history and interpolates between the two samples bracketing
the requested stamp. This single detail is the difference between a pipeline
that works only when the robot is stationary and one that works while it moves.

WHY WE REFUSE TO EXTRAPOLATE
----------------------------
Asking for a stamp newer than any sample is an error here, never a silent
"use the latest". Extrapolating robot state is guessing about the future
position of a heavy machine; the honest response is to fail so the caller can
wait or abort. tf2 made the same call, and for the same reason.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from vgp.utils.transforms import SE3, interpolate

DEFAULT_BUFFER_S = 10.0


class TFLookupError(KeyError):
    """No path, or no valid time sample, between two frames."""


@dataclass
class _Edge:
    """One parent->child relationship, static or time-buffered."""

    parent: str
    child: str
    static: bool
    stamps: list[float] = field(default_factory=list)
    poses: list[SE3] = field(default_factory=list)
    _static_pose: SE3 | None = None

    def insert(self, stamp: float, pose: SE3, buffer_s: float) -> None:
        if self.static:
            self._static_pose = pose
            return
        # Samples usually arrive in order, so append-then-check beats a full
        # insort on the hot path; we only pay for sorting when they do not.
        if self.stamps and stamp < self.stamps[-1]:
            i = bisect_left(self.stamps, stamp)
            self.stamps.insert(i, stamp)
            self.poses.insert(i, pose)
        else:
            self.stamps.append(stamp)
            self.poses.append(pose)
        # Evict history older than the buffer window. Unbounded history is a
        # slow memory leak in a process meant to run for hours.
        cutoff = self.stamps[-1] - buffer_s
        drop = bisect_left(self.stamps, cutoff)
        if drop > 1:                      # keep one sample before the cutoff so
            del self.stamps[: drop - 1]   # a lookup at the window edge can still
            del self.poses[: drop - 1]    # interpolate rather than fail

    def at(self, stamp: float | None) -> SE3:
        if self.static:
            if self._static_pose is None:
                raise TFLookupError(f"static transform {self.parent}<-{self.child} never set")
            return self._static_pose
        if not self.stamps:
            raise TFLookupError(f"no samples for {self.parent}<-{self.child}")
        if stamp is None:                 # "latest" -- explicit opt-in only
            return self.poses[-1]
        if stamp > self.stamps[-1] + 1e-9:
            raise TFLookupError(
                f"lookup of {self.parent}<-{self.child} at t={stamp:.4f} is "
                f"{stamp - self.stamps[-1]:.4f}s beyond the newest sample "
                f"(t={self.stamps[-1]:.4f}). Refusing to extrapolate robot state."
            )
        if stamp < self.stamps[0] - 1e-9:
            raise TFLookupError(
                f"lookup of {self.parent}<-{self.child} at t={stamp:.4f} predates "
                f"the buffer (oldest t={self.stamps[0]:.4f}). Increase buffer_s "
                "or reduce pipeline latency."
            )
        i = bisect_left(self.stamps, stamp)
        if i < len(self.stamps) and abs(self.stamps[i] - stamp) < 1e-9:
            return self.poses[i]
        lo, hi = i - 1, i
        t0, t1 = self.stamps[lo], self.stamps[hi]
        s = (stamp - t0) / (t1 - t0)
        return interpolate(self.poses[lo], self.poses[hi], s)


class TFTree:
    def __init__(self, buffer_s: float = DEFAULT_BUFFER_S) -> None:
        self.buffer_s = buffer_s
        self._edges: dict[tuple[str, str], _Edge] = {}
        self._parent_of: dict[str, str] = {}
        # Undirected adjacency: an edge is traversable in reverse by inverting
        # the transform, which is what makes lookup(A, B) and lookup(B, A) both
        # work from a single stored relationship.
        self._adj: dict[str, set[str]] = {}

    # ------------------------------------------------------------- writing

    def set_transform(
        self, pose: SE3, stamp: float | None = None, static: bool = False
    ) -> None:
        """Insert `pose` (which must carry parent/child labels) into the tree."""
        if pose.parent is None or pose.child is None:
            raise ValueError(
                "TFTree requires labelled transforms; got an unlabelled SE3. "
                "Pass parent=/child= when constructing it."
            )
        if pose.parent == pose.child:
            raise ValueError(f"self-loop: frame '{pose.parent}' cannot parent itself")

        key = (pose.parent, pose.child)
        # Single-parent rule: this is what keeps it a TREE. Violating it is how
        # you get two disagreeing paths between the same pair of frames.
        prior = self._parent_of.get(pose.child)
        if prior is not None and prior != pose.parent:
            raise ValueError(
                f"frame '{pose.child}' already has parent '{prior}'; refusing to "
                f"re-parent to '{pose.parent}'. A frame with two parents makes "
                "lookups ambiguous. Re-parent deliberately with detach() instead."
            )

        edge = self._edges.get(key)
        if edge is None:
            edge = _Edge(pose.parent, pose.child, static)
            self._edges[key] = edge
            self._parent_of[pose.child] = pose.parent
            self._adj.setdefault(pose.parent, set()).add(pose.child)
            self._adj.setdefault(pose.child, set()).add(pose.parent)
        edge.insert(0.0 if stamp is None else stamp, pose, self.buffer_s)

    def detach(self, child: str) -> None:
        """Remove a frame's link to its parent, so it can be re-parented.

        Needed when a grasped object stops being observed by the camera and
        becomes rigidly attached to the gripper -- a real re-parent, made
        explicit rather than silently overwritten.
        """
        parent = self._parent_of.pop(child, None)
        if parent is None:
            return
        self._edges.pop((parent, child), None)
        self._adj.get(parent, set()).discard(child)
        self._adj.get(child, set()).discard(parent)

    # ------------------------------------------------------------- reading

    def lookup(self, target: str, source: str, stamp: float | None = None) -> SE3:
        """Return T_target_source: maps points in `source` into `target`."""
        if target == source:
            return SE3.identity(target, source)
        path = self._path(source, target)
        if path is None:
            raise TFLookupError(
                f"no transform path from '{source}' to '{target}'.\n"
                f"Known frames: {sorted(self._adj)}\n"
                f"Tree:\n{self.as_string()}"
            )
        # Walk source -> target, composing. At each hop we either have the
        # stored parent->child transform or need its inverse. Because SE3 is
        # frame-labelled, a mistake here raises FrameMismatch rather than
        # returning a wrong pose -- which is exactly why the labels exist.
        result = SE3.identity(source, source)
        for a, b in zip(path[:-1], path[1:]):
            edge = self._edges.get((b, a))
            if edge is not None:
                hop = edge.at(stamp)            # T_b_a, already the direction we walk
            else:
                hop = self._edges[(a, b)].at(stamp).inv()
            result = hop @ result
        return SE3(result.matrix, parent=target, child=source)

    def can_lookup(self, target: str, source: str, stamp: float | None = None) -> bool:
        try:
            self.lookup(target, source, stamp)
            return True
        except (TFLookupError, ValueError):
            return False

    def _path(self, start: str, goal: str) -> list[str] | None:
        """BFS over the undirected frame graph.

        BFS not DFS: in a tree the path is unique so either finds it, but BFS
        gives a useful error message sooner on a disconnected tree, and keeps
        the chain short if the single-parent rule is ever relaxed.
        """
        if start not in self._adj or goal not in self._adj:
            return None
        prev: dict[str, str | None] = {start: None}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node == goal:
                out = []
                cur: str | None = node
                while cur is not None:
                    out.append(cur)
                    cur = prev[cur]
                return out[::-1]
            for nxt in self._adj.get(node, ()):
                if nxt not in prev:
                    prev[nxt] = node
                    queue.append(nxt)
        return None

    # ----------------------------------------------------- introspection

    def as_string(self) -> str:
        """ASCII render of the tree -- our `tf2_tools view_frames`."""
        roots = [f for f in self._adj if f not in self._parent_of]
        children: dict[str, list[str]] = {}
        for child, parent in self._parent_of.items():
            children.setdefault(parent, []).append(child)
        lines: list[str] = []

        def walk(frame: str, prefix: str, last: bool, top: bool) -> None:
            conn = "" if top else ("└── " if last else "├── ")
            edge = self._edges.get((self._parent_of.get(frame, ""), frame))
            tag = ""
            if edge is not None:
                tag = " [static]" if edge.static else f" [{len(edge.stamps)} samples]"
            lines.append(f"{prefix}{conn}{frame}{tag}")
            kids = sorted(children.get(frame, []))
            new_prefix = prefix + ("" if top else ("    " if last else "│   "))
            for i, k in enumerate(kids):
                walk(k, new_prefix, i == len(kids) - 1, False)

        for r in sorted(roots):
            walk(r, "", True, True)
        return "\n".join(lines) if lines else "(empty tree)"

    @property
    def frames(self) -> set[str]:
        return set(self._adj)
