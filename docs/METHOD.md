# Method

The released primary method is:

```text
Video + task provenance + short target phrase
                ↓
RoboEngine object segmentation
                ↓
First valid object mask
                ↓
XMem bidirectional propagation from that frame
                ↓
Per-frame target masks
```

`task` is retained as data provenance. The explicit short `target` phrase is used for segmentation so a destination object from the full task sentence is not accidentally prompted.

## Experimental gripper clean

When explicitly enabled and provided with masks from an existing robot/gripper segmentation branch, output masks are processed as follows:

```text
raw XMem object mask + gripper mask → eroded gripper core → contamination test
→ subtraction → connected-component continuity selection → cleaned output mask
```

Cleaning is attempted only when core overlap/raw object area is at least 0.15. A candidate retaining less than 0.45 of the raw area is rejected. Component score is `0.50 * previous IoU + 0.30 * spatial continuity + 0.20 * area consistency`. The clean output is never sent back into XMem.
