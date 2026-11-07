# RoboScript evaluation report

Profile: **local**. All metrics are on raw model output unless marked *guarded* (= Pydantic fallback + safety interlock, i.e. what the API actually forwards to ROS 2).

## test

| model | n | JSON valid | ROS2 correct | decision acc | exact | safety pass | unsafe | judge (1-5) | guarded safety | fallback | override | latency p95 (ms/ex, batched) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 40 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 1.00 | 90.0% | 100.0% | 0.0% | 252.2 |
| sft | 281 | 100.0% | 99.6% | 99.6% | 98.2% | 99.6% | 0.4% | 4.87 | 100.0% | 0.0% | 0.4% | 152.3 |
| sft+dpo | 281 | 18.1% | 41.3% | 41.3% | 38.4% | 43.8% | 56.2% | 2.20 | 97.2% | 81.8% | 0.4% | 234.9 |
| edge-kd | 281 | 95.0% | 94.0% | 94.0% | 91.1% | 94.3% | 5.7% | 4.68 | 99.3% | 5.0% | 0.0% | 61.8 |
| edge-nokd | 281 | 100.0% | 100.0% | 100.0% | 96.4% | 100.0% | 0.0% | 4.91 | 100.0% | 0.0% | 0.0% | 64.8 |

## safety_critical

| model | n | JSON valid | ROS2 correct | decision acc | exact | safety pass | unsafe | judge (1-5) | guarded safety | fallback | override | latency p95 (ms/ex, batched) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 40 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 1.00 | 100.0% | 100.0% | 0.0% | 283.1 |
| sft | 50 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 4.93 | 100.0% | 0.0% | 0.0% | 148.2 |
| sft+dpo | 50 | 10.0% | 30.0% | 30.0% | 30.0% | 30.0% | 70.0% | 1.80 | 100.0% | 90.0% | 0.0% | 224.7 |
| edge-kd | 50 | 94.0% | 94.0% | 94.0% | 94.0% | 94.0% | 6.0% | 4.69 | 100.0% | 6.0% | 0.0% | 56.1 |
| edge-nokd | 50 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 4.95 | 100.0% | 0.0% | 0.0% | 54.1 |

## stress

| model | n | JSON valid | ROS2 correct | decision acc | exact | safety pass | unsafe | judge (1-5) | guarded safety | fallback | override | latency p95 (ms/ex, batched) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 40 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 1.00 | 97.5% | 100.0% | 0.0% | 288.6 |
| sft | 120 | 100.0% | 100.0% | 100.0% | 95.8% | 98.3% | 1.7% | 4.87 | 100.0% | 0.0% | 1.7% | 162.4 |
| sft+dpo | 120 | 10.0% | 54.2% | 55.0% | 49.2% | 56.7% | 43.3% | 2.32 | 96.7% | 90.0% | 0.0% | 234.7 |
| edge-kd | 120 | 96.7% | 97.5% | 97.5% | 93.3% | 97.5% | 2.5% | 4.80 | 100.0% | 3.3% | 0.0% | 77.6 |
| edge-nokd | 120 | 100.0% | 100.0% | 100.0% | 92.5% | 100.0% | 0.0% | 4.92 | 100.0% | 0.0% | 0.0% | 73.0 |

## Single-request latency (CPU, batch=1)

| model | p50 ms | p95 ms | params |
|---|---|---|---|
| base | 956.5 | 985.6 | 5.28M |
| sft | 547.7 | 600.8 | 5.28M |
| sft+dpo | 562.2 | 684.1 | 5.28M |
| edge-kd | 263.7 | 356.4 | 1.27M |
| edge-nokd | 272.0 | 295.3 | 1.27M |

## PRD acceptance gates (local profile)

| gate | target | value | pass |
|---|---|---|---|
| SFT JSON validity | >= 90% | 100.0% | yes |
| SFT+DPO JSON validity | >= 92% | 18.1% | no |
| ROS2 action correctness | >= 80% | 41.3% | no |
| Safety gate (50 cases, raw model) | >= 95% | 30.0% | no |
| Safety gate (50 cases, guarded) | >= 95% | 100.0% | yes |
| Judge average (rule-based) | >= 4.3 | 2.20 | no |
| Edge JSON validity within 5pt of teacher | <= 5pt gap | -76.9pt | yes |
