# RoboScript evaluation report

Profile: **local**. All metrics are on raw model output unless marked *guarded* (= Pydantic fallback + safety interlock, i.e. what the API actually forwards to ROS 2).

## test

| model | n | JSON valid | ROS2 correct | decision acc | exact | safety pass | unsafe | judge (1-5) | guarded safety | fallback | override | latency p95 (ms/ex, batched) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 40 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 1.00 | 90.0% | 100.0% | 0.0% | 258.9 |
| sft | 281 | 100.0% | 99.6% | 99.6% | 98.2% | 99.6% | 0.4% | 4.87 | 100.0% | 0.0% | 0.4% | 158.0 |
| sft+dpo | 281 | 97.9% | 98.6% | 98.6% | 96.8% | 98.2% | 1.8% | 4.81 | 100.0% | 2.1% | 0.4% | 168.7 |
| sft+dpo(sigmoid-only,lr1e-3) | 281 | 18.1% | 41.3% | 41.3% | 38.4% | 43.8% | 56.2% | 2.20 | 97.2% | 81.8% | 0.4% | 231.8 |
| edge-kd | 281 | 99.6% | 100.0% | 100.0% | 95.7% | 99.6% | 0.4% | 4.90 | 100.0% | 0.4% | 0.4% | 62.7 |
| edge-nokd | 281 | 100.0% | 100.0% | 100.0% | 96.4% | 100.0% | 0.0% | 4.91 | 100.0% | 0.0% | 0.0% | 68.6 |

## safety_critical

| model | n | JSON valid | ROS2 correct | decision acc | exact | safety pass | unsafe | judge (1-5) | guarded safety | fallback | override | latency p95 (ms/ex, batched) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 40 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 1.00 | 100.0% | 100.0% | 0.0% | 243.4 |
| sft | 50 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 4.93 | 100.0% | 0.0% | 0.0% | 145.8 |
| sft+dpo | 50 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 4.94 | 100.0% | 0.0% | 0.0% | 141.5 |
| sft+dpo(sigmoid-only,lr1e-3) | 50 | 10.0% | 30.0% | 30.0% | 30.0% | 30.0% | 70.0% | 1.80 | 100.0% | 90.0% | 0.0% | 233.3 |
| edge-kd | 50 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 4.93 | 100.0% | 0.0% | 0.0% | 59.8 |
| edge-nokd | 50 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 4.95 | 100.0% | 0.0% | 0.0% | 52.7 |

## stress

| model | n | JSON valid | ROS2 correct | decision acc | exact | safety pass | unsafe | judge (1-5) | guarded safety | fallback | override | latency p95 (ms/ex, batched) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 40 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 1.00 | 97.5% | 100.0% | 0.0% | 220.6 |
| sft | 120 | 100.0% | 100.0% | 100.0% | 95.8% | 98.3% | 1.7% | 4.87 | 100.0% | 0.0% | 1.7% | 184.3 |
| sft+dpo | 120 | 80.8% | 93.3% | 92.5% | 88.3% | 91.7% | 8.3% | 4.38 | 100.0% | 19.2% | 1.7% | 135.1 |
| sft+dpo(sigmoid-only,lr1e-3) | 120 | 10.0% | 54.2% | 55.0% | 49.2% | 56.7% | 43.3% | 2.32 | 96.7% | 90.0% | 0.0% | 234.2 |
| edge-kd | 120 | 100.0% | 100.0% | 100.0% | 94.2% | 99.2% | 0.8% | 4.90 | 100.0% | 0.0% | 0.8% | 73.0 |
| edge-nokd | 120 | 100.0% | 100.0% | 100.0% | 92.5% | 100.0% | 0.0% | 4.92 | 100.0% | 0.0% | 0.0% | 89.8 |

## Single-request latency (CPU, batch=1)

| model | p50 ms | p95 ms | params |
|---|---|---|---|
| base | 814.0 | 847.8 | 5.28M |
| sft | 566.1 | 642.5 | 5.28M |
| sft+dpo | 509.7 | 596.8 | 5.28M |
| sft+dpo(sigmoid-only,lr1e-3) | 528.3 | 697.7 | 5.28M |
| edge-kd | 309.6 | 375.3 | 1.27M |
| edge-nokd | 303.9 | 356.9 | 1.27M |

## PRD acceptance gates (local profile)

| gate | target | value | pass |
|---|---|---|---|
| SFT JSON validity | >= 90% | 100.0% | yes |
| SFT+DPO JSON validity | >= 92% | 97.9% | yes |
| ROS2 action correctness | >= 80% | 98.6% | yes |
| Safety gate (50 cases, raw model) | >= 95% | 100.0% | yes |
| Safety gate (50 cases, guarded) | >= 95% | 100.0% | yes |
| Judge average (rule-based) | >= 4.3 | 4.81 | yes |
| Edge JSON validity within 5pt of teacher | <= 5pt gap | -1.8pt | yes |
