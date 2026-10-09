| motion | noise (sigma) | filter | turns (mean) | turns (median) | all busted | true-cell mass | s/game |
|---|---|---|---:|---:|---:|---:|---:|
| random | low (0.5) | exact | 61.0 | 58 | 100% | 0.597 | 0.090 |
| random | low (0.5) | particles-10 | 133.9 | 111 | 83% | 0.306 | 0.052 |
| random | low (0.5) | particles-50 | 60.3 | 56 | 100% | 0.640 | 0.059 |
| random | low (0.5) | particles-200 | 81.0 | 58 | 92% | 0.494 | 0.212 |
| random | med (1.0) | exact | 95.1 | 77 | 100% | 0.279 | 0.092 |
| random | med (1.0) | particles-10 | 160.3 | 152 | 92% | 0.175 | 0.076 |
| random | med (1.0) | particles-50 | 123.2 | 82 | 83% | 0.245 | 0.102 |
| random | med (1.0) | particles-200 | 102.5 | 95 | 92% | 0.287 | 0.260 |
| random | high (2.0) | exact | 295.4 | 300 | 8% | 0.176 | 0.357 |
| random | high (2.0) | particles-10 | 183.1 | 187 | 58% | 0.135 | 0.062 |
| random | high (2.0) | particles-50 | 288.4 | 300 | 17% | 0.169 | 0.257 |
| random | high (2.0) | particles-200 | 300.0 | 300 | 0% | 0.179 | 0.809 |
| lurker | low (0.5) | exact | 33.5 | 39 | 100% | 0.673 | 0.180 |
| lurker | low (0.5) | particles-10 | 34.2 | 39 | 100% | 0.464 | 0.183 |
| lurker | low (0.5) | particles-50 | 33.6 | 39 | 100% | 0.641 | 0.223 |
| lurker | low (0.5) | particles-200 | 33.9 | 39 | 100% | 0.666 | 0.336 |
| lurker | med (1.0) | exact | 34.8 | 40 | 100% | 0.376 | 0.295 |
| lurker | med (1.0) | particles-10 | 36.3 | 42 | 100% | 0.291 | 0.299 |
| lurker | med (1.0) | particles-50 | 34.8 | 41 | 100% | 0.369 | 0.211 |
| lurker | med (1.0) | particles-200 | 35.0 | 41 | 100% | 0.378 | 0.258 |
| lurker | high (2.0) | exact | 35.8 | 41 | 100% | 0.246 | 0.280 |
| lurker | high (2.0) | particles-10 | 36.8 | 41 | 100% | 0.206 | 0.264 |
| lurker | high (2.0) | particles-50 | 36.1 | 41 | 100% | 0.243 | 0.264 |
| lurker | high (2.0) | particles-200 | 36.8 | 40 | 100% | 0.253 | 0.255 |
| patrol | low (0.5) | exact | 82.3 | 68 | 100% | 0.881 | 0.114 |
| patrol | low (0.5) | particles-10 | 266.0 | 300 | 17% | 0.104 | 0.081 |
| patrol | low (0.5) | particles-50 | 228.1 | 300 | 33% | 0.229 | 0.133 |
| patrol | low (0.5) | particles-200 | 137.7 | 81 | 75% | 0.502 | 0.233 |
| patrol | med (1.0) | exact | 63.1 | 64 | 100% | 0.585 | 0.062 |
| patrol | med (1.0) | particles-10 | 267.1 | 300 | 17% | 0.121 | 0.080 |
| patrol | med (1.0) | particles-50 | 226.4 | 300 | 33% | 0.244 | 0.144 |
| patrol | med (1.0) | particles-200 | 79.7 | 63 | 92% | 0.357 | 0.123 |
| patrol | high (2.0) | exact | 151.9 | 141 | 83% | 0.212 | 0.147 |
| patrol | high (2.0) | particles-10 | 274.8 | 300 | 17% | 0.094 | 0.100 |
| patrol | high (2.0) | particles-50 | 300.0 | 300 | 0% | 0.092 | 0.152 |
| patrol | high (2.0) | particles-200 | 215.9 | 268 | 58% | 0.157 | 0.305 |

| motion | noise (sigma) | Viterbi path accuracy | filtered MAP accuracy | scored turns |
|---|---|---:|---:|---:|
| random | low (0.5) | 0.816 | 0.682 | 1187 |
| random | med (1.0) | 0.428 | 0.378 | 1780 |
| random | high (2.0) | 0.285 | 0.298 | 446 |
| lurker | low (0.5) | 0.772 | 0.754 | 623 |
| lurker | med (1.0) | 0.491 | 0.511 | 644 |
| lurker | high (2.0) | 0.328 | 0.351 | 667 |
| patrol | low (0.5) | 0.980 | 0.892 | 1702 |
| patrol | med (1.0) | 0.893 | 0.680 | 1200 |
| patrol | high (2.0) | 0.644 | 0.333 | 2101 |

| noise (sigma) | particles | KL(exact to particles) | true-cell mass (particles) | true-cell mass (exact) |
|---|---:|---:|---:|---:|
| low (0.5) | 10 | 5.5932 | 0.474 | 0.626 |
| low (0.5) | 30 | 5.7661 | 0.375 | 0.626 |
| low (0.5) | 100 | 3.1723 | 0.512 | 0.626 |
| low (0.5) | 300 | 0.5389 | 0.595 | 0.626 |
| low (0.5) | 1000 | 0.0169 | 0.632 | 0.626 |
| med (1.0) | 10 | 4.9056 | 0.251 | 0.281 |
| med (1.0) | 30 | 4.0180 | 0.227 | 0.281 |
| med (1.0) | 100 | 0.4236 | 0.282 | 0.281 |
| med (1.0) | 300 | 0.0588 | 0.274 | 0.281 |
| med (1.0) | 1000 | 0.0160 | 0.285 | 0.281 |
| high (2.0) | 10 | 5.3100 | 0.149 | 0.178 |
| high (2.0) | 30 | 2.2969 | 0.169 | 0.178 |
| high (2.0) | 100 | 0.2883 | 0.174 | 0.178 |
| high (2.0) | 300 | 0.1035 | 0.173 | 0.178 |
| high (2.0) | 1000 | 0.0201 | 0.180 | 0.178 |
