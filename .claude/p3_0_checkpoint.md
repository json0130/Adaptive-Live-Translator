# P3-0 Corpus Checkpoint

## en_ko: raw=1084 -> kept=789 (train 710 / dev 79)
- drops: {'ko_contaminated': 14, 'no_en_term': 31, 'no_canonical_ko': 226, 'dup': 24}
- src length words: {'min': 7, 'p50': 13, 'mean': 13.0, 'max': 26}
- glossary-trigger density (terms/sentence): 1.02
- canonical-term presence (post-filter): 100%
- eval-slice overlap (hash-verified): 0
- per-term counts: {'quantization': 69, 'embedding': 65, 'latency': 77, 'PyTorch': 27, 'benchmark': 45, 'fine-tuning': 46, 'LLM': 72, 'inference': 66, 'throughput': 70, 'tokenizer': 27, 'NVIDIA': 70, 'RLHF': 74, 'attention mechanism': 58, 'hallucination': 23}

## ko_en: raw=1104 -> kept=958 (train 862 / dev 96)
- drops: {'no_en_term': 112, 'dup': 1, 'ko_contaminated': 2, 'no_canonical_ko': 31}
- src length words: {'min': 5, 'p50': 10, 'mean': 10.0, 'max': 21}
- glossary-trigger density (terms/sentence): 1.01
- canonical-term presence (post-filter): 100%
- eval-slice overlap (hash-verified): 0
- per-term counts: {'latency': 73, 'fine-tuning': 55, 'quantization': 64, 'RLHF': 75, 'throughput': 74, 'PyTorch': 74, 'HuggingFace': 75, 'embedding': 63, 'hallucination': 74, 'attention mechanism': 74, 'benchmark': 75, 'inference': 32, 'NVIDIA': 63, 'tokenizer': 57, 'LLM': 30}

Spot-check sample: .claude/p3_0_spotcheck_sample.tsv (87 pairs)
BAIL CRITERION (user native review): >20% ungrammatical Korean OR >30% missing canonical term -> REJECT, pivot to (b) templated.