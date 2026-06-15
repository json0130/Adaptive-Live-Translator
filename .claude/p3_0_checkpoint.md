# P3-0 Corpus Checkpoint

## en_ko: raw=1056 -> kept=1056 (train 950 / dev 106)
- drops: {}
- src length words: {'min': 4, 'p50': 8, 'mean': 8.1, 'max': 12}
- glossary-trigger density (terms/sentence): 1.01
- canonical-term presence (post-filter): 100%
- eval-slice overlap (hash-verified): 0
- per-term counts: {'benchmark': 71, 'latency': 70, 'quantization': 71, 'PyTorch': 70, 'attention mechanism': 71, 'LLM': 70, 'throughput': 70, 'RLHF': 71, 'hallucination': 70, 'NVIDIA': 70, 'fine-tuning': 71, 'tokenizer': 71, 'embedding': 70, 'HuggingFace': 70, 'inference': 70}

## ko_en: raw=1056 -> kept=1056 (train 950 / dev 106)
- drops: {}
- src length words: {'min': 4, 'p50': 7, 'mean': 6.8, 'max': 10}
- glossary-trigger density (terms/sentence): 1.01
- canonical-term presence (post-filter): 100%
- eval-slice overlap (hash-verified): 0
- per-term counts: {'tokenizer': 71, 'PyTorch': 70, 'hallucination': 70, 'LLM': 71, 'quantization': 71, 'HuggingFace': 70, 'fine-tuning': 70, 'throughput': 70, 'latency': 70, 'inference': 70, 'RLHF': 71, 'embedding': 70, 'benchmark': 70, 'NVIDIA': 71, 'attention mechanism': 71}

Spot-check sample: .claude/p3_0_spotcheck_sample.tsv (90 pairs)
BAIL CRITERION (user native review): >20% ungrammatical Korean OR >30% missing canonical term -> REJECT, pivot to (b) templated.