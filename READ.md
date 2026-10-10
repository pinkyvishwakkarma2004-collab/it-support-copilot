# IT Support AI Copilot

Classifies support tickets, drafts replies from similar resolved tickets (RAG),
checks ticket status via function calling, and escalates to a human when unsure.

## Pipeline
- Classifier: TF-IDF + LinearSVC (C chosen by 3-fold CV, test set used once)
- RAG: ChromaDB with 2000 resolved tickets (train split only)
- LLM: Groq (gpt-oss) with function calling, max 3 tool steps
- Guardrails: PII masking, prompt-injection check
- Escalation: retrieval distance threshold

## Results

### Classifier (test set, 3,268 tickets, 10 queues)
| Model | Macro F1 |
|---|---|
| TF-IDF + Logistic Regression | 0.54 |
| Sentence embeddings + LR | 0.29 |
| TF-IDF + LinearSVC (CV-tuned) | 0.725 |

### LLM zero-shot vs SVC (same 40 tickets, accuracy)
LLM zero-shot: 0.275-0.325 (two runs) | TF-IDF+SVC: 0.75

### Answer quality (LLM-as-judge, 12 tickets, score 1-5)
| Model | Judge score | Latency | Avg tokens |
|---|---|---|---|
| gpt-oss-120b | 4.33 | 2.24 s | 958 |
| gpt-oss-20b | 3.58 | 1.21 s | 1117 |

### Retrieval (200 tickets, hit@3 on queue)
Vector 0.59 | BM25 0.615 | Hybrid 0.61 (differences within noise)

### Other
- Demo eval (15 tickets): 73% classifier accuracy, 7% escalation, 2.75 s avg latency
- Exact-match cache: repeat ticket skips the LLM call (1.5 s to 0 s)

## Limitations
- Dataset is synthetic, so retrieval may look better than in real life
- Small eval samples (12-40 tickets); differences are indicative only
- Judge model is the same as the large answering model (self-preference bias)
- Cache only matches identical tickets; a semantic cache would be next
- Free-tier API rate limits affected latency

## Next steps
- Tune escalation threshold on a validation set
- RAGAS for answer faithfulness
- Semantic cache (embedding similarity)
- Wrap in FastAPI + Docker, deploy
