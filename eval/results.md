# RAG Eval Results

**Run date:** 2026-05-28  |  **Wall time:** 10.9 min
**Questions:** 65 (59 answerable, 6 unanswerable)
**Headline combo:** sec_section_aware x hybrid_rerank_filter

---

## Table 1 - Retrieval Metrics (all 8 configs)

Recall computed on 59 answerable questions only.

| Collection    | Config                | Recall@5 | Recall@10 | Latency p50 (ms) | Latency p95 (ms) |
|---------------|-----------------------|----------|-----------|------------------|------------------|
| recursive     | dense                 |    84.7% |     89.8% |              320 |              339 |
| recursive     | hybrid                |    83.1% |     93.2% |              398 |              456 |
| recursive     | hybrid_rerank         |    79.7% |     94.9% |              402 |              447 |
| recursive     | hybrid_rerank_filter  |    94.9% |    100.0% |              159 |              243 |
| section_aware | dense                 |    81.4% |     89.8% |              318 |              358 |
| section_aware | hybrid                |    83.1% |     93.2% |              400 |              449 |
| section_aware | hybrid_rerank         |    79.7% |     94.9% |              400 |              452 |
| section_aware | hybrid_rerank_filter  |    94.9% |    100.0% |              152 |              230 |

---

## Table 1 Observations

**Metadata filtering is the dominant lever.** `hybrid_rerank_filter` lifts recall@5 by 10-15
points over every non-filtered config and pushes recall@10 to 100.0% on both collections.
The filter narrows the candidate pool to the correct company/year before reranking, removing
most of the noise that would otherwise displace the relevant chunk.

**Reranking alone hurts recall@5 by 5 points** (hybrid_rerank 79.7% vs hybrid 83.1%), while
recall@10 improves (94.9% vs 93.2%). The reranker is moving relevant chunks *out of* positions
1-5 and into positions 6-10. bge-reranker-base's general-domain training is likely misaligned
with dense financial tables: it overweights superficially similar chunks and demotes
the exact-match row. Filter + rerank recovers this; rerank alone does not.

**Chunking strategy: null result.** Expected: section-aware chunking would outperform
recursive on boundary_sensitive questions by keeping relevant sentences within a single chunk
instead of splitting at an arbitrary token boundary. Observed: section_aware scored 3.3 points
*worse* on dense retrieval and tied on all other configs, including a 100%/100% tie on
boundary_sensitive specifically. Recursive chunking with 200-token overlap already preserves
sentences across most section edges at chunk_size=1000, and the metadata filter removes the
remaining noise. The section boundary hypothesis was not supported at this chunk size.

---

## Table 2 - Generation Metrics (headline combo only)

Collection: `sec_section_aware` · Config: `hybrid_rerank_filter`

| Metric                    | Value         |
|---------------------------|---------------|
| Numerical exact match     | 22/23 (95.7%) |
| Refusal accuracy          | 5/6    |
| False refusal rate        | 2/59  (3.4%) |
| Append violations         | 3/65 (4.6%) |
| Generation latency p50    | 7823 ms        |
| Generation latency p95    | 14494 ms        |

> **Append violations** = responses where the model provided a cited answer AND appended the refusal phrase (prompt Rule 6/7 ambiguity). v1 prompt (Task 6 system prompt): 24/65 (36.9%). v2 prompt (mutually exclusive rules): 3/65.

---

## Generation Failure: Q20 (Goldman Sachs net earnings FY2023)

**Expected:** $8,520M (= $8.52B)  **Model answer:** "$952 million [N1]"

Retrieval succeeded (recall@5 hit - correct chunk fetched). The chunk contained the GS income
statement, which has 30+ dollar figures across multiple line items. Qwen 2.5 7B read the wrong
row - $952M plausibly appears as a single-segment or per-share-related figure in the same
table. This is the characteristic failure mode of dense financial tables: lexical density of
numbers near the target makes table-row disambiguation hard for a 7B model. The citation
format `[N1]` is also non-standard (prompt specifies `[1]`, `[2]`), indicating the model's
response template diverged from the instruction, a secondary sign of confusion.

---

## Table 3 - Recall@5 by Question Type (headline combo)

| Question type            | Recall@5 | n  |
|--------------------------|----------|----|
| single_doc_lookup        |   100.0% | 10 |
| numerical_extraction     |   100.0% | 12 |
| cross_company_comparison |    90.0% | 10 |
| time_series              |   100.0% | 8  |
| narrative_risk           |    71.4% | 7  |
| boundary_sensitive       |   100.0% | 12 |

---

## Boundary-sensitive: recursive vs section_aware

| Collection    | Recall@5 (boundary_sensitive) |
|---------------|-------------------------------|
| recursive     | 100.0% (12/12)          |
| section_aware | 100.0% (12/12)          |
