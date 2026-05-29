# Ground Truth Q&A - Writing Guide

This is the most important deliverable in the project. Block 3-4 hours.
Do not auto-generate. Do not rush.

## Why this matters

Your evaluation is only as good as your ground truth. If your questions are vague or
your answers are wrong, the scores you report are meaningless, and the project
cannot demonstrate real competence.

## Rules for writing good questions

1. **Each question must have a single, verifiable answer** from the actual filings.
   "What was Apple's total revenue in fiscal year 2023?" - good.
   "How is Apple doing?" - bad.

2. **Record the exact source**: company name, section, and page/paragraph.
   You will need this when scoring retrieval correctness.

3. **Cover all 5 companies** - at least 5 questions per company minimum,
   but vary by section too.

4. **Cover different question types**:
   - Exact number lookup ("What was Nvidia's gross margin in FY2024?")
   - Comparison across companies ("Which company had the highest R&D spend?")
   - Risk factor summary ("What cybersecurity risks does Microsoft disclose?")
   - Multi-hop ("What does Apple cite as the primary risk to its supply chain?")

5. **Include some hard questions** - ones where you expect the baseline to fail.
   These are where query rewriting will show lift.

## CSV format

Save your answers to `eval/ground_truth.csv` with this schema:

```
id,question,answer,company,section,difficulty
```

- `id`: q001, q002, …, q030
- `question`: The full question text
- `answer`: The correct answer (exact or paraphrased from the filing)
- `company`: AAPL / MSFT / NVDA / META / GOOGL
- `section`: e.g., "Item 1A Risk Factors", "Item 7 MD&A", "Item 8 Financial Statements"
- `difficulty`: easy / medium / hard

## Example rows

```csv
id,question,answer,company,section,difficulty
q001,What was Apple's total net revenue for fiscal year 2024?,Apple reported total net revenue of $391.0 billion for fiscal year 2024.,AAPL,Item 8 Financial Statements,easy
q002,What does Apple identify as its primary supply chain risk?,Apple cites concentration of manufacturing in China and dependence on single-source suppliers as primary supply chain risks.,AAPL,Item 1A Risk Factors,medium
q003,"Which of the five companies (Apple, Microsoft, Nvidia, Meta, Google) spent the most on research and development in their most recent fiscal year?","Nvidia spent approximately 27% of revenue on R&D, the highest ratio among the five companies.",NVDA,Item 7 MD&A,hard
```

## Distribution target (30 questions)

| Company | Count | Sections to cover |
|---------|-------|-------------------|
| AAPL    | 6     | Revenue, risk factors, segments, supply chain |
| MSFT    | 6     | Cloud revenue, AI investments, risk factors |
| NVDA    | 6     | Data center revenue, gross margin, demand risk |
| META    | 6     | Ad revenue, Reality Labs losses, regulation risk |
| GOOGL   | 6     | Search revenue, YouTube, cloud, AI investments |

Aim for: 10 easy, 12 medium, 8 hard.

## After writing ground truth

1. Rename this file to `ground_truth_template.md` (leave it as reference)
2. Save your actual answers to `eval/ground_truth.csv`
3. Commit it: `git add eval/ground_truth.csv && git commit -m "add 30 ground truth Q&A pairs"`

Do not continue to Days 3-5 until this CSV exists and has 30 rows.
