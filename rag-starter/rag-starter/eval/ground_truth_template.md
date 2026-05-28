# Ground Truth Evaluation Set

30 questions covering the 5 10-K filings. You write these manually after skimming the documents. This is the single most important step of the project, do not skip it.

## Why manual ground truth

Auto-generated ground truth (asking an LLM to make questions) creates circular eval, the same kind of LLM that fails to answer also fails to ask good questions. Manual ground truth catches real failure modes.

## Distribution targets

- 10 single-document factual questions (easy retrieval)
- 10 numerical/financial questions (tests table handling)
- 5 multi-document comparison questions (hard, will likely fail, that is fine)
- 5 reasoning questions requiring inference (hard, also fine to fail)

## Format

Each row: question_id, question, expected_answer, expected_source_doc, expected_section, difficulty

## Template

| id | question | expected_answer | source_doc | section | difficulty |
|----|----------|-----------------|------------|---------|------------|
| 1 | What is Apple's primary risk factor related to China? | Apple has significant exposure to China for both supply chain and revenue | AAPL_2024 | Risk Factors | easy |
| 2 | How much did Microsoft spend on R&D in fiscal 2024? | $29.5 billion | MSFT_2024 | MD&A | numerical |
| 3 | ... | ... | ... | ... | ... |

## Process for writing your 30 questions

1. Open each 10-K, skim Risk Factors and MD&A sections (15 min per doc)
2. Note 6 to 7 facts per doc that have clear, unambiguous answers
3. Phrase as natural-language questions
4. Verify the answer is actually in the document
5. Save to `eval/ground_truth.csv` once finalized

This will take 3 to 4 hours. Block the time. Do not rush it. The quality of your eval is what makes the project credible.
