# RAG for QMUL Policy Search

**Design and evaluation of a retrieval-augmented generation (RAG) system for university policy documents**

*MSc Data Science and Artificial Intelligence · Queen Mary University of London · Information Retrieval group coursework (Group 88) · Mark: 86%*

## Overview

University policy documents contain important rules and procedures, but finding an answer can mean searching through several lengthy PDFs. This project investigates whether a RAG pipeline can retrieve relevant policy sections and use them to produce grounded, source-referenced answers.

We compared **BM25**, **TF-IDF / Vector Space Model (VSM)** and **Reciprocal Rank Fusion (RRF)**, then evaluated both retrieval effectiveness and LLM-generated answers. The aim was to understand the trade-offs between retrieval quality, answer grounding and the practical limitations of evaluating a small document collection.

## Pipeline

```text
20 QMUL policy PDFs
         |
PDF extraction and text cleaning
         |
Section-based chunking (237 chunks with source metadata)
         |
Tokenisation, query expansion and section-title boosting
         |
   BM25 ranking           TF-IDF ranking
         |                      |
         +------ RRF fusion ----+

Three retrieval methods: BM25 / TF-IDF / RRF
         |
         +--------------------------+
         |                          |
Stage 1: retrieval evaluation   Stage 2: answer generation
40 queries and graded QRELs     Top 5 chunks per method
P@10, Recall@10, MRR@10,       (up to 1,200 chars per chunk)
nDCG@10                        |
                                GPT-4o-mini + source references
                                |
                                RAGAS evaluation
                                (Gemini 2.5 Flash judge;
                                 OpenAI evaluation embeddings)
```

The three retrieval approaches were compared separately; RRF combines the BM25 and TF-IDF rankings.

## Methodology

### Documents and retrieval

- **Corpus:** 20 QMUL Finance and Governance policy documents, processed into **237 section-based chunks**.
- **Preprocessing:** text extraction with PyMuPDF and a pdfplumber fallback; removal of page-number, header/footer and whitespace artefacts.
- **Chunking:** one detected top-level section per chunk, with fallback rules when headings could not be identified. This was **not** a fixed-size, overlapping chunking strategy.
- **Indexing:** BM25 (`k1=1.2`, `b=0.5`) and TF-IDF with cosine similarity. Both give extra weight to section titles.
- **Query processing:** tokenisation, stopword removal and a manually defined, domain-specific synonym map.
- **Fusion:** RRF combines the two lexical rankings (`k=60`).

### Stage 1 — Retrieval evaluation

The evaluation used **40 manually constructed questions** (two per source document) and **230 recorded relevance judgements**. QRELs used a three-grade scale: `0` = not relevant, `1` = relevant, `2` = highly relevant.

Models were evaluated using **Precision@10, Recall@10, MRR@10 and nDCG@10** with `ir_measures`.

### Stage 2 — Answer generation and evaluation

For each retrieval method, the system supplied up to five retrieved chunks to **GPT-4o-mini**, truncating each chunk to 1,200 characters for answer generation. The prompt instructed the model to use only the supplied context and cite the retrieved sources.

Generated answers were assessed through **RAGAS**, using **Gemini 2.5 Flash** as the LLM judge and **OpenAI `text-embedding-3-small`** for evaluation embeddings. The reported metrics are faithfulness, answer relevancy and context utilisation. The embedding model was used for **RAGAS evaluation, not a separate dense-retrieval benchmark**.

## Original coursework results

### Retrieval performance

| Method | Precision@10 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.280 | 0.6086 | **0.7858** | **0.5886** |
| TF-IDF / VSM | **0.295** | 0.6320 | 0.6725 | 0.5460 |
| RRF | **0.295** | **0.6390** | 0.7609 | 0.5884 |

BM25 ranked relevant sections earliest on average, with the highest MRR@10 and nDCG@10. RRF achieved the highest recall, although its nDCG@10 was only marginally below BM25.

### RAGAS answer evaluation

| Method | Faithfulness | Answer relevancy | Context utilisation |
|---|---:|---:|---:|
| BM25 | 0.9652 | 0.9084 | **0.6650** |
| TF-IDF / VSM | 0.9431 | 0.9120 | 0.5910 |
| RRF | **0.9735** | **0.9261** | 0.6293 |

RRF recorded the highest faithfulness and answer relevancy, while BM25 recorded the highest context utilisation. These outcomes illustrate why retrieval metrics and answer-quality metrics should be interpreted together rather than treated as interchangeable.

**Results note:** These are the **original submitted coursework results**, preserved from the executed notebook. They have **not been independently revalidated or rerun** for this portfolio. The RAGAS stage submitted 38 answers for BM25, 38 for TF-IDF and 39 for RRF for evaluation; some metric scores were missing because of API timeouts. Reported averages exclude missing values, so comparisons need care.

## Limitations and lessons

The following limitations reflect both the original coursework evaluation and a subsequent technical review during portfolio preparation. The submitted methodology and reported results have not been replaced or rerun.

- **Small, manually judged test set:** 40 queries cannot establish performance across all policy questions. No statistical significance testing or hyperparameter sensitivity analysis was performed.
- **QREL consistency:** A subsequent review found eight judgements for questions `q07` and `q08` that did not match the indexed Free Speech policy chunks. A difference in document versions is plausible but unconfirmed. The original results have not been altered.
- **Uneven chunk lengths:** section-based chunks could be long; only the first 1,200 characters of each selected chunk reached the generation model.
- **Generation/evaluation mismatch:** RAGAS assessed full retrieved chunks rather than the truncated passages used for answer generation.
- **Incomplete evaluations:** some responses were skipped, and API timeouts resulted in missing RAGAS scores.
- **Source verification:** generated citations identify retrieved sections and pages, but individual answer claims were not independently verified against those citations.
- **Limited error analysis:** the submitted evaluation did not provide detailed per-query failure examples illustrating where individual retrieval or generation methods struggled.

These are limitations of the historical coursework, not claims that the system is ready for live institutional use.

## Source documents and reproducibility

The original **20 policy PDFs are not redistributed** in this repository. Policies can be located through QMUL's official resources:

- [QMUL policies by category](https://www.qmul.ac.uk/governance-and-legal-services/policy/policies-by-category/)
- [QMUL non-academic policy archive](https://www.qmul.ac.uk/governance-and-legal-services/policy/non-academic-policy-archive/)

**Important:** Those pages may contain newer or different versions from the ones used in the coursework. The original notebook contains source document identifiers, but the historical dataset is not automatically downloadable here. Exact reproduction requires the same document versions.

The repository preserves the submitted, Google Colab-oriented notebook, **[`IR_CW2_Main_G-88.ipynb`](IR_CW2_Main_G-88.ipynb)**, including its recorded outputs. To attempt a rerun:

1. Obtain the source PDFs and place them in the notebook's expected Google Drive folder, `/content/drive/MyDrive/IR_Qmary/PDFs/`, or adapt its hard-coded paths locally.
2. Run the notebook in Google Colab with compatible Python dependencies. The preprocessing stage produces `clean_chunks.csv`; retrieval evaluation uses the notebook's queries and QRELs.
3. The **Stage 1 retrieval evaluation** uses local computation once the data and dependencies are available; it does not require paid model API calls.
4. **Stage 2** requires private OpenAI and Google API credentials, network access and access to the named models. The original notebook uses API-key placeholders that must be configured privately for a local run. **Never commit populated keys.**

Running generation and RAGAS evaluation can incur API charges, and model or dependency changes may produce different outputs. **The paid-API stages were deliberately not rerun for GitHub publication**, because the purpose of this repository is to document the assessed submission rather than create new benchmarks at additional cost.

This is an **archived coursework implementation**, not a one-click reproducible package or a deployed service.

## Technologies

Python · Jupyter / Google Colab · pandas · NumPy · PyMuPDF · pdfplumber · `rank_bm25` · scikit-learn · `ir_measures` · OpenAI API (`gpt-4o-mini`, `text-embedding-3-small`) · RAGAS · Google Gemini 2.5 Flash · LangChain integrations

## Group work and my contributions

This was a **collaborative Group 88 submission** for QMUL's MSc Data Science and Artificial Intelligence programme, awarded **86%**.

My main responsibilities were the **RAGAS answer-quality evaluation** and preparing the **query relevance judgements (QRELs)**. I also made a substantial contribution to the Python notebook's implementation, integration and execution, using AI-assisted coding tools during development, and contributed to interpreting the results.

Groupmates primarily worked on document chunking, retrieval and reranking, and LLM answer generation. Work was shared and reviewed across the group; the repository represents the **original group submission**, not a claim of sole authorship.
