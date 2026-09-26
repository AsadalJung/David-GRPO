# Original synthetic training source

`train_original_5276.jsonl` contains the 5,276 distinct synthetic QA instances
selected before later oversampling, document-ratio experiments, or shuffling.
It is not the 3-document-15%/2-document-fill derivative. No rows were copied to
fill a training budget in this source. The file is released byte-for-byte from
the preserved source selection.

## Contents

Each JSONL line has exactly these fields:

- `id`: unique, source-namespaced sample ID;
- `question`: synthetic question;
- `answer`: reference answer;
- `evidence_documents`: documents, each with `id`, `title`, and `content`.

| Supporting documents per row | Rows |
| --- | ---: |
| 1 | 872 |
| 2 | 2,321 |
| 3 | 2,083 |
| Total | 5,276 |

The source selection combines 3,955 eligible rows from an initial 3,980-row
generation batch (excluding its 25 four/five-document rows) with 1,321 distinct
rows selected from a second 1,888-row generation batch. It is therefore not the
earliest generation batch alone, and is not a raw unfiltered union of both batches.
Original IDs were namespaced when the sources were combined to avoid collisions.

Fresh duplicate checks found zero duplicate sample IDs, normalized questions,
compact-normalized questions, normalized question-answer pairs, and normalized
title/content evidence bundles. There are no repeated evidence documents within
a row. Reuse of an individual Wikipedia document across different QA examples
is not treated as a duplicate example. This is a structural/textual duplicate
check, not a guarantee that no two questions are semantic paraphrases.

SHA-256: `b326ff78710ccadd9581fa2ef10046d400bcc14ff8af1b166dae57e8dcea3842`.
