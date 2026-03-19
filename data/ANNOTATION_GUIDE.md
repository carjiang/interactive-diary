# BIO Annotation Guide for Diary NER Training Data

## Deliverable

A JSONL file where each line is one diary sentence with token-level BIO labels:

```json
{"tokens": ["I", "told", "Sarah", "about", "the", "meeting"], "bio_tags": ["B-PARTICIPANT", "B-ACTION", "B-PARTICIPANT", "O", "B-CONTENT", "I-CONTENT"]}
```

Validate your output before delivering:

```bash
python -c "from data.schema_spec import load_dataset; load_dataset('your_file.jsonl', strict=False)"
```

This will report every invalid line at once.

---

## How BIO Tagging Works

Every token gets exactly one label. The label is either `O` (not an entity) or a prefix + entity type. **MN: does this necessarily mean that each word is a token? My understanding of LLMs is that they tokenize in chunks of words.**

**Prefixes:**


| Prefix | Meaning                             | When to use                                    |
| ------ | ----------------------------------- | ---------------------------------------------- |
| `B-`   | **Beginning** of an entity span     | First word of any entity                       |
| `I-`   | **Inside** a continuing entity span | Second, third, ... word of a multi-word entity |
| `O`    | **Outside** any entity              | Words that aren't part of any entity           |


**Rules:**

- Every entity starts with `B-`. Always. Even single-word entities.
- `I-X` can only follow `B-X` or `I-X` of the **same** entity type.
- `I-ACTION` after `B-PARTICIPANT` is **illegal** — that's two separate entities.
- When in doubt, tag `O`. False negatives are better than wrong labels.

---

## The 6 Entity Types

### 1. PARTICIPANT

**What it tags:** People mentioned in the diary entry.

**Tag as PARTICIPANT:**

- Proper names: "Sarah", "Jake", "Dr. Miller"
- Role references: "my boss", "the professor", "our team lead"
- First-person pronouns: "I", "me", "my" — tag these as `B-PARTICIPANT`. The downstream inference layer maps them to the canonical author token `"ME"`.

**Examples:**


| Tokens               | Tags                                |
| -------------------- | ----------------------------------- |
| `Sarah`              | `B-PARTICIPANT`                     |
| `Dr.` `Miller`       | `B-PARTICIPANT` `I-PARTICIPANT`     |
| `my` `boss`          | `B-PARTICIPANT` `I-PARTICIPANT`     |
| `I`                  | `B-PARTICIPANT`                     |
| `Jake` `and` `Sarah` | `B-PARTICIPANT` `O` `B-PARTICIPANT` |


---

### 2. ACTION

**What it tags:** Verbs or verb phrases describing what someone did.

**Tag as ACTION:**

- Physical actions: "told", "sent", "cancelled", "moved"
- Communication: "asked", "mentioned", "texted", "emailed"
- Planning: "planned", "scheduled", "decided", "agreed"
- Do NOT tag generic verbs like "was", "had", "is", "went" unless they carry specific meaning **what does specific meaning mean?**

**Examples:**


| Tokens              | Tags                                                                |
| ------------------- | ------------------------------------------------------------------- |
| `told`              | `B-ACTION`                                                          |
| `sent` `an` `email` | `B-ACTION` `O` `O` — only the verb is ACTION, "an email" is CONTENT |
| `called` `off`      | `B-ACTION` `I-ACTION` — phrasal verb                                |
| `set` `up`          | `B-ACTION` `I-ACTION` — phrasal verb                                |


---

### 3. CONTENT

**What it tags:** The informational payload — what was said, decided, or is being referred to.

**Tag as CONTENT:** by I-CONTENT


- Objects of actions: "the meeting", "the report", "the deadline"
- Specific information: "the meeting was at 3pm", "the project timeline"
- Key-value style info: "room 204", "next Tuesday"
- This is often multi-word — use B-CONTENT followed
**Examples:**


| Tokens                    | Tags                                |
| ------------------------- | ----------------------------------- |
| `the` `meeting`           | `B-CONTENT` `I-CONTENT`             |
| `the` `report` `deadline` | `B-CONTENT` `I-CONTENT` `I-CONTENT` |
| `room` `204`              | `B-CONTENT` `I-CONTENT`             |


**Important:** CONTENT answers "what?" — what was told, what was planned, what changed.

---

### 4. BELIEF_CUE

**What it tags:** Words that signal a mental state, assumption, uncertainty, or commitment.

**Tag as BELIEF_CUE:**

- Cognitive verbs: "thought", "believed", "assumed", "expected", "figured"
- Uncertainty markers: "maybe", "probably", "might", "wasn't sure"
- Commitment language: "promised", "committed", "guaranteed"
- Surprise/discovery: "realized", "found out", "discovered", "turned out"

**Why this matters:** These are the signals the downstream Theory-of-Mind system uses to track what people believe vs. what's actually true.

**Examples:**


| Tokens          | Tags                          |
| --------------- | ----------------------------- |
| `thought`       | `B-BELIEF_CUE`                |
| `wasn't` `sure` | `B-BELIEF_CUE` `I-BELIEF_CUE` |
| `I` `assumed`   | `B-PARTICIPANT` `B-BELIEF_CUE` |
| `probably`      | `B-BELIEF_CUE`                |
| `turned` `out`  | `B-BELIEF_CUE` `I-BELIEF_CUE` |


---

### 5. TEMPORAL

**What it tags:** Time expressions and sequencing cues that establish when things happened.

**Tag as TEMPORAL:**

- Absolute time: "3pm", "Tuesday", "next week", "January 5th"
- Relative time: "yesterday", "earlier", "later that day", "last night"
- Sequencing words: "then", "after", "before", "finally", "first"
- Duration: "for an hour", "all morning"

**Why this matters:** These give the model temporal ordering — which events happened before or after others.

**Examples:**


| Tokens                   | Tags                                                          |
| ------------------------ | ------------------------------------------------------------- |
| `after` `lunch`          | `B-TEMPORAL` `I-TEMPORAL`                                     |
| `then`                   | `B-TEMPORAL`                                                  |
| `yesterday` `morning`    | `B-TEMPORAL` `I-TEMPORAL`                                     |
| `at` `3pm`               | `B-TEMPORAL` `I-TEMPORAL`                                     |
| `before` `the` `meeting` | `B-TEMPORAL` `O` `O` — "the meeting" is not a time expression |


**Edge case:** "before the meeting" — "before" is TEMPORAL (it's a sequencing cue), but "the meeting" is CONTENT. Don't extend TEMPORAL into the object.

---

### 6. PERCEPTION

**What it tags:** Verbs indicating that someone observed or sensed something.

**Tag as PERCEPTION:**

- Visual: "saw", "noticed", "watched", "looked at"
- Auditory: "heard", "overheard"
- General awareness: "found", "spotted", "caught", "recognized"

**Why this matters:** Perception determines what an agent knows. If Sarah "saw" the schedule change, her beliefs update. If she didn't, they don't. This is critical for Theory-of-Mind reasoning.

**Examples:**


| Tokens        | Tags                          |
| ------------- | ----------------------------- |
| `saw`         | `B-PERCEPTION`                |
| `noticed`     | `B-PERCEPTION`                |
| `overheard`   | `B-PERCEPTION`                |
| `looked` `at` | `B-PERCEPTION` `I-PERCEPTION` |


---

## Full Annotation Walkthrough

**Sentence:** "After lunch I thought Sarah noticed the schedule had changed to 4pm"

Step by step:


| Token      | Reasoning                                            | Tag             |
| ---------- | ---------------------------------------------------- | --------------- |
| `After`    | Time cue — starts a temporal phrase                  | `B-TEMPORAL`    |
| `lunch`    | Continues the time phrase "after lunch"              | `I-TEMPORAL`    |
| `I`        | Diary author pronoun — tag as participant             | `B-PARTICIPANT` |
| `thought`  | Mental state verb — signals a belief                 | `B-BELIEF_CUE`  |
| `Sarah`    | A person's name                                      | `B-PARTICIPANT` |
| `noticed`  | Perception verb — Sarah observed something           | `B-PERCEPTION`  |
| `the`      | Start of the informational payload                   | `B-CONTENT`     |
| `schedule` | Continues content                                    | `I-CONTENT`     |
| `had`      | Continues content                                    | `I-CONTENT`     |
| `changed`  | Continues content                                    | `I-CONTENT`     |
| `to`       | Continues content                                    | `I-CONTENT`     |
| `4pm`      | Continues content (the time is part of what changed) | `I-CONTENT`     |


**Result:**

```json
{"tokens": ["After", "lunch", "I", "thought", "Sarah", "noticed", "the", "schedule", "had", "changed", "to", "4pm"], "bio_tags": ["B-TEMPORAL", "I-TEMPORAL", "B-PARTICIPANT", "B-BELIEF_CUE", "B-PARTICIPANT", "B-PERCEPTION", "B-CONTENT", "I-CONTENT", "I-CONTENT", "I-CONTENT", "I-CONTENT", "I-CONTENT"]}
```

---

## Tokenisation Rules

- Split on whitespace. Each word is one token.
- Keep punctuation attached: `"3pm"` is one token, not `"3"` `"pm"`.
- Contractions stay together: `"wasn't"` is one token.
- If a sentence has no entities at all, every token gets `O`. These samples are valuable — include them.

---

## Target Volume

- **500-1000 samples** for initial training
- Aim for roughly even coverage across all 6 entity types
- Include ~10-15% all-O sentences (normal sentences with no extractable entities)
- Vary sentence structure: first-person diary style, short notes, longer narratives

