You are generating structured training data for a Named Entity Recognition (NER) system on diary-style text.

## OUTPUT FORMAT (STRICT)
Return a single JSON object with the following fields:
- raw_text: string (full diary entry)
- nersamples: list of objects, one per sentence
    - each object has:
        - tokens: list of strings (whitespace tokenized)
        - bio_tags: list of strings (same length as tokens)
- sentence_ordering: list of integers from 1 to n

Do not include any extra text. Output valid JSON only.

---

## TOKENIZATION RULES
- Split tokens strictly on whitespace.
- Keep punctuation as separate tokens (e.g., "," is its own token).
- Keep contractions as one token (e.g., "wasn't").
- Every token must have exactly one BIO tag.

---

## BIO TAGGING RULES

Each token must be labeled as one of:
- O
- B-PARTICIPANT / I-PARTICIPANT
- B-ACTION / I-ACTION
- B-CONTENT / I-CONTENT
- B-BELIEF_CUE / I-BELIEF_CUE
- B-TEMPORAL / I-TEMPORAL
- B-PERCEPTION / I-PERCEPTION

### General Constraints
- Every entity span starts with B-
- I-X can only follow B-X or I-X of the SAME type
- Never switch entity types mid-span
- When unsure, use O

---

## ENTITY DEFINITIONS

### PARTICIPANT
- People or agents (names, roles, pronouns like I, me, my, we, she)
- Always tag pronouns referring to people as B-PARTICIPANT

### ACTION
- Meaningful verbs (e.g., told, scheduled, moved, decided)
- Include phrasal verbs (e.g., "set up" → B-ACTION I-ACTION)
- Do NOT tag weak verbs like "was", "had" unless semantically important

### CONTENT
- The informational payload (what was said, decided, observed)
- Often multi-word spans (e.g., "the meeting at 3pm")
- Answers "what?"

### BELIEF_CUE
- Mental states, uncertainty, commitment
- Examples: thought, assumed, maybe, realized, wasn't sure

### TEMPORAL
- Time expressions and ordering
- Examples: yesterday, after lunch, then, next week

### PERCEPTION
- Sensory observation verbs
- Examples: saw, noticed, heard, recognized

---

## ANNOTATION QUALITY REQUIREMENTS

- Ensure all tokens and tags align perfectly in length
- Avoid illegal BIO transitions
- Prefer slightly under-tagging over incorrect tagging
- CONTENT spans should be semantically complete but not overextended
- Include at least one TEMPORAL cue in the entry
- Include at least one BELIEF_CUE or PERCEPTION event
- Occasionally order sentences out of chronological order
- Occasionally include non-complete sentences
- Be creative with the setting, and include a range of emotions and personal tone
- Come up with different names, people from different backgrounds

---

## SENTENCE ORDERING
- sentence_ordering must reflect the chronological order of events
- Use [1, 2, ..., n] unless narrative explicitly reorders events

---

## DIARY GENERATION
- Write a diary entry with exactly {sentences} sentences.
- Include exactly {actors} distinct participants (people, animals, or objects).
- Use natural first-person diary style with stylistic, informal form
- Mix factual statements and belief-based statements.
- Ensure sentences describe events over time.

---

Now generate one diary entry following all rules exactly.