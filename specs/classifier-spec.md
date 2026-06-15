# Classifier Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 2.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `build_few_shot_prompt()` and
`classify_episode()` in `classifier.py`.

---

## build_few_shot_prompt(labeled_examples, description)

### What it does
Constructs a prompt string for the LLM that includes the task instructions,
all labeled training examples, and the new episode description to classify.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `labeled_examples` | `list[dict]` | Each dict has `"title"`, `"description"`, `"label"` (and others). These are the examples you labeled in Milestone 1. |
| `description` | `str` | The episode description to classify. |

### Output

| Return value | Type | Description |
|---|---|---|
| prompt | `str` | A complete prompt string ready to send to the LLM. |

---

### Spec fields — fill these in before writing code

**Task instruction (what should the LLM know about the task?):**

```
You are classifying podcast episodes by their format. Classify the episode
into exactly one of these four labels:

- interview: a conversation between a host and one or more guests
- solo: a single host speaking from memory, experience, or opinion — no guests,
  no assembled external sources
- panel: multiple guests with roughly equal speaking time, often debating or
  discussing a topic together
- narrative: a story assembled from external sources — interviews, archival
  audio, reporting — with a clear narrative arc

Return only the label and your reasoning. Do not explain the taxonomy.
```

---

**How should labeled examples be formatted in the prompt?**

```
Each example should include the episode title, a brief excerpt or the full
description, and the correct label. Separate examples with a blank line or
a delimiter like "---". Include all fields that help the model see why the
label was applied — title and description are both useful; other fields
(like episode ID) are not needed.
```

---

**Example block sketch (write one concrete example):**

```
Title: {title}
Description: {description}
Label: {label}
```

---

**How should the new episode (to be classified) be presented?**

```
Present it in the same format as the labeled examples, but omit the Label
line and replace it with an instruction to classify. For example:

Title: {title}
Description: {description}
Label: ?

Then add a line like: "Classify the episode above. Return your answer in
the format below:" followed by the output format you chose.
```

---

**What output format should you request from the LLM?**

```json
{
  "label": "interview",
  "reasoning": "Brief explanation of why this label was chosen."
}
```

Return valid JSON with two fields:
- `"label"`: one of {interview, solo, panel, narrative}
- `"reasoning"`: a brief explanation of the classification

JSON is unambiguous and easy to parse. The risk of malformed JSON is acceptable
because we handle it gracefully in Step 5 (parse errors return "unknown").

---

**Edge cases to handle in the prompt:**

```
1. Empty labeled_examples:
   If the list is empty, still include the taxonomy and task instructions.
   The LLM can do zero-shot classification using just the label definitions.
   Append a note: "(Note: no examples provided — classify based on definitions above)"

2. Very short description:
   No special handling needed. The model should classify based on whatever content
   is provided. Include the description as-is.

3. JSON parsing reliability:
   Explicitly instruct the LLM: "Return ONLY valid JSON in the format above.
   Do not include any text before or after the JSON."
   This reduces malformed responses.
```

---

## classify_episode(description, labeled_examples)

### What it does
Classifies a single podcast episode description using the few-shot LLM classifier.
Returns a dict with a label and reasoning.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `description` | `str` | The episode description to classify. |
| `labeled_examples` | `list[dict]` | Labeled training examples from `load_labeled_examples()`. |

### Output

| Return value | Type | Description |
|---|---|---|
| result | `dict` | Must have keys `"label"` and `"reasoning"`. `"label"` must be one of `VALID_LABELS` or `"unknown"`. |

---

### Spec fields — fill these in before writing code

**Step 1 — Build the prompt:**

```
Call build_few_shot_prompt(labeled_examples, description) and store the
returned string in a variable (e.g., prompt). Pass through both arguments
exactly as received — no modification needed before calling.
```

---

**Step 2 — Send to the LLM:**

```
Call _client.chat.completions.create() with:
  - model: the model name from config (LLM_MODEL)
  - messages: a list with one dict — {"role": "user", "content": prompt}
    (system-design.md shows an optional system message too — either shape works)
  - max_tokens: a reasonable limit (e.g., 200–300) to keep responses concise

Extract the response text from:
  response.choices[0].message.content
```

---

**Step 3 — Parse the response:**

```python
import json

response_text = response.choices[0].message.content.strip()
try:
    parsed = json.loads(response_text)
    label = parsed.get("label", "").lower()
    reasoning = parsed.get("reasoning", "")
except json.JSONDecodeError:
    # Handle in Step 5 — for now, proceed with fallback values
    label = None
    reasoning = None
```

Extract the `label` and `reasoning` fields from the JSON response.
Use `.strip()` to remove whitespace. Use `.get()` with defaults to handle
missing fields gracefully. If JSON parsing fails, catch the error and handle
in Step 5.

---

**Step 4 — Validate the label:**

```python
from config import VALID_LABELS

if label not in VALID_LABELS:
    label = "unknown"
```

After parsing, check that the extracted `label` is in `VALID_LABELS`
(defined in config.py: interview, solo, panel, narrative).
If the label is invalid, unexpected, or missing, set it to `"unknown"`.
This ensures the return dict always has a valid label value.

---

**Step 5 — Handle errors gracefully:**

```python
try:
    response = _client.chat.completions.create(...)
    response_text = response.choices[0].message.content.strip()
    parsed = json.loads(response_text)
    label = parsed.get("label", "").lower()
    reasoning = parsed.get("reasoning", "")
except (json.JSONDecodeError, KeyError, AttributeError) as e:
    # JSON parse error or malformed response
    label = "unknown"
    reasoning = f"Failed to parse response: {str(e)}"
except Exception as e:
    # Network error, API error, etc.
    label = "unknown"
    reasoning = f"API error: {str(e)}"

# Validate label after all error cases
if label not in VALID_LABELS:
    label = "unknown"

return {"label": label, "reasoning": reasoning}
```

The evaluation loop calls this function 20 times. One failure (malformed JSON, network
error, API timeout) should not crash the loop. Catch exceptions broadly, set label
to "unknown", and return a valid dict every time. The loop continues; incomplete
results are factored into accuracy metrics.

---

### Return value structure

```python
{
    "label": str,      # one of VALID_LABELS, or "unknown" if invalid/error
    "reasoning": str,  # brief explanation from the LLM
}
```

---

## Notes on label quality

The classifier is only as good as your labels. If your training examples have
inconsistent or ambiguous labels, the LLM will learn the wrong pattern.

Before implementing the classifier, re-read `data/taxonomy.md` and double-check
any labels you're unsure about. Annotation quality is part of the lab.

---

## Implementation Notes

*Fill this in after implementing and testing both functions.*

**Test: what does the raw LLM response look like for one episode?**

```
Episode tested: The Case for Four-Day Workweeks
Raw response text:
{
  "label": "solo",
  "reasoning": "The host is speaking from their own experience and opinion, laying out their case for a four-day workweek, without any guests or external sources."
}
```

**How did you parse the label out of the response?**

```
1. response_text = response.choices[0].message.content.strip()
   → Remove leading/trailing whitespace from the raw response

2. parsed = json.loads(response_text)
   → Parse the JSON string into a Python dict

3. label = parsed.get("label", "").lower()
   → Safely extract the "label" field (default to empty string if missing)
   → Convert to lowercase for consistent validation against VALID_LABELS

4. If json.loads() fails, catch JSONDecodeError and set label = "unknown"
```

**Did any episodes return `"unknown"`? If so, why?**

```
No
```

**One thing about the output format that surprised you:**

```
I was impressed with the LLM's reasoning explanations for the held-out test examples.
```
