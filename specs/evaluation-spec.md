# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:
- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does
Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`. |

### Output

| Return value | Type | Description |
|---|---|---|
| accuracy | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
Accuracy = (Number of correct predictions) / (Total number of predictions)

Correct = predicted label exactly matches ground truth label
Divide by = total number of episodes
```

---

**Step-by-step logic:**

```
1. Initialize a counter for correct predictions to 0
2. Loop through each (prediction, ground_truth) pair
3. If prediction == ground_truth, increment the correct counter
4. After the loop, divide correct by the total length of the lists
5. Return the accuracy as a float between 0.0 and 1.0
```

---

**Edge case — what if both lists are empty?**

```
Return 1.0 (perfect accuracy). If there are no episodes to classify,
the classifier hasn't made any mistakes. This aligns with the convention
that an empty prediction set is considered perfect.
```

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

Position 0: "interview" == "interview"  ✓ correct
Position 1: "solo" == "solo"           ✓ correct
Position 2: "panel" == "solo"          ✗ incorrect
Position 3: "interview" == "narrative"  ✗ incorrect

Correct count = 2
Total count = 4
Accuracy = 2 / 4 = 0.5
```

---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does
Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order. |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
An episode counts as correctly classified for the "interview" class if:
  - The ground truth label IS "interview" (it truly is an interview), AND
  - The predicted label IS "interview" (we predicted it correctly).

In other words: ground_truth == "interview" AND prediction == "interview"

This is only counted toward the "interview" class metrics, not other classes.
```

---

**What does "total" mean for a given class?**

```
"Total" for a class is the count of episodes where the ground truth label
IS that class, regardless of whether we predicted it correctly.

For example, "total" for "interview" = count of all episodes with
ground_truth == "interview"

This is the denominator for computing per-class accuracy.
```

---

**Step-by-step logic:**

```
1. Initialize result dict with all VALID_LABELS, each set to:
   {"correct": 0, "total": 0, "accuracy": 0.0}

2. Loop through paired (prediction, ground_truth) indices

3. For each pair:
   a. Increment result[ground_truth]["total"] by 1
   b. If prediction == ground_truth:
      Increment result[ground_truth]["correct"] by 1

4. After the loop, for each label in result:
   a. If total > 0: accuracy = correct / total
   b. If total == 0: accuracy = 0.0 (no examples means no success)

5. Return the result dict
```

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
Set accuracy = 0.0 for that class. If we were never asked to predict
that class in the test set, we have no evidence of success. Setting it
to 0.0 is conservative and clear: we don't claim accuracy on unseen classes.
```

---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

Walking through each position:
   Pos 0: pred=interview, truth=interview → interview matches, total++ & correct++
   Pos 1: pred=interview, truth=solo      → solo total++, no match
   Pos 2: pred=solo,      truth=solo      → solo matches, total++ & correct++
   Pos 3: pred=panel,     truth=panel     → panel matches, total++ & correct++
   Pos 4: pred=panel,     truth=narrative → narrative total++, no match

label       correct  total  accuracy
----------  -------  -----  --------
interview   1        1      1.0
solo        1        2      0.5
panel       1        1      1.0
narrative   0        1      0.0
```

---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
   Why is per-class accuracy a more informative metric than overall accuracy alone?

   Overall accuracy hides class imbalance and systematic failures. If "narrative" episodes make up only 5% of the test set and the model always guesses wrong on them, the overall number barely moves — but per-class accuracy immediately surfaces that blind spot.

2. If `panel` episodes consistently get misclassified as `interview`, what does
   that tell you about your training labels or your prompt?

   It suggests the model can't reliably distinguish multiple guests discussing together (panel) from a host-and-guest conversation (interview). This could mean training examples for those two classes are ambiguous or too similar, or the prompt's label definitions don't give enough contrast between them — e.g., emphasizing "equal speaking time" and "debate" for panel more explicitly.

3. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
   How might the evaluation results change if you had labeled 100 training episodes?
   What if you had 200 test episodes?

   More training labels (100) would give the few-shot prompt richer, more diverse examples, likely improving accuracy on edge cases — though with 100% accuracy already, the gain here would only show up on harder or noisier inputs. More test episodes (200) would make the accuracy estimates more statistically reliable: with only 5 examples per class, a single misclassification swings per-class accuracy by 20%, so 200 test episodes would reveal subtle weaknesses that 20 episodes mask.
