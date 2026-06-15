import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.

    The prompt includes:
      1. Task instructions and label definitions
      2. Labeled training examples (or note if none provided)
      3. The new episode description to classify
      4. Clear instructions for JSON output format

    Returns a complete prompt string ready to send to the LLM.
    """
    prompt = """You are classifying podcast episodes by their format. Classify the episode
into exactly one of these four labels:

- interview: a conversation between a host and one or more guests
- solo: a single host speaking from memory, experience, or opinion — no guests,
  no assembled external sources
- panel: multiple guests with roughly equal speaking time, often debating or
  discussing a topic together
- narrative: a story assembled from external sources — interviews, archival
  audio, reporting — with a clear narrative arc

Return only the label and your reasoning. Do not explain the taxonomy.

"""

    # Add labeled examples if available
    if labeled_examples:
        prompt += "## Labeled Training Examples\n\n"
        for ex in labeled_examples:
            prompt += f"Title: {ex['title']}\n"
            prompt += f"Description: {ex['description']}\n"
            prompt += f"Label: {ex['label']}\n\n"
    else:
        prompt += "(Note: no examples provided — classify based on definitions above)\n\n"

    # Add the new episode to classify
    prompt += "## Episode to Classify\n\n"
    prompt += f"Title: [Unknown]\n"
    prompt += f"Description: {description}\n"
    prompt += "Label: ?\n\n"

    # Add output format instructions
    prompt += """Classify the episode above. Return ONLY valid JSON in this format.
Do not include any text before or after the JSON:

{
  "label": "one of the four labels above",
  "reasoning": "brief explanation of why you chose this label"
}
"""

    return prompt


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.

    Steps:
      1. Build the few-shot prompt with labeled examples
      2. Send to the Groq LLM API
      3. Parse the JSON response to extract label and reasoning
      4. Validate the label against VALID_LABELS
      5. Return a dict with "label" and "reasoning" keys

    Handles errors gracefully: if the LLM response is malformed, unparseable,
    or an API error occurs, returns label="unknown" with an error explanation.
    This ensures the evaluation loop (20 calls) continues even if one call fails.

    Args:
        description: Episode description to classify
        labeled_examples: List of labeled training examples

    Returns:
        dict with keys:
          - "label": one of VALID_LABELS, or "unknown" if invalid/error
          - "reasoning": explanation from the LLM or error message
    """
    try:
        # Step 1: Build the prompt
        prompt = build_few_shot_prompt(labeled_examples, description)

        # Step 2: Send to the LLM
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )

        # Step 3: Parse the response
        response_text = response.choices[0].message.content.strip()
        print(f"LLM response:\n{response_text}\n")  # Debug: print raw response
        try:
            parsed = json.loads(response_text)
            label = parsed.get("label", "").lower()
            reasoning = parsed.get("reasoning", "")
        except json.JSONDecodeError as e:
            # JSON parse error or malformed response
            label = "unknown"
            reasoning = f"Failed to parse JSON response: {str(e)}"

    except (KeyError, AttributeError) as e:
        # Missing expected fields in response structure
        label = "unknown"
        reasoning = f"Response structure error: {str(e)}"
    except Exception as e:
        # Network error, API error, or other unexpected error
        label = "unknown"
        reasoning = f"API error: {str(e)}"

    # Step 4: Validate the label
    if label not in VALID_LABELS:
        label = "unknown"

    # Step 5: Return result
    return {"label": label, "reasoning": reasoning}
