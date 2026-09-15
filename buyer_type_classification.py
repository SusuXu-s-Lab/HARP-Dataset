"""Buyer-type classification of property owner names.

Reference implementation of the classifier behind the ``buyer_type`` field of
the Harvey release. Each post-sale owner name from the Harris Central Appraisal
District (HCAD) record is labeled ``individual``, ``investor``, or ``unknown``
in two levels:

1. A rule-based pre-filter resolves clear cases. Blank or placeholder names are
   ``unknown`` and names containing organizational keywords (LLC, INC, TRUST,
   REALTY, ...) are ``investor``.
2. The remaining names are sent in batches to an LLM (gpt-4o-mini, temperature 0)
   restricted to the three labels.

Owner records are not redistributed. Running this file classifies a few made-up
names; the LLM step runs only when ``OPENAI_API_KEY`` is set.
"""

import json
import os
import re
import time

MODEL = "gpt-4o-mini"
BATCH_SIZE = 200
MAX_RETRIES = 4
LABELS = ("individual", "investor", "unknown")

PLACEHOLDER_NAMES = {"CURRENT OWNER", "UNKNOWN", "N/A", "NA", "", "NONE"}

ORGANIZATION_PATTERN = re.compile(
    r"\bLLC\b|\bL\.L\.C\b|\bINC\b|\bINCORPORATED\b|"
    r"\bCORP\b|\bCORPORATION\b|\bLTD\b|\bLIMITED\b|"
    r"\bLP\b|\bL\.P\b|\bPARTNERSHIP\b|\bTRUST\b|"
    r"\bHOLDINGS?\b|\bREALTY\b|\bREAL ESTATE\b|"
    r"\bINVESTMENT\b|\bINVESTORS?\b|\bPROPERTIES\b|\bPROPERTY\b|"
    r"\bHOMES?\b|\bGROUP\b|\bENTERPRISES?\b|\bFUND\b|"
    r"\bFINANCIAL\b|\bMANAGEMENT\b|\bCOMPANY\b|\bVENTURES?\b|"
    r"\bDEVELOPMENT\b|\bDEVELOPERS?\b|\bACQUISITIONS?\b|\bCAPITAL\b|"
    r"\bFOUNDATION\b|\bDISTRICT\b|\bCITY OF\b|\bCOUNTY OF\b|\bSTATE OF\b|"
    r"\bESTATE OF\b|\bOFFICE\b|\bAUTHORITY\b|\bCHURCH\b|\bMINISTRIES?\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = "You are a precise data labeling assistant."
USER_PROMPT = (
    "Classify each owner name into exactly one label: individual, investor, or unknown. "
    "Rules: people names -> individual; company/LLC/trust/government/non-person org -> investor; "
    "insufficient/garbled/blank -> unknown. "
    "Return strict JSON object with key 'results' containing a list of objects "
    "with keys 'owner_name' and 'buyer_type'. Do not omit any name."
)


def clean_label(label):
    """Normalize a model label; anything outside the allowed set becomes 'unknown'."""
    x = str(label).strip().lower()
    if x == "unkown":
        x = "unknown"
    return x if x in LABELS else "unknown"


def prefilter(name):
    """Level 1. Returns a label for clear cases, or None when the name needs the LLM."""
    n = str(name).strip()
    if n.upper() in PLACEHOLDER_NAMES:
        return "unknown"
    if ORGANIZATION_PATTERN.search(n.upper()):
        return "investor"
    return None


def classify_batch(client, names):
    """Level 2. One LLM request for a batch of names; names the model omits become 'unknown'."""
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT + "\n" + json.dumps({"names": names})},
        ],
        response_format={"type": "json_object"},
    )
    results = json.loads(resp.choices[0].message.content).get("results", [])
    labels = {str(r.get("owner_name", "")): clean_label(r.get("buyer_type", "unknown"))
              for r in results}
    return {n: labels.get(n, "unknown") for n in names}


def classify_batch_with_retry(client, names):
    """Retry with exponential backoff; a batch that keeps failing is labeled 'unknown'."""
    for attempt in range(MAX_RETRIES):
        try:
            return classify_batch(client, names)
        except Exception:
            time.sleep(2 ** attempt)
    return {n: "unknown" for n in names}


def classify_names(names, client=None):
    """Label every unique name: pre-filter first, then the LLM for what remains."""
    labels, pending = {}, []
    for n in dict.fromkeys(str(x).strip() for x in names):
        label = prefilter(n)
        if label is None:
            pending.append(n)
        else:
            labels[n] = label
    if pending:
        if client is None:
            raise ValueError(f"{len(pending)} names need the LLM step but no client was given")
        for i in range(0, len(pending), BATCH_SIZE):
            labels.update(classify_batch_with_retry(client, pending[i:i + BATCH_SIZE]))
    return labels


if __name__ == "__main__":
    examples = [
        "ABC REALTY LLC",
        "GULF COAST HOMES INC",
        "SMITH FAMILY TRUST",
        "CURRENT OWNER",
        "",
        "JOHN A SMITH",
        "MARIA GONZALEZ",
        "J & M RENTALS",
    ]

    print("Level 1: rule-based pre-filter")
    for n in examples:
        print(f"  {n!r:<24} -> {prefilter(n) or 'sent to LLM'}")

    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI

        print("\nLevel 2: LLM")
        for n, label in classify_names(examples, OpenAI()).items():
            print(f"  {n!r:<24} -> {label}")
    else:
        print("\nSet OPENAI_API_KEY to run the LLM step on the remaining names.")
