"""LLM judge for student explanations (Part 2).

Given a student's explanation and the key points of the lesson, an LLM returns a verdict
(understood / partial / wrong) and short feedback: one thing right, one thing missed, one hint.

Works with any OpenAI-compatible chat API. Configure it with environment variables
(see .env.example): LLM_API_KEY, LLM_BASE_URL, LLM_MODEL. The key is never stored in code.

Quick demo:  python src/judge.py --n 3
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

VERDICTS = ("understood", "partial", "wrong")
TEXT_FIELDS = ("got_right", "missed", "hint")

PROMPTS = {
    "v1": (
        "You are a kind tutor. You judge a student's explanation of a lesson.",
        """Lesson:
{lesson}

Student explanation:
{explanation}

Decide the verdict: "understood" (covers the key points correctly), "partial" (some key points right, some missing) or "wrong" (mostly incorrect).
Then write short feedback for the student: one thing they got right, one thing they missed, and one hint that nudges them without giving the full answer.

Reply with JSON only: {{"verdict": "...", "got_right": "...", "missed": "...", "hint": "..."}}""",
    ),
}


def load_lessons(path=ROOT / "data" / "raw" / "lessons.md"):
    """Return {"L1": "<title>\\n<summary and key points>", ...} from lessons.md."""
    text = Path(path).read_text(encoding="utf-8")
    lessons = {}
    for block in re.split(r"^## ", text, flags=re.M)[1:]:
        title, body = block.split("\n", 1)
        lesson_id = title.split()[0]
        lessons[lesson_id] = f"{title.strip()}\n{body.split(chr(10) + '---')[0].strip()}"
    return lessons


def call_llm(system, user, retries=4):
    """One chat-completion call to an OpenAI-compatible API. Retries on rate limits and server errors."""
    key = os.environ.get("LLM_API_KEY")
    base = os.environ.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL")
    if not (key and base and model):
        raise RuntimeError("Set LLM_API_KEY, LLM_BASE_URL and LLM_MODEL in .env (see .env.example)")
    body = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    effort = os.environ.get("LLM_REASONING_EFFORT")    # optional: lower = faster for "thinking" models
    if effort:
        body["reasoning_effort"] = effort
    status = None
    for attempt in range(retries):
        try:
            r = requests.post(base.rstrip("/") + "/chat/completions",
                              headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
        except (requests.Timeout, requests.ConnectionError):
            status = "timeout or connection error"
            time.sleep(2 ** attempt * 2)
            continue
        status = r.status_code
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt * 2)
            continue
        if not r.ok:
            raise RuntimeError(f"LLM request failed: {r.status_code} {r.text[:200]}")
        return r.json()["choices"][0]["message"]["content"]
    raise RuntimeError(f"LLM request failed after {retries} tries (last status: {status})")


def parse_json(text):
    """Pull the JSON object out of the reply, even if the model wrapped it in code fences or extra text."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


def validate(d):
    """Check the reply has a valid verdict and the three feedback texts."""
    verdict = str(d.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        raise ValueError(f"bad verdict: {d.get('verdict')!r}")
    out = {"verdict": verdict}
    for k in TEXT_FIELDS:
        if not isinstance(d.get(k), str):
            raise ValueError(f"missing or non-text field: {k}")
        out[k] = d[k].strip()
    return out


def judge_one(explanation_id, lesson_id, text, lessons, version="v1"):
    """Judge one explanation. Returns explanation_id, verdict, got_right, missed, hint."""
    system, template = PROMPTS[version]
    user = template.replace("{lesson}", lessons[lesson_id]).replace("{explanation}", text)
    last_err = None
    for _ in range(2):                      # one retry if the reply is not usable JSON
        try:
            result = validate(parse_json(call_llm(system, user)))
            return {"explanation_id": explanation_id, **result}
        except ValueError as e:
            last_err = e
    return {"explanation_id": explanation_id, "verdict": "error", "got_right": "", "missed": "",
            "hint": "", "error": str(last_err)}


def run_batch(df, version="v1", out_path=None, delay=None):
    """Judge every row of df (needs explanation_id, lesson_id, explanation_text).

    Results are appended to a .jsonl file as they arrive, so a crash or rate limit
    does not lose work and a rerun skips what is already done.
    """
    lessons = load_lessons()
    out_path = Path(out_path or ROOT / "outputs" / f"judgments_{version}.jsonl")
    out_path.parent.mkdir(exist_ok=True)
    delay = float(os.environ.get("LLM_DELAY", 2)) if delay is None else delay

    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["explanation_id"]] = r

    with out_path.open("a", encoding="utf-8") as f:
        for i, row in enumerate(df.itertuples(), 1):
            if row.explanation_id in done and done[row.explanation_id]["verdict"] != "error":
                continue
            res = judge_one(row.explanation_id, row.lesson_id, row.explanation_text, lessons, version)
            done[row.explanation_id] = res
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"{i}/{len(df)} done")
            time.sleep(delay)
    return pd.DataFrame([done[i] for i in df.explanation_id])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Judge a few explanations and print the JSON")
    ap.add_argument("--n", type=int, default=3, help="how many explanations to judge")
    ap.add_argument("--version", default="v1", help="prompt version")
    args = ap.parse_args()
    expl = pd.read_csv(ROOT / "data" / "clean" / "explanations_clean.csv").head(args.n)
    lessons = load_lessons()
    for row in expl.itertuples():
        res = judge_one(row.explanation_id, row.lesson_id, row.explanation_text, lessons, args.version)
        print(json.dumps(res, indent=2, ensure_ascii=False))
