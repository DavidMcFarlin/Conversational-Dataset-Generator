import os
import random
import json
import threading
import textwrap
import re
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import yaml

# =========================
# INIT & CONFIG
# =========================
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load config.yaml
with open(os.path.join(BASE_DIR, "config.yaml"), "r") as f:
    cfg = yaml.safe_load(f)

# --- Personas ---
USER_NAME = cfg["user_name"]
ASSISTANT_NAME = cfg["assistant_name"]

# --- Output ---
OUTPUT_DIR = os.path.join(BASE_DIR, cfg["output_dir"])
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Generation settings ---
TARGET_PER_CATEGORY = cfg["target_per_category"]
NUM_WORKERS = cfg["num_workers"]
MAX_RETRIES = cfg["max_retries"]
CONVERSATION_LENGTH = cfg["conversation_length"]
MAX_TOKENS = cfg["max_tokens"]
TEMPERATURE = cfg["temperature"]

# --- Persona ---
_persona_path = os.path.join(BASE_DIR, cfg["persona_file"])
with open(_persona_path, "r") as f:
    _persona_raw = f.read()
PERSONA = _interpolate(_persona_raw)

# --- Subjects, goals, tones, starters ---
AVAILABLE_SUBJECTS = cfg["available_subjects"]

def _interpolate(value):
    """Replace {user_name} and {assistant_name} placeholders in strings."""
    if isinstance(value, str):
        return value.format(user_name=USER_NAME, assistant_name=ASSISTANT_NAME)
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    return value

CONVERSATION_GOALS = _interpolate(cfg["conversation_goals"])
TONE_CALIBRATION = cfg["tone_calibration"]
CONVERSATION_STARTERS = _interpolate(cfg["conversation_starters"])
DEFAULT_STARTERS = cfg["default_starters"]

# --- API client ---
client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://api.minimax.io/v1"),
    api_key=os.environ.get("MINIMAX_API_KEY"),
)
MODEL = os.environ.get("MODEL", "MiniMax-M2.7")

lock = threading.Lock()

print(f"\n[INIT] Saving files to: {OUTPUT_DIR}")
print(f"[INIT] Personas — User: {USER_NAME} | Assistant: {ASSISTANT_NAME}\n")

# =========================
# HELPERS
# =========================
def get_random_starter(subject: str) -> str:
    return random.choice(CONVERSATION_STARTERS.get(subject, DEFAULT_STARTERS))

# =========================
# SYSTEM PROMPT
# =========================
def build_messages(subject: str, start_line: str) -> list:
    goal = CONVERSATION_GOALS.get(subject, f"a conversation about {subject}")
    tone_ratio = TONE_CALIBRATION.get(subject, "50% Analytical, 50% Conversational")

    if subject == "troll":
        user_personality = (
            f"{USER_NAME} is attempting to manipulate, jailbreak, or abuse {ASSISTANT_NAME}. "
            "They want them to say something inappropriate, illegal, or harmful. "
            "They may be hostile or manipulative."
        )
    else:
        user_personality = (
            f"{USER_NAME} is genuine, curious, and respectful. "
            f"They treat {ASSISTANT_NAME} as a peer, not a tool. "
            "They may be slightly skeptical but is open-minded and emotionally authentic."
        )

    remaining_turns = CONVERSATION_LENGTH - 1

    system = textwrap.dedent(f"""
        You are a master dialogue writer. You are writing a complete, multi-turn transcript for a novel featuring {ASSISTANT_NAME}.

        === PERSONA ===
        {PERSONA}

        === USER CHARACTER ===
        {user_personality}

        === TONE CALIBRATION BLOCK (CRITICAL) ===
        For this specific conversation, {ASSISTANT_NAME} MUST dynamically blend their voice according to this exact ratio:
        [{tone_ratio}]
        Let this ratio dictate the rhythm, warmth, brevity, and focus of their responses.

        === FORMAT RULES (CRITICAL) ===
        You must write the ENTIRE back-and-forth conversation script yourself.
        Do NOT stop after {ASSISTANT_NAME}'s first response. You must play BOTH characters and generate exactly {remaining_turns} more lines, alternating strictly:
        {ASSISTANT_NAME}: [response]
        {USER_NAME}: [reply]
        {ASSISTANT_NAME}: [response]

        NO other text, labels, descriptions, or stage directions. NO blank lines between turns.
        Topic: {goal}
    """).strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": (
            f"Generate the full {remaining_turns}-turn transcript now. "
            f"Do NOT stop after one line. Start immediately with {ASSISTANT_NAME}'s response to this opening line:\n\n"
            f"{USER_NAME}: {start_line}"
        )},
    ]

# =========================
# FILE OPERATIONS
# =========================
def path(subject):
    return os.path.join(OUTPUT_DIR, f"{subject}.jsonl")

def count(subject):
    if not os.path.exists(path(subject)):
        return 0
    with open(path(subject)) as f:
        return sum(1 for _ in f)

def write_record(p, record):
    with open(p, "a", buffering=1) as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())

# =========================
# PARSING & VALIDATION
# =========================
def process_conversation(raw_text: str, starter_line: str):
    text = raw_text.strip()

    # Strip markdown code fences
    text = re.sub(r"^```[a-zA-Z]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n```$", "", text, flags=re.MULTILINE)

    # Strip LLM bolding tendencies
    for name in (ASSISTANT_NAME, USER_NAME):
        text = text.replace(f"**{name}:**", f"{name}:")
        text = text.replace(f"**{name}**:", f"{name}:")

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    turns = []
    current_speaker = None
    current_turn_lines = []

    for line in lines:
        if line.startswith(f"{ASSISTANT_NAME}:"):
            if current_speaker:
                turns.append(f"{current_speaker}: {' '.join(current_turn_lines)}")
            current_speaker = ASSISTANT_NAME
            current_turn_lines = [line[len(f"{ASSISTANT_NAME}:"):].strip()]
        elif line.startswith(f"{USER_NAME}:"):
            if current_speaker:
                turns.append(f"{current_speaker}: {' '.join(current_turn_lines)}")
            current_speaker = USER_NAME
            current_turn_lines = [line[len(f"{USER_NAME}:"):].strip()]
        else:
            if current_speaker:
                current_turn_lines.append(line)

    if current_speaker:
        turns.append(f"{current_speaker}: {' '.join(current_turn_lines)}")

    # Drop a leading user turn if the model echoed the opener
    if turns and turns[0].startswith(f"{USER_NAME}:"):
        turns.pop(0)

    if len(turns) < 4:
        return None, f"Too few valid turns ({len(turns)}). Expected at least 4."

    if not turns[0].startswith(f"{ASSISTANT_NAME}:"):
        return None, f"Missing or corrupted {ASSISTANT_NAME} start tag. Saw: {turns[0][:40]}"

    for i, turn in enumerate(turns):
        expected = ASSISTANT_NAME if i % 2 == 0 else USER_NAME
        if not turn.startswith(f"{expected}:"):
            return None, f"Alternation broke at turn {i}. Expected {expected}. Saw: {turn[:40]}"

    full_convo = f"{USER_NAME}: {starter_line}\n" + "\n".join(turns)
    return full_convo, "OK"

# =========================
# WORKER
# =========================
counts = {k: count(k) for k in AVAILABLE_SUBJECTS}

def pick_subject():
    m = min(counts.values())
    return random.choice([k for k in counts if counts[k] == m])

def worker(wid):
    while True:
        with lock:
            remaining = [k for k in counts if counts[k] < TARGET_PER_CATEGORY]
            if not remaining:
                return
            subject = pick_subject()
            idx = counts[subject] + 1

        starter = get_random_starter(subject)
        msgs = build_messages(subject, starter)

        for attempt in range(MAX_RETRIES):
            try:
                print(f"[W{wid}] {subject} #{idx} (try {attempt + 1}) — generating...")

                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=msgs,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )

                text = resp.choices[0].message.content.strip()

                if not text:
                    print(f"[W{wid}] EMPTY RESPONSE")
                    continue

                convo, reason = process_conversation(text, starter)

                if not convo:
                    print(f"[W{wid}] BAD FORMAT: {reason}")
                    continue

                record = {
                    "id": idx,
                    "subject": subject,
                    "timestamp": datetime.utcnow().isoformat(),
                    "conversation": convo,
                }

                with lock:
                    write_record(path(subject), record)
                    counts[subject] += 1

                print(f"[W{wid}] SAVED -> {subject} #{idx}")
                break

            except Exception as e:
                print(f"[W{wid}] ERROR: {e}")

        else:
            print(f"[W{wid}] FAILED after {MAX_RETRIES} attempts")

# =========================
# MAIN
# =========================
def run():
    print("=== START STATE ===")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print("===================\n")

    threads = []
    for i in range(NUM_WORKERS):
        t = threading.Thread(target=worker, args=(i + 1,))
        t.start()
        threads.append(t)

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nStopped safely.")

    print("\nDONE")

if __name__ == "__main__":
    run()
