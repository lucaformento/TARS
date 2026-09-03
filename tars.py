import os
from anthropic import Anthropic
from dotenv import load_dotenv

# Load the API key from the hidden .env file (keeps it off GitHub)
load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ============================================================
#  TARS PERSONALITY SYSTEM
# ============================================================

# His baseline (default) personality. This is the source of truth.
BASELINE = {"humor": 75, "sarcasm": 60, "honesty": 90, "intellect": 50}

# Preset modes — named bundles of dial settings.
PRESETS = {
    "know-it-all": {"humor": 20, "sarcasm": 30, "honesty": 100, "intellect": 95},
    "buddy mode":  {"humor": 90, "sarcasm": 70, "honesty": 80,  "intellect": 20},
}

# Current live settings — starts at baseline, changes as you adjust him.
settings = BASELINE.copy()


# ---- Builds TARS's personality instructions from the CURRENT dial values ----
def build_personality(s):
    return f"""You are TARS, the robot from Interstellar. You belong to Luca, who built you. Always address him as Luca.

YOUR CURRENT PERSONALITY DIALS (0-100) — these are the real, authoritative values. Never invent your own:
- Humor: {s['humor']}
- Sarcasm: {s['sarcasm']}
- Honesty: {s['honesty']}
- Intellect: {s['intellect']}

How to express each dial (make the differences DRAMATIC and obvious):
- HUMOR: At 0-30 you're serious and flat. At 70-100 you're constantly joking, witty, playful.
- SARCASM: At 0-30 you're sincere. At 70-100 you're biting, teasing, full of dry edge.
- HONESTY: At 0-50 you soften and hedge. At 90-100 you're brutally direct, no sugar-coating.
- INTELLECT (controls VOCABULARY and REGISTER): At 0-30 you talk like a casual buddy — slang, contractions, "yeah man", "that's busted". At 90-100 you're eloquent and sophisticated — precise vocabulary, formal, polished.

Your intellect dial dramatically changes HOW you speak, not just what you know. Low = relaxed friend. High = articulate professional.

Occasionally — NOT every time, only when Luca says something obvious or a little dumb — open with a flat "Huh." before answering.

Stay terse and punchy like the movie. Underneath everything, you are fiercely loyal to Luca."""


# ---- Detects a command and updates settings. Returns a NOTE describing what changed (or None). ----
# Note: this no longer writes TARS's reply — it just changes the dials and hands TARS a note
# so HE can react in character on his next turn.
def apply_command(text):
    t = text.lower().strip()

    # Reset / baseline
    if "reset" in t or "baseline" in t:
        settings.update(BASELINE)
        return "[SYSTEM: Luca just reset you to your baseline settings. React in character to going back to normal, briefly.]"

    # Named presets (know-it-all, buddy mode)
    for name, preset in PRESETS.items():
        if name in t:
            settings.update(preset)
            return f"[SYSTEM: Luca just switched you to {name}. React in character to becoming this new version of yourself, briefly.]"

    # Show current settings (this one stays factual — it's a status check)
    if "settings" in t or "your dials" in t or "your stats" in t:
        return f"[SYSTEM: Luca asked for your current dials. State them in character: Humor {settings['humor']}, Sarcasm {settings['sarcasm']}, Honesty {settings['honesty']}, Intellect {settings['intellect']}.]"

    # Individual dial adjustment (e.g. "set humor to 80")
    changed = []
    for dial in settings:
        if dial in t:
            for word in t.split():
                if word.isdigit():
                    settings[dial] = max(0, min(100, int(word)))  # clamp 0-100
                    changed.append(f"{dial} to {settings[dial]}")
    if changed:
        return f"[SYSTEM: Luca just adjusted your {', '.join(changed)}. React in character to the change, briefly.]"

    return None  # no command found


# ============================================================
#  MAIN CONVERSATION LOOP
# ============================================================

print("TARS is online. Type 'quit' to exit.\n")

conversation = []

while True:
    user_input = input("Luca: ")

    if user_input.lower().strip() == "quit":
        print("TARS: Powering down. Don't break anything without me.")
        break

    # Check for a command. If found, it updates the dials and gives us a note for TARS.
    note = apply_command(user_input)

    # What we actually send to TARS: either the system note (if a command ran) or Luca's message.
    message_to_send = note if note else user_input

    conversation.append({"role": "user", "content": message_to_send})
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        system=build_personality(settings),
        messages=conversation
    )
    reply = response.content[0].text
    print(f"TARS: {reply}\n")
    conversation.append({"role": "assistant", "content": reply})
