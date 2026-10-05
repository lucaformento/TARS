# Memory v2 plan: model-judged notes and check-ins

Status: **revision 4, approved by Codex for implementation after the memory
branch is installed and live-checked; nothing built.** Implements Luca's
October 2 design decisions: model-judged notes (1B), no blocked topics
except secrets, and ask-when-relevant review (3B)
with at most one check-in per day. Revisions 2–4 answer Codex's three October 2
engineering reviews.

## Behavior

1. **Quiet notes.** After a reply has been generated and spoken in full, a
   small model reads that exchange and applies one test: *is this about Luca,
   and will it still be true and useful a month from now?* It returns at most
   one short note, stored as `UNCONFIRMED`.
2. **Check-ins.** When an unconfirmed note is directly relevant, TARS may ask
   about it, at most once per local day: "You mentioned a printer arriving in
   November. Still the plan?"
3. **Confirmation.** If Luca confirms in his next turn, the note becomes
   `CONFIRMED`. If he says it is wrong or no longer true, it is removed.
4. Explicit commands ("remember…", "forget…", "what do you remember") are
   unchanged and take priority. Following Luca's description of "forget that"
   ("whatever the last comment or note he recorded"), a quiet note saved during
   this run counts as the newest memory saved. TARS says exactly what it forgot.

## Design

### 1. Only a fully successful turn produces a note

`brain.py` finalizes partial replies after an interruption, and in cloud mode
generation runs inside the lookahead worker. Neither is a safe trigger.

- `respond_stream` marks the turn **complete** only when the Anthropic stream
  ends on its own: its text loop is exhausted, not closed early or ended by an
  exception. Partial replies are still kept in history as today, but they
  never produce a turn record.
- At that point the brain builds an immutable turn record. It holds:
  - Luca's utterance;
  - the reply exactly as spoken, after marker handling;
  - whether the turn was a memory command;
  - the check-in outcome (section 4);
  - the store's command epoch (section 3).
- The **main thread** submits the record to the note-taker, and only after
  `speak_stream` returns without an exception, meaning the reply was fully
  generated and played. Failed, cancelled, or interrupted turns submit nothing,
  so no orphaned work can later change memory. `tars.py` (text mode) submits
  after `respond()` returns.
- A turn that was any memory command never schedules a note. That includes a
  rejected "remember" question and "what do you remember".

### 2. Note-taker worker

- One daemon thread holds at most one queued job. If a job is already queued,
  the new job is dropped with one diagnostic line.
- Each job runs in this order:
  1. Snapshot the store while holding the lock.
  2. Release the lock.
  3. Make the model request.
  4. Validate the response.
  5. Apply the result while holding the lock again, after the staleness checks
     in section 3.
- The worker has its own Anthropic client: an 8 s timeout, no retries, and
  `max_tokens` 80. Any failure (timeout, rate limit, network, invalid output)
  means no change and one diagnostic line.
- It never imports or touches the audio device. It never runs on the turn's
  critical path, and it never holds the store lock during a model request.
- On shutdown, the main thread sets a stop flag and waits at most 1 s. Any
  result that arrives after the stop is discarded; the apply step checks the
  flag while holding the lock.

### 3. Store transactions and stale results

- **Lock.** `MemoryStore` gains a reentrant lock. Every public method holds it
  across its whole read–modify–persist sequence. This includes the
  `_saved_this_run` bookkeeping, `load`, `for_prompt`, `latest`, and
  `spoken_summary`.
- **Copies.** Methods return copies, never live entries. Today `add` returns
  the stored dict itself.
- **Command epoch.** The store keeps an in-process **command epoch**. Each note
  job records the epoch when it is submitted. The apply step drops the result
  if the epoch has changed.
- **Explicit memory commands.** Every explicit memory command runs in two
  steps:
  1. **Wait.** Without holding the lock, wait up to 2 s for a queued or running
     note job to finish. "Forget that" then sees the newest quiet note.
  2. **Invalidate, then execute.** While holding the lock, increment the epoch,
     then run the command in the same critical section.

  The epoch increments for every command, including remember, forget,
  remember-last, recall, and rejected remember questions. It increments even
  when the command changes nothing; for example, "forget that" with nothing
  saved yet still invalidates outstanding work. A job still running after the
  wait therefore cannot save its note afterward. The wait applies only to
  memory-command turns, and the previous job normally finishes well before the
  next turn's transcription.
- **Check-in answers.** Applying a check-in confirm or retract also increments
  the epoch.
- **Quiet adds.** Quiet notes use a dedicated `add_note()`. It always stores
  `UNCONFIRMED`. If the note matches an existing entry under the identity rules,
  nothing happens: no promotion, no timestamp refresh, and no "saved this run"
  mark.

### 4. Check-ins, enforced in code

The application controls the daily allowance and makes each check-in
observable. What the code guarantees is narrower than what the prompt asks for:

- **Enforced:** at most one *marked* check-in sentence is spoken on any local
  date.
- **Model compliance, not enforced:** TARS asking for confirmation without a
  marker. The prompt forbids it, and the live session checks for it.

**Offer.** Before generation, the brain may offer check-in candidates. It offers
up to five, newest first, labeled `c1`–`c5`, and only when all of these hold:

- the turn is not a memory command;
- the state file is readable and writable;
- today's local date is available (state rules below);
- at least one unconfirmed note that is not a running joke exists.

Entry IDs stay local.

**Reserve.** Before the request is sent, the brain atomically writes a
reservation `{date, candidates}` to `memory-state.json`. If the write fails,
nothing is offered, check-ins are disabled for this run, and one diagnostic line
is printed.

**Marker filter.**
- The prompt tells TARS that it may ask about at most one candidate, only when
  directly relevant, as a single sentence that begins with `[check-in cN]`.
- A filter in `respond_stream` processes the reply sentence by sentence.
  It uses the same boundary function as the speech splitter, moved into a
  small shared module.
- Playback already waits for whole sentences, so this adds no audio delay. The
  console and the `first_text` diagnostic then advance per sentence instead of
  per delta.
- For each sentence:

  | Sentence | Result |
  | --- | --- |
  | First valid marker for an offered label | Marker removed; sentence spoken; check-in recorded |
  | Any later marked sentence | Whole sentence suppressed |
  | Marker with no offer this turn | Whole sentence suppressed |
  | Label that was not offered (for example `c4` when two were offered) | Whole sentence suppressed |
  | Malformed marker: text starting with `[check-in` that is not exactly `[check-in c1]`–`[check-in c5]` | Whole sentence suppressed |
  | Incomplete marker at the end of the stream | Whole sentence suppressed |
  | Ordinary bracket text such as `[laughs]` | Passes unchanged |

- Suppressed text never reaches speech, the console, or history. History holds
  exactly what was spoken.

**Settle** after the turn:
- **Complete turn with a recorded check-in:** the reservation date and the
  current local date both count as used (normally the same date; see Midnight
  below). Write `{entry_id, revision}` as the pending check-in. It is valid
  for the **next user turn in the same wake session only**.
- **Complete turn with no recorded check-in** (including turns where every
  marked sentence was suppressed): remove the reservation, so a later turn
  today may offer again. Nothing was asked aloud.
- **Failed or interrupted turn:** the reservation date and the current date
  count as used. Part of a question may have been spoken, so the code fails
  closed.
- **Leftover reservation at startup** (after a crash): the reservation date and
  the startup date both count as used. A crash can come after a question that
  played past midnight, so the code fails closed.

**Revision** means the entry's `updated` timestamp plus its text.

**State.** The state file keeps a set of **used dates**, not a single "last
date":
- Today is available only when it is not in the set.
- **Clock rollback:** a used date one or two days after today also makes today
  unavailable, because the clock probably moved backward.
- **Bad clock:** a used date more than two days ahead is treated as a
  bad-clock record. It stays in the set, so that date is already used if the
  calendar really reaches it, but it does not block today.
- **No pruning:** every used date is kept permanently (about 5 KB per year), so
  a clock rollback of any size finds an earlier used date still recorded. No
  repair ever deletes a used date.

**Midnight.** Two rules prevent two marked questions on one local date:
- A check-in whose turn crosses midnight consumes both dates.
- Each offer checks the current date at reservation time.

**Prompt wording.**
- The current memory prompt says to ask about UNCONFIRMED notes "if it
  matters". It becomes: never state an UNCONFIRMED note as fact, and do not ask
  Luca to confirm one except through the offered check-in.
- On the turn after a check-in, the reply prompt tells TARS to acknowledge
  Luca's answer briefly without claiming that memory changed. The update happens
  afterward, on the worker.

**State-file failures.**
- *Unreadable* (for example, permissions): check-ins are disabled for the run.
- *Corrupt:* the file is quarantined as `memory-state.corrupt.<hex>.json` and
  today counts as used. A fresh state file is written; if that write fails,
  check-ins are disabled for the run.
- Ordinary conversation and memory commands are unaffected in every case.

**Clock.** All dates are the Pi's local date, read from an injectable clock.

**Git.** `.gitignore` gains `/memory-state.json`, `/memory-state.json.*.tmp`,
and `/memory-state.corrupt*.json`.

### 5. Strict action schema

The model is asked for exactly one action through a forced tool call. Local
validation is authoritative.

| Action | Fields | Accepted only when |
| --- | --- | --- |
| `none` | — | always |
| `add` | `text` (string, 1–240 chars after normalization), `kind` (`fact` or `preference`) | the turn was not a memory command, the secret guard passes, and the command epoch is unchanged |
| `confirm` | — | a check-in is pending for this job, its entry still exists with the same revision, and the epoch is unchanged |
| `retract` | — | same as `confirm` |

- **Rejected output:** extra fields, missing or wrongly typed fields,
  unsupported actions or kinds (including `bit`), oversized text, non-JSON
  output, and more than one action are all treated as `none`, with one
  diagnostic line.
- **No IDs.** `confirm` and `retract` carry no ID; they can only refer to the
  pending entry. An unrelated "yes", a turn with no pending check-in, and an
  answer about an entry edited or deleted since the question all do nothing.
- **Effects.** `confirm` upgrades the entry to `CONFIRMED` and marks it as saved
  this run, matching today's explicit upgrade path. `retract` deletes it.

### 6. Secret guard

The guard runs on both Luca's utterance and the proposed note, so a paraphrase
cannot remove the context that marks something as secret. Any match rejects the
quiet `add`. An explicit "remember…" is unaffected.

The rule is **credential context or an unambiguous secret format**. A number or
code alone is not a secret. Phone numbers, years, ports, model numbers, order
counts, and zip codes stay eligible for quiet notes, matching Luca's "no blocked
topics except secrets".

- **Order.** Formats that depend on separators (the 3-2-4 social security
  pattern) are checked on the raw text first. Normalization runs afterward:
  lowercase the text, convert spoken digits ("four eight two one") to numerals,
  and join digit groups separated by spaces, dashes, or dots. The remaining
  checks run on the normalized text.
- **Credential terms (always block):**
  - password, passcode, passphrase;
  - "PIN" as a word;
  - security, verification, or one-time codes, OTP, and 2FA codes;
  - CVV and CVC;
  - account, card, and routing numbers;
  - social security number and SSN;
  - API keys, access tokens, and secret or private keys;
  - recovery or seed phrases.
- **Codes in credential context:**
  - Blocked: four or more digits, or an alphanumeric code, immediately
    following a keyword phrase such as "code is", "combination is", "PIN",
    "password was", or "lock code". Examples: "code is 4821", "the combination
    is 12 34 56", "the code is A7X9K2".
  - Exempt: "zip code", "postal code", "area code", "error code", and "status
    code".
  - Not a match: "I wrote code in 2026".
- **Unambiguous formats (block without context):**
  - a 13–19 digit run that passes the Luhn checksum (card numbers);
  - the 3-2-4 social security pattern (checked before normalization);
  - known API-key prefixes.

  A 10-digit phone number fails none of these, so it passes.
- **Diagnostics** never print rejected text or raw model output. They print only
  the action type and the outcome, for example `[memory note] rejected: secret
  guard`.

### 7. Bounded input, cost, and privacy

- **Character caps on the input** (enforced):
  - a fixed rubric;
  - the **same bounded memory block** the reply request already sends that turn
    (`for_prompt()`: at most 20 entries and 1,200 characters);
  - Luca's utterance and TARS's reply, each capped at 1,000 characters;
  - on a check-in answer turn only, the check-in sentence (capped at 300
    characters) and the note (at most 240).
- **Never sent:** the full store.
- **Output cap:** 80 tokens, enforced.
- **Token count:** about 1,500 input tokens, including the tool definition and
  tool-use overhead. This is an *estimate* derived from the character caps, not
  an enforced token budget.
- **Cost:** about $0.002 per completed turn at Haiku 4.5 list prices ($1 per
  million input tokens, $5 per million output). This is also an estimate.
  Diagnostics log only the API's reported token counts, and the rubric check
  measures real usage.
- **Model:** `claude-haiku-4-5-20251001`.
- **Privacy correction:** the note-taker makes a second Anthropic request each
  completed turn. It contains a second copy of data the reply request already
  sent that turn, and nothing outside it. It is still an additional request,
  under the same account data settings.

## Files (engineering)

| File | Change |
| --- | --- |
| `memory.py` | Lock, copies, command epoch, `add_note()` |
| `memory_notes.py` (new) | Worker, schema validation, secret guard, check-in state |
| `sentences.py` (new) | Shared sentence-boundary function used by brain and speech |
| `brain.py` | Completion flag, sentence-level marker filter, offer and settle, memory-command wait and invalidation |
| `personality.py` | UNCONFIRMED wording and the check-in block |
| `tars_voice.py`, `tars.py` | Shared splitter, submit after a successful turn, shutdown |
| `.gitignore` | State-file patterns |
| `tests/` | New test files for the cases below |

## Dependencies and order

1. Install and live-check the memory parsing branch (`99b4320`, `c20fa07`).
   This work builds on its session tracking.
2. Build in a new worktree from that installed commit.
3. Codex reviews; Luca approves the commit, install, and push separately.

## Tests (offline, deterministic, fake model and clock)

**Turn gating**
- A complete turn submits one record.
- Interrupted, cancelled, failed-speech, and failed-generation turns submit
  nothing.
- Every memory-command turn (including `remember_rejected` and `recall`)
  submits nothing.

**Schema**
- Valid `none`, `add`, `confirm`, and `retract` are accepted.
- Each rejection case in section 5 is treated as `none`.

**Stale results**
- A delayed `add` after a forget is dropped.
- "Forget that" waits for a queued note, then deletes it and names it.
- **Regression:** a job still running after the wait, followed by "forget that"
  that finds nothing to delete, leaves the late note unsaved.
- A recall or rejected remember also invalidates outstanding jobs.

**Check-in answers**
- A stale confirmation (entry edited or deleted since the question) does
  nothing.
- "Yes" with no pending check-in does nothing.
- A pending check-in expires after one turn and at sleep.

**Quiet adds**
- An add that matches a `CONFIRMED` or `UNCONFIRMED` entry changes nothing.

**Concurrency**
- A worker apply and an explicit forget are serialized and durable.
- The fake model asserts the store lock is not held during its call.
- With a queue full, the new job is dropped.
- A late result after shutdown is discarded.
- The worker never touches the audio device.

**Marker filter**
- Each row of the section 4 table: first valid marker, repeated markers in the
  same and later sentences, a marker without an offer, an unknown label, a
  malformed marker, and an incomplete marker at the end of the stream.
- Ordinary bracket text passes.
- A marker split across deltas is handled.
- Suppressed text is absent from speech, the console, and history.
- Spoken audio is unchanged for replies without markers.

**Check-in state**
- Reserve, settle with and without a recorded check-in, and release.
- A failed turn counts as used.
- **Crash across midnight:** a reservation made before midnight, followed by a
  crash after midnight and a restart that morning, uses both dates. No check-in
  is offered that morning.
- The limit survives a restart.
- **Midnight:** a reservation before midnight with the question settled after
  midnight, followed by a later offer that day, produces no second marked
  question on either date.
- Clock rollback by one day leaves today unavailable.
- A far-future bad clock does not block today, and its date stays used when the
  clock reaches it.
- A rollback of more than 14 days to an earlier used date finds it used. No
  used date is ever deleted.
- Corrupt state is quarantined, with no check-in that day.
- Unreadable or unwritable state disables check-ins while conversation
  continues.

**Secret guard**

Blocked:
- four-digit PINs;
- spoken digits;
- spaced card numbers;
- alphanumeric codes in context;
- the SSN pattern, with dashes and with spaces, exercised through the complete
  guard pipeline rather than the normalized text alone;
- key prefixes;
- credential phrasing in either the utterance or the note.

Allowed, tested alongside the blocked examples:
- phone numbers;
- years and dates;
- ports;
- model numbers;
- order counts;
- zip codes;
- "error code 404";
- `CS101`;
- "I wrote code in 2026".

Also:
- An explicit "remember" still saves.
- Captured diagnostics contain no rejected text.

**Rubric:** the fake model proves only the plumbing. Whether the real model
rejects trivia is measured separately (below).

## Real-model checks (paid; Luca approves each)

1. **Rubric check (optional, before the robot session).** About 20 scripted,
   disposable exchanges go straight to the note-taker, with no audio:
   - personal facts;
   - trivia questions;
   - small talk;
   - secret phrasing;
   - check-in answers.

   It reports only action counts and API token usage, never note text. The cost
   is a few cents.
2. **Live acceptance at the robot.** About ten natural turns:
   - "I'm getting a 3D printer around November 13 to build you a body" produces
     one `UNCONFIRMED` note.
   - General questions and small talk produce no notes.
   - A later related turn produces at most one check-in that day, and "yes"
     makes the note `CONFIRMED`.
   - No unmarked confirmation questions are asked.
   - "What do you remember?" lists the note with the correct label.
   - Disposable entries are removed afterward, so the real store keeps only what
     Luca wants.

## Implementation notes (phase 1)

Phase 1 builds the standalone parts in `memory_notes.py`: the worker, schema,
secret guard, check-in state, and marker filter. It also adds `sentences.py`,
the splitter now shared with `tars_voice.py`. `memory.py`, `brain.py`, and
`personality.py` are untouched until the honesty guard lands; phase 2 wires
everything in.

Deviations from revision 4, all engineering:

- **Pending check-in.** It stays in memory, not in the state file. It is valid
  only for the next turn of the same wake session, so a restart should end it
  anyway.
- **Output cap.** `max_tokens` is 120 instead of 80. A 240-character `add` plus
  its JSON can exceed 80 tokens. The rubric asks for notes under 150
  characters, and a truncated response is treated as `none`.
- **Model ID.** The model is `claude-haiku-4-5`, the current alias for the
  Haiku 4.5 snapshot.
- **No strict tool use.** The request does not set `strict: true` on the
  tool, so it cannot fail on model support for strict tools. Local validation
  is authoritative either way.
- **PIN.** "Pin" alone is usually hardware here ("GPIO pin 18"). Only these
  count as credentials:
  - "PIN number" or "PIN code";
  - "my PIN", "bank PIN", "card PIN", "debit PIN", "ATM PIN", "phone PIN", or
    "SIM PIN";
  - "pin" followed by four or more digits.
- **OTP and 2FA.** A bare "OTP" or "2FA" mention is allowed. "OTP code" and
  "2FA code" are blocked.
- **Extra fields on `none`.** They are ignored without a diagnostic line,
  because they cannot change anything.
- **Partial markers.** A partial marker such as `[che` at the end of a reply is
  suppressed rather than spoken.

After Codex's phase 1 review:

- **Secret guard.** The guard now ties a code word to a value instead of
  requiring them to be adjacent.
  - Code words: "code", "combination", "combo", "PIN", "passcode", and
    "password".
  - Matching values: four or more digits, or letters and digits together,
    written as numerals or spoken ("twelve thirty four", "a 7 x 9").
  - Allowed positions: right after the code word ("pin 4821"); after a
    connector such as "is", ":", or "'s" within six words ("the code to the
    safe is 1234", "the PIN for my card is 4821"); or before the code word
    ("4821 is the door code").
  - Credential terms such as "one-time code" are matched before number words
    become digits.
  - Accepted false positive: a sentence like "the code is 1000 lines" is not
    saved as a quiet note.
  - After Codex's second review, "OTP", "2FA", and "MFA" also act as code
    words. Account words ("account", "card", "IBAN", "routing") need a value
    with six or more digits. "My card is 4242424242424243" is therefore
    blocked, while "my graphics card is 4090" is allowed.
- **Check-in state shape.** The file must have exactly the expected keys,
  version 1, a list of `YYYY-MM-DD` strings, and a well-formed reservation.
  Anything else is quarantined as corrupt and today counts as used. A string,
  an empty mapping, or a partial record therefore cannot reopen the allowance.
- **Frozen memory block.** Each `TurnRecord` carries the memory block its reply
  used. `build_request` keeps whole lines within the 1,200-character prompt
  budget and drops any single oversized line. A job waiting in the queue
  therefore never sends newer store contents.
- **Phase 2 requirement.** Every guarded memory attempt (an unparsed
  remember/forget mention, or a forget that deletes nothing) counts as a memory
  command. It skips the note job and increments the command epoch, even when
  memory is unchanged.

## Implementation notes (phase 2)

Phase 2 wires the standalone parts into TARS. All 213 offline tests pass.

- **Store (`memory.py`).**
  - Every public method holds an `RLock` across read, modify, and persist, and
    returns copies.
  - `epoch` counts explicit memory commands.
  - `add_note` never promotes or refreshes an entry.
  - `confirm_checked` and `retract_checked` require the exact
    `entry_revision`.
  - `handle_memory_turn(..., infer=False)` turns off pattern inference while
    notes are on.
- **Brain (`brain.py`).** `respond_stream` handles each turn in this order:
  1. Classify the turn with `is_memory_request`. For a memory request, wait up
     to 2 s for the note-taker, without holding the lock.
  2. Holding the lock, increment the epoch and run the command.
  3. Reserve and offer check-in candidates.
  4. Filter the reply sentence by sentence through `MarkerFilter`.
  5. Settle the check-in.
  6. Store the turn's inputs only when the model stream finished on its own.

  `submit_note()` builds the `TurnRecord` with the current epoch. The front
  end calls it only after playback succeeded. `begin_session()` on each wake
  expires an unanswered check-in. `close()` stops the worker.
- **Front ends.** `tars_voice.py` calls `begin_session` at each wake and
  `submit_note` after `speak_stream` returns. It wraps the session in
  `closing(tars)`. `tars.py` submits after each reply and closes on exit.
  `TARS_MEMORY_NOTES=0` disables memory v2.
- **Prompt (`personality.py`).**
  - A standing rule says only the application changes memory, and TARS must
    never claim it just saved, updated, deleted, or forgot a memory without a
    control event. This closes the gap for synonyms such as "delete that" and
    "make a note".
  - UNCONFIRMED guesses are asked about only through an offered check-in.
  - On an answer turn, a control note tells TARS not to claim that memory
    changed.
- **Behavior change.** Replies are now yielded sentence by sentence, as the
  plan required. Spoken audio is unchanged, but the console and the
  `first_text` diagnostic advance per sentence instead of per delta.
