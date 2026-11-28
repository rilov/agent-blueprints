"""The simplest memory example, with no AI model.

It is the chain from Part 5 (prompt chaining), with one step added at each end:

    recall  ->  gather  ->  decide  ->  draft  ->  check  ->  remember
    memory      facts       a choice    a reply    is it      write
    in                                             right?     back

recall reads what the assistant remembers about the person asking.
remember writes down what happened, so next week starts with it.
The memory is a JSON file (memory.json), so you can open it and look.

The story covers one week:
    Sat 21 Nov  I ask. The bike is flat. The reply is vague. I complain.
    Wed 25 Nov  I tell the assistant the bike is fixed, and that I don't
                like cycling in the rain.
    Fri 27 Nov  I ask again. Then Sam, who shares the bike, asks too.

Every step here is plain Python, so it runs for free. All data is pretend.

Run it:  python example.py
"""
import re                 # Used by check() to find the times (like 09:10) in a reply.
from pathlib import Path  # Puts memory.json next to this file.

from memory_store import MemoryStore  # The long-term memory, saved as JSON.

# The bus trip, from the earlier parts.
LEAVE_HOME = "09:00"  # When you leave home.
BUS_OUT = ["09:10", "09:25", "09:40", "09:55", "10:10"]  # Buses that leave the stop.
BIKE_LAST_CHECKED = "2026-11-21"  # The bike check below only knows about this day.
QUESTION = "Can I bike to the farmers market tomorrow morning?"


# ---- The facts ----
# The same four fake lookups as Part 5. gather() calls all four, every time.

def check_weather() -> str:
    """Fake weather lookup. Used only by gather()."""
    return "Light rain until 9am, then sunny and 18C."


def check_calendar() -> str:
    """Fake calendar lookup. Used only by gather()."""
    return "You are free until noon."


def check_bike() -> str:
    """Fake bike check. Used only by gather().

    It still says "flat tire", because it was last checked on 21 Nov.
    It doesn't know I fixed the tire on Wednesday. Only memory knows that.
    """
    return "Your bike has a flat tire."


def check_bus() -> str:
    """Fake bus timetable lookup. Used only by gather()."""
    return "Bus 12 runs every 15 minutes to Market Square."


def count(items: list, word: str) -> str:
    """Return "1 fact" or "2 facts". Used only for printing."""
    return f"{len(items)} {word}" + ("" if len(items) == 1 else "s")


# ---- Step 0: recall (new in Part 6) ----

def recall(state: dict) -> dict:
    """Step 0. Read what the assistant remembers about this person.

    Input:  the state, with "store" and "person".
    Output: the same state, plus "memory" (facts) and "lessons" (from past runs).
    Called by: run_chain(), before gather.
    The store only returns memories this person is allowed to see.
    """
    state["memory"] = state["store"].facts_for(state["person"])
    state["lessons"] = state["store"].lessons_for(state["person"])
    print(f"  0. recall   -> {count(state['memory'], 'fact')}, {count(state['lessons'], 'lesson')}")
    for fact in state["memory"]:
        print(f"                 fact:   {fact['text']} ({fact['date']}, from {fact['source']})")
    for lesson in state["lessons"]:
        print(f"                 lesson: {lesson}")
    return state


# ---- Step 1: gather (same as Part 5) ----

def gather(state: dict) -> dict:
    """Step 1. Collect the four facts. No decisions here."""
    state["facts"] = {
        "weather": check_weather(),
        "calendar": check_calendar(),
        "bike": check_bike(),
        "bus": check_bus(),
    }
    print(f"  1. gather   -> {len(state['facts'])} facts")
    return state


# ---- Step 2: decide (now uses memory) ----

def decide(state: dict) -> dict:
    """Step 2. Choose bike or bus, using the facts and the memory.

    Input:  the state, with "facts" and "memory".
    Output: the state, plus "decision": {"way", "bus_time", "reason"}.
    Called by: run_chain(), after gather.
    How:
      1. Which bike fact is newer? The bike check (21 Nov) or a remembered
         fact? The newer one wins. That's how "fixed" beats "flat".
      2. A flat bike means the bus.
      3. A working bike, rain when I leave, and "prefers the bus in rain"
         also means the bus.
      4. Otherwise, the bike.
    """
    facts, memory = state["facts"], state["memory"]
    bike = facts["bike"]
    for fact in memory:
        if fact["topic"] == "bike" and fact["date"] > BIKE_LAST_CHECKED:
            bike = fact["text"]  # A newer remembered fact beats the old check.

    raining_when_i_leave = "rain until 9am" in facts["weather"] and LEAVE_HOME <= "09:00"
    likes_bus_in_rain = any(f["topic"] == "rain" for f in memory)

    if "flat tire" in bike:
        decision = {"way": "bus", "reason": "the bike has a flat tire"}
    elif raining_when_i_leave and likes_bus_in_rain:
        decision = {"way": "bus", "reason": "your bike is fixed, but it's raining "
                                            "until 9 and you'd rather not cycle in the rain"}
    else:
        decision = {"way": "bike", "reason": "the bike is fixed"}
    decision["bus_time"] = BUS_OUT[0] if decision["way"] == "bus" else None
    state["decision"] = decision
    print(f"  2. decide   -> {decision['way']}, because {decision['reason']}")
    return state


# ---- Step 3: draft (now uses lessons) ----

def draft(state: dict) -> dict:
    """Step 3. Write the reply.

    Without lessons, it writes the reply a real model wrote in Part 5:
    friendly, with the right times, but it never says why, or which bus.
    With the lesson "start with the decision and the reason", it does.
    Called by: run_chain(), after decide.
    """
    d = state["decision"]
    if not state["lessons"]:
        if d["way"] == "bus":
            text = (f"Sounds like a great plan! Just remember to leave home at "
                    f"{LEAVE_HOME} to catch the {d['bus_time']} bus.")
        else:
            text = f"Sounds like a great plan! Leave home at {LEAVE_HOME}."
    elif d["way"] == "bus":
        text = (f"Take Bus 12, not the bike: {d['reason']}. "
                f"Leave home at {LEAVE_HOME} and catch the {d['bus_time']}.")
    else:
        text = f"Yes, take the bike: {d['reason']}. Leave home at {LEAVE_HOME}."
    state["reply"] = text
    print(f"  3. draft    -> {text}")
    return state


# ---- Step 4: check (same as Part 5) ----

def check(state: dict) -> dict:
    """Step 4. A gate: only let a reply through if every time is on the timetable."""
    times = re.findall(r"\d{2}:\d{2}", state["reply"])  # Finds text like "09:10".
    wrong = [t for t in times if t not in set(BUS_OUT + [LEAVE_HOME])]
    if wrong:
        state["reply"] = None
        print(f"  4. check    -> STOP: times not on the timetable: {', '.join(wrong)}")
    else:
        print("  4. check    -> OK: every time is on the timetable.")
    return state


# ---- Step 5: remember (new in Part 6) ----

def remember(state: dict) -> dict:
    """Step 5. Write this run down as an episode, so it can be learned from.

    The episode is shared with the household, so a lesson from my run also
    helps Sam. Called by: run_chain(), as the last step.
    """
    home = {"household": "home"} if state["person"] in ("me", "sam") else {"user": state["person"]}
    state["episode_id"] = state["store"].log_episode(
        goal=QUESTION, decision=state["decision"]["way"], reply=state["reply"],
        scope=home, date=state["today"])
    print(f"  5. remember -> saved as {state['episode_id']}")
    return state


CHAIN = [recall, gather, decide, draft, check, remember]


def run_chain(store: MemoryStore, person: str, today: str) -> str:
    """Run all six steps in order and print each result.

    Input:  the memory store, who is asking, and today's date.
    Output: the episode id, so feedback can be added to this run later.
    Called by: main().
    The state is one dictionary. Each step adds to it and hands it on.
    """
    print(f"{person.capitalize()} ({today}): {QUESTION}")
    state = {"store": store, "person": person, "today": today}
    for step in CHAIN:
        state = step(state)
    print(f"Answer: {state['reply'] or 'Nothing sent. The check stopped the chain.'}\n")
    return state["episode_id"]


def remember_facts(store: MemoryStore, person: str, message: str, today: str) -> None:
    """Pull lasting facts out of something a person tells the assistant.

    Here it's two simple rules, so it runs without a model.
    agent.py asks a real model to do this instead.
    Called by: main(), when someone tells the assistant something.
    """
    print(f"{person.capitalize()} ({today}): \"{message}\"")
    text = message.lower()
    saved = False
    if "flat" in text:
        store.save_fact("bike", "The bike's tire is flat.", {"household": "home"}, person, today)
        print("  saved fact: The bike's tire is flat. (whole household)")
        saved = True
    if "bike" in text and "fixed" in text:
        store.save_fact("bike", "The bike's tire is fixed.", {"household": "home"}, person, today)
        print("  saved fact: The bike's tire is fixed. (whole household, replaces 'flat')")
        saved = True
    if "rain" in text and "bus" in text:
        store.save_fact("rain", "Prefers the bus when it's raining.", {"user": person}, person, today)
        print(f"  saved fact: Prefers the bus when it's raining. (only {person})")
        saved = True
    if not saved:
        print("  nothing worth remembering")
    print()


def main() -> None:
    """Start here. Plays one week of the story, then shows what Ben can see."""
    store = MemoryStore(Path(__file__).with_name("memory.json"), fresh=True)

    # An older run from October. Bus 12 didn't come because of roadworks.
    # That's the outside world, not the assistant's mistake, so: no lesson.
    old = store.log_episode(QUESTION, "bus", "Take Bus 12 at 09:10.", {"household": "home"}, "2026-10-31")
    store.add_feedback(old, "Bus 12 didn't come. Roadworks.", cause="outside_world")

    print("=== Saturday: no memory yet ===")
    run_id = run_chain(store, "me", "2026-11-21")
    store.add_feedback(run_id, "The reply said 'great plan' and never said the bike was flat. "
                               "I nearly took the flat bike.",
                       cause="assistant_mistake",
                       lesson="Start with the decision and the reason. If it's the bus, name Bus 12.")
    print("Feedback saved: start with the decision and the reason.\n")
    remember_facts(store, "me", "Yes, the tire's flat. I'll fix it this week.", "2026-11-21")

    print("=== Wednesday: I tell it two things ===")
    remember_facts(store, "me", "Bike's fixed, by the way.", "2026-11-25")
    remember_facts(store, "me", "If it's raining, I'd rather take the bus.", "2026-11-25")

    print("=== Friday: same question, with memory ===")
    run_chain(store, "me", "2026-11-27")

    print("=== Friday: Sam asks the same question ===")
    run_chain(store, "sam", "2026-11-27")

    print("=== Ben, our neighbour, uses the same app ===")
    print(f"Ben can see {len(store.facts_for('ben'))} facts and "
          f"{len(store.lessons_for('ben'))} lessons from our house.\n")

    print("=== What memory changed ===")
    print("Same chain, same four facts, same model calls (0 here, 2 in agent.py).")
    print("Two steps were added: recall before, remember after.")
    print(f"Open {store.path.name} to see every fact and episode that was saved.")


if __name__ == "__main__":
    main()  # Only runs when you run this file directly, not when it is imported.
