"""The simplest prompt chaining example, with no AI model.

The question is "Can I bike to the farmers market tomorrow morning?"
The answer comes from four steps that always run in the same order:

    gather  ->  decide  ->  draft  ->  check
    facts       a choice    a reply    is it right?

Each step's output is the next step's input. Nothing is picked at random,
so you always know what happens next. A ReAct agent chooses its next tool
at every turn. A chain does not.

Every step here is plain Python, so it runs for free. All data is pretend.

Run it:  python example.py
"""
import re  # Used by check() to find the times (like 09:10) in a reply.

# The bus trip, from the routing example.
LEAVE_HOME = "09:00"  # When you leave home.
BUS_OUT = ["09:10", "09:25", "09:40", "09:55", "10:10"]  # Buses that leave the stop.


# ---- The facts ----
# These four helpers fake a lookup. step_gather() calls all four, every time.

def check_weather() -> str:
    """Fake weather lookup. Used only by step_gather()."""
    return "Light rain until 9am, then sunny and 18C."


def check_calendar() -> str:
    """Fake calendar lookup. Used only by step_gather()."""
    return "You are free until noon."


def check_bike() -> str:
    """Fake bike check. Used only by step_gather()."""
    return "Your bike has a flat tire."


def check_bus() -> str:
    """Fake bus timetable lookup. Used only by step_gather()."""
    return "Bus 12 runs every 15 minutes to Market Square."


# ---- Step 1: gather ----

def step_gather(question: str) -> dict:
    """Step 1. Collect the facts. No decisions here.

    Input:  the question (not used yet, but kept so every step has the same shape).
    Output: a dictionary with the four facts.
    Called by: run_chain(), as the first step.
    """
    return {
        "weather": check_weather(),
        "calendar": check_calendar(),
        "bike": check_bike(),
        "bus": check_bus(),
    }


# ---- Step 2: decide ----

def step_decide(facts: dict) -> dict:
    """Step 2. Turn the facts into one choice: bike or bus.

    Input:  the facts dictionary from step 1.
    Output: {"way": "bus" or "bike", "bus_time": a time or None, "reason": text}.
    Called by: run_chain(), after step 1.
    How: if the bike has a flat tire, take the bus. Otherwise, use the bike.
    """
    if "flat tire" in facts["bike"]:
        return {"way": "bus", "bus_time": BUS_OUT[0],
                "reason": "the bike has a flat tire"}
    return {"way": "bike", "bus_time": None, "reason": "the bike is ready"}


# ---- Step 3: draft ----

def step_draft(decision: dict) -> str:
    """Step 3. Write the reply, using only the choice from step 2.

    Input:  the decision dictionary from step 2.
    Output: a short reply as text.
    Called by: run_chain(), after step 2. A broken version is passed in
    as draft_step to show what the check catches.
    """
    if decision["way"] == "bus":
        return (f"Your bike has a flat tire, so take Bus 12. Leave home at "
                f"{LEAVE_HOME} and catch the {decision['bus_time']} bus.")
    return f"Take your bike. Leave home at {LEAVE_HOME}."


def broken_draft(decision: dict) -> str:
    """A deliberately wrong draft (09:30 is not on the timetable).

    Used only in main(), to show that check() stops a bad reply.
    """
    return "Take Bus 12 at 09:30, leaving home at 09:00."


# ---- Step 4: check ----

def step_check(draft: str):
    """Step 4. A gate: only let a reply through if its times are correct.

    Input:  the reply text from step 3.
    Output: a pair (reply, verdict).
            If all times are on the timetable: (the reply, "OK: ...").
            If not: (None, "STOP: ..."), so nothing is sent.
    Called by: run_chain(), as the last step.
    How: find every HH:MM in the text and check each one is on the timetable.
    """
    times = re.findall(r"\d{2}:\d{2}", draft)  # Finds text like "09:10".
    allowed = set(BUS_OUT + [LEAVE_HOME])       # The only times that are allowed.
    wrong = [t for t in times if t not in allowed]
    if wrong:
        return None, f"STOP: the draft has times not on the timetable: {', '.join(wrong)}"
    return draft, "OK: every time is on the timetable."


# ---- Run the chain ----

def run_chain(question: str, draft_step=step_draft) -> None:
    """Run all four steps in order and print each result.

    Input:  the question, and optionally a different draft step (for the demo).
    Output: nothing is returned. It prints each step and the final answer.
    Called by: main(), once for the good run and once for the broken run.
    """
    print(f"You: {question}")
    facts = step_gather(question)                       # Step 1
    print(f"  1. gather  -> {len(facts)} facts: {facts}")
    decision = step_decide(facts)                       # Step 2 uses step 1's output.
    print(f"  2. decide  -> {decision}")
    draft = draft_step(decision)                        # Step 3 uses step 2's output.
    print(f"  3. draft   -> {draft}")
    reply, verdict = step_check(draft)                  # Step 4 uses step 3's output.
    print(f"  4. check   -> {verdict}")
    print(f"Answer: {reply if reply else 'No answer sent. The chain stopped at step 4.'}")
    print("Model calls: 0 (every step is plain code)\n")


def main() -> None:
    """Start here. Runs the good example, then the broken one, then the summary.

    Called by: the `if __name__ == "__main__"` line at the bottom, when you run the file.
    """
    run_chain("Can I bike to the farmers market tomorrow morning?")

    print("=== The same chain with a mistake in step 3 ===")
    run_chain("Can I bike to the farmers market tomorrow morning?",
              draft_step=broken_draft)
    print("The check caught the wrong time. The chain stopped before the reply\n"
          "went out. Each step is small, so each one is easy to test.\n")

    print("=== What chaining saved ===")
    print("The ReAct agent used 5 steps for this question: 4 tools and an answer.")
    print("The chain also has 4 steps, but the order is fixed and every step has")
    print("one job. With a real model (agent.py), only steps 2 and 3 call the")
    print("model: 2 calls, not 5.")


if __name__ == "__main__":
    main()  # Only runs when you run this file directly, not when it is imported.
