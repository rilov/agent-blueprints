"""The simplest evaluator-optimizer example.

This builds on the ReAct example (patterns/react/example.py). The ReAct loop
is the same. What's new is one more step after it:

    ReAct agent drafts an answer -> evaluator checks it -> feedback ->
    the same agent gathers what's missing and revises -> evaluator checks again

No AI model is needed here: a rule-based "thinker" plays the agent and a
rule-based evaluator plays the reviewer, so you can watch the loop for free.
All data is pretend.

Run it:  python example.py
"""
import re

MAX_ROUNDS = 3  # Initial draft plus two revisions.
MAX_TURNS = 10  # Tool steps per round, so the agent always stops.

# What the user needs. These are demo settings, not part of the original question.
NEEDS = {
    "leave_after": "09:00",   # Leave after the rain stops.
    "home_before": "12:00",   # Strictly before noon. 12:00 fails.
    "min_visit": 60,          # Minutes at the market.
}


# Pretend tools. The first four are the same as the ReAct example.
def check_weather(when: str) -> str:
    return "Light rain until 9am, then sunny and 18C."


def check_calendar(day: str) -> str:
    return "You are free until noon."


def check_bike() -> str:
    return "Your bike has a flat tire."


def check_bus(destination: str) -> str:
    return "Bus 12 runs every 15 minutes to Market Square."


# New tools. The draft never used them; the evaluator will ask for them.
def check_bus_times(direction: str) -> str:
    if direction == "to market":
        return "Bus 12 to the market departs 09:10, 09:25, 09:40, 09:55, 10:10. Ride: 25 minutes."
    return "Bus 12 home departs 10:35, 10:50, 11:05, 11:20, 11:35. Ride: 25 minutes."


def check_walking_time(route: str) -> str:
    minutes = {"home to bus stop": 10, "bus stop to market": 5,
               "market to bus stop": 5, "bus stop to home": 5}
    return f"Walking {route}: {minutes[route]} minutes."


TOOLS = {
    "check_weather": check_weather,
    "check_calendar": check_calendar,
    "check_bike": check_bike,
    "check_bus": check_bus,
    "check_bus_times": check_bus_times,
    "check_walking_time": check_walking_time,
}


def think(facts: list[str], feedback: list[tuple[str, str]]) -> tuple[str, str]:
    """The agent's turn. Same as the ReAct example, plus: act on feedback."""
    known = " ".join(facts).lower()

    if "weather" not in known:
        return ("First I need to know if it will even be nice enough to bike.",
                "check_weather('tomorrow morning')")
    if "rain" in known and "calendar" not in known:
        return ("There is rain until 9am. The market opens at 8 - can I go later?",
                "check_calendar('tomorrow')")
    if "bike" not in known:
        return ("Weather and timing work. Now, is the bike itself ready?", "check_bike()")
    if "flat tire" in known and "check_bus('" not in known:
        return ("The bike is out. Is there a backup way to get there?",
                "check_bus('farmers market')")

    # New: the evaluator said something is missing. Go and get it.
    for problem, action in feedback:
        if action not in " ".join(f.split(" -> ")[0] for f in facts):
            return (f"The reviewer said: {problem} Let me check.", action)

    return ("I have what I need. Time to write the answer.", "finish")


def clock(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def minutes_of(hhmm: str) -> int:
    hours, mins = hhmm.split(":")
    return int(hours) * 60 + int(mins)


def find(facts: list[str], action: str) -> str:
    """The observation for one tool call, or '' if it was never made."""
    return next((f.split(" -> ")[1] for f in facts if f.startswith(action + " ->")), "")


def plan_trip(facts: list[str]):
    """Build the bus itinerary from the facts, or None if facts are missing."""
    walk = {}
    for route in ("home to bus stop", "bus stop to market", "market to bus stop", "bus stop to home"):
        text = find(facts, f"check_walking_time('{route}')")
        if not text:
            return None
        walk[route] = int(re.search(r"(\d+) minutes", text).group(1))
    out, back = find(facts, "check_bus_times('to market')"), find(facts, "check_bus_times('home')")
    if not out or not back:
        return None
    ride = int(re.search(r"Ride: (\d+)", out).group(1))

    # Catch the first bus you can reach after leaving home.
    at_stop = minutes_of(NEEDS["leave_after"]) + walk["home to bus stop"]
    bus_out = next(minutes_of(t) for t in re.findall(r"\d\d:\d\d", out) if minutes_of(t) >= at_stop)
    shop_start = bus_out + ride + walk["bus stop to market"]
    shop_end = shop_start + NEEDS["min_visit"]
    at_stop = shop_end + walk["market to bus stop"]
    bus_back = next(minutes_of(t) for t in re.findall(r"\d\d:\d\d", back) if minutes_of(t) >= at_stop)
    home = bus_back + ride + walk["bus stop to home"]

    return [
        (bus_out - walk["home to bus stop"], bus_out, "Walk to the bus stop"),
        (bus_out, bus_out + ride, "Bus 12 to the market"),
        (bus_out + ride, shop_start, "Walk to the market"),
        (shop_start, shop_end, "Shop at the market"),
        (shop_end, at_stop, "Walk to the bus stop"),
        (at_stop, bus_back, "Wait for the bus"),
        (bus_back, bus_back + ride, "Bus 12 home"),
        (bus_back + ride, home, "Walk home"),
    ]


def write_answer(facts: list[str]) -> str:
    trip = plan_trip(facts)
    if trip is None:  # The original ReAct answer: it sounds right, but is it?
        return "Your bike has a flat tire. The rain stops at 9 AM, so take Bus 12 after 9."
    steps = "\n".join(f"    {clock(start)}-{clock(end)}  {what}" for start, end, what in trip)
    return f"Your bike has a flat tire, so take Bus 12:\n{steps}"


def evaluate(facts: list[str]) -> tuple[str, list[tuple[str, str]]]:
    """The evaluator. Checks the answer against the user's needs.

    Returns ("PASS" or "REVISE", feedback). Each feedback item is a problem
    and the tool call that would fix it.
    """
    feedback = []
    if not find(facts, "check_bus_times('to market')"):
        feedback.append(("'Every 15 minutes' is not a timetable. Which bus do you catch?",
                         "check_bus_times('to market')"))
    if not find(facts, "check_bus_times('home')"):
        feedback.append(("There is no way home. Can you be back before noon?",
                         "check_bus_times('home')"))
    for route in ("home to bus stop", "bus stop to market", "market to bus stop", "bus stop to home"):
        if not find(facts, f"check_walking_time('{route}')"):
            feedback.append((f"The walk {route} is not counted.",
                             f"check_walking_time('{route}')"))
    if feedback:
        return "REVISE", feedback

    # Everything is known: now check the actual numbers.
    trip = plan_trip(facts)
    home, visit = trip[-1][1], trip[3][1] - trip[3][0]
    if home >= minutes_of(NEEDS["home_before"]):
        return "REVISE", [(f"You get home at {clock(home)}, not before noon.", "finish")]
    if visit < NEEDS["min_visit"]:
        return "REVISE", [(f"Only {visit} minutes at the market.", "finish")]
    return "PASS", []


def main():
    question = "Can I bike to the farmers market tomorrow morning?"
    print(f"You: {question}")
    print(f"Demo settings: bus is fine, leave after {NEEDS['leave_after']}, "
          f"{NEEDS['min_visit']} minutes at the market, home before {NEEDS['home_before']}.\n")

    facts, feedback, turn = [], [], 0
    for round_number in range(1, MAX_ROUNDS + 1):
        print(f"=== Round {round_number}: the ReAct agent ===\n")

        # The ReAct loop, exactly as before.
        for _ in range(MAX_TURNS):
            turn += 1
            thought, action = think(facts, feedback)
            print(f"Thought {turn}:     {thought}")
            if action == "finish":
                break
            print(f"Action {turn}:      {action}")
            tool_name, tool_args = action.split("(", 1)
            tool_args = tool_args.strip(")").strip("'\"")
            try:
                observation = TOOLS[tool_name](tool_args) if tool_args else TOOLS[tool_name]()
            except Exception as exc:
                observation = f"This tool failed: {type(exc).__name__}"
            facts.append(f"{action} -> {observation}")
            print(f"Observation {turn}: {observation}\n")

        answer = write_answer(facts)
        print(f"\nDraft answer:\n    {answer}\n")

        # New: the evaluator checks the draft.
        verdict, feedback = evaluate(facts)
        print(f"=== Evaluator: {verdict} ===")
        for problem, _ in feedback:
            print(f"    - {problem}")
        print()

        if verdict == "PASS":
            trip = plan_trip(facts)
            home = trip[-1][1]
            margin = minutes_of(NEEDS["home_before"]) - home
            print(f"Answer: The draft above is verified. The bike is out, but Bus 12 works: "
                  f"{NEEDS['min_visit']} minutes at the market, home at {clock(home)}, "
                  f"{margin} minutes before noon.")
            return

    print("Answer: I could not verify a plan in time. Please check the bus times yourself.")


if __name__ == "__main__":
    main()
