"""The simplest routing example.

Same market morning as the ReAct and evaluator-optimizer examples. What's
new is one small step at the front:

    question -> router picks a route -> only that route does the work

Not every question needs the full agent. "What time does the market open?"
is a lookup. "When will I be home?" is arithmetic. Only "Can I bike there
tomorrow?" needs the agent.

No AI model is needed here: the router is a few keyword rules, so you can
watch it for free. All data is pretend.

Run it:  python example.py
"""

# Things we already know. No need to think, just look them up.
FACTS = [  # (words to look for, the fact)
    (["open", "close"], "The market opens at 08:00 and closes at 13:00."),
    (["which bus"], "Bus 12 goes to Market Square."),
]

# The bus trip, from the evaluator-optimizer example.
LEAVE_HOME = "09:00"  # After the rain stops.
VISIT = 60            # Minutes at the market.
BUS_OUT = ["09:10", "09:25", "09:40", "09:55", "10:10"]
BUS_BACK = ["10:35", "10:50", "11:05", "11:20", "11:35"]
RIDE = 25
WALK = {"home to bus stop": 10, "bus stop to market": 5,
        "market to bus stop": 5, "bus stop to home": 5}

# The router's rules: if the question has one of these words, take that route.
# The first match wins.
RULES = [
    ("fact", ["open", "close", "which bus"]),
    ("times", ["what time will", "when will", "how long"]),
    ("plan", ["can i", "should i"]),
]


def route(question: str) -> str:
    """The router. It does not answer. It only picks a route."""
    text = question.lower()
    for name, words in RULES:
        if any(word in text for word in words):
            return name
    return "other"  # Nothing matched. Never guess.


# Route 1: a lookup. No thinking at all.
def answer_fact(question: str):
    text = question.lower()
    for words, fact in FACTS:
        if any(word in text for word in words):
            return fact, "1 lookup"
    return "I don't have that fact.", "1 lookup"


# Route 2: time arithmetic, done in code so it is always right.
def minutes_of(hhmm: str) -> int:
    hours, mins = hhmm.split(":")
    return int(hours) * 60 + int(mins)


def clock(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def answer_times(question: str):
    at_stop = minutes_of(LEAVE_HOME) + WALK["home to bus stop"]
    bus_out = next(minutes_of(t) for t in BUS_OUT if minutes_of(t) >= at_stop)
    shop_start = bus_out + RIDE + WALK["bus stop to market"]
    shop_end = shop_start + VISIT
    at_stop = shop_end + WALK["market to bus stop"]
    bus_back = next(minutes_of(t) for t in BUS_BACK if minutes_of(t) >= at_stop)
    home = bus_back + RIDE + WALK["bus stop to home"]
    answer = (f"Leave at {LEAVE_HOME}, take the {clock(bus_out)} bus, shop "
              f"{clock(shop_start)}-{clock(shop_end)}, take the {clock(bus_back)} "
              f"bus back. You are home at {clock(home)}.")
    return answer, "code, no model"


# Route 3: the ReAct agent from Part 2. The only route that has to think.
def check_weather() -> str:
    return "Light rain until 9am, then sunny and 18C."


def check_calendar() -> str:
    return "You are free until noon."


def check_bike() -> str:
    return "Your bike has a flat tire."


def check_bus() -> str:
    return "Bus 12 runs every 15 minutes to Market Square."


AGENT_STEPS = [  # (thought, tool) - the same steps as the ReAct example.
    ("Will it be nice enough to bike?", check_weather),
    ("Rain until 9. Am I free later?", check_calendar),
    ("Is the bike ready?", check_bike),
    ("The bike is out. Is there a bus?", check_bus),
]


def answer_plan(question: str):
    for number, (thought, tool) in enumerate(AGENT_STEPS, 1):
        print(f"    Thought {number}: {thought}")
        print(f"    Tool {number}:    {tool.__name__}() -> {tool()}")
    print(f"    Thought {len(AGENT_STEPS) + 1}: I know enough to answer.")
    answer = "Your bike has a flat tire. The rain stops at 9 AM, so take Bus 12 after 9."
    return answer, f"{len(AGENT_STEPS) + 1} agent steps"


# Route 4: everything else. Say what you can do. Don't make something up.
def answer_other(question: str):
    return "Sorry, I can only help with your trip to the market.", "nothing"


HANDLERS = {
    "fact": answer_fact,
    "times": answer_times,
    "plan": answer_plan,
    "other": answer_other,
}


def ask(question: str):
    print(f"You: {question}")
    name = route(question)                 # Step 1: pick a route.
    print(f"Router: -> {name}")
    answer, work = HANDLERS[name](question)  # Step 2: only that route works.
    print(f"Answer: {answer}")
    print(f"Work:   {work}\n")
    return name, work


def main():
    questions = [
        "What time does the market open?",
        "What time will I get home if I take the bus?",
        "Can I bike to the farmers market tomorrow morning?",
        "Can you book me a flight to Paris?",
    ]
    results = [ask(question) for question in questions]

    agent_steps = len(AGENT_STEPS) + 1
    used = sum(agent_steps for name, _ in results if name == "plan")
    print("=== What routing saved ===")
    print(f"Without a router, all {len(questions)} questions go to the agent: "
          f"{len(questions)} x {agent_steps} = {len(questions) * agent_steps} agent steps.")
    print(f"With the router: {used} agent steps. The other questions never "
          f"needed the agent.\n")

    print("=== Where keyword rules fall short ===")
    ask("Is cycling a good idea tomorrow?")
    print("That was a miss. It is a planning question, but it has none of the\n"
          "router's words, so it went to 'other'. Rules only know the words you\n"
          "gave them. agent.py uses a small model as the router instead.")


if __name__ == "__main__":
    main()
