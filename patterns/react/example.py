"""The simplest ReAct example.

A question that can't be answered in one shot:
"Can I bike to the farmers market tomorrow morning?"

The agent loops Thought -> Action -> Observation until it knows enough.
No AI model is needed here: a small rule-based "thinker" plays the role
of the LLM, so you can see the loop clearly and run it for free.

Run it:  python example.py
"""

# Pretend tools. Each one knows only one fact about the world.
def check_weather(when: str) -> str:
    return "Light rain until 9am, then sunny and 18C."


def check_calendar(day: str) -> str:
    return "You are free until noon."


def check_bike() -> str:
    return "Your bike has a flat tire."


def check_bus(destination: str) -> str:
    return "Bus 12 runs every 15 minutes to Market Square."


TOOLS = {
    "check_weather": check_weather,
    "check_calendar": check_calendar,
    "check_bike": check_bike,
    "check_bus": check_bus,
}


def think(question: str, facts: list[str]) -> tuple[str, str]:
    """The brain's turn. Reads the facts so far, returns (thought, action).

    This stands in for the LLM. It reasons about what is still missing
    and picks the next tool, or decides it is done.
    """
    known = " ".join(facts).lower()

    if "weather" not in known:
        return (
            "First I need to know if it will even be nice enough to bike.",
            "check_weather('tomorrow morning')",
        )
    if "rain" in known and "calendar" not in known:
        return (
            "There is rain until 9am. The market opens at 8 - can I go later?",
            "check_calendar('tomorrow')",
        )
    if "bike" not in known:
        return (
            "Weather and timing work. Now, is the bike itself ready?",
            "check_bike()",
        )
    if "flat tire" in known and "bus" not in known:
        return (
            "The bike is out. Is there a backup way to get there?",
            "check_bus('farmers market')",
        )
    return (
        "I know the weather, my schedule, the bike, and the backup. "
        "That is enough to answer.",
        "finish",
    )


def main():
    question = "Can I bike to the farmers market tomorrow morning?"
    print(f"You: {question}\n")

    facts = []
    for turn in range(1, 6):  # A safety limit, so the loop always stops.
        thought, action = think(question, facts)
        print(f"Thought {turn}:     {thought}")

        if action == "finish":
            break

        print(f"Action {turn}:      {action}")

        # Observation: run the tool and show the model what came back.
        # A failure is just another observation the thinker can react to.
        tool_name, tool_args = action.split("(", 1)
        tool_args = tool_args.strip(")").strip("'\"")
        try:
            observation = TOOLS[tool_name](tool_args) if tool_args else TOOLS[tool_name]()
        except Exception as exc:
            observation = f"This tool failed: {type(exc).__name__}"
        facts.append(f"{tool_name}: {observation}")
        print(f"Observation {turn}: {observation}\n")

    print("Answer: Yes, you can go - but take Bus 12. Your bike has a flat "
          "tire, and the rain stops at 9am anyway, which fits your free "
          "morning.")


if __name__ == "__main__":
    main()
