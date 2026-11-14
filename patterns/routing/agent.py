"""Routing with a real AI model.

This is example.py with one change: the router is a small model call
instead of keyword rules. So "Is cycling a good idea tomorrow?" now finds
the right route, even though it shares no words with the other questions.

    question -> router (1 small model call) -> one of four routes:

    fact   a small model answers from a short fact sheet   1 model call
    times  plain code works out the bus trip               0 model calls
    plan   the ReAct agent from Part 2                     about 5 model calls
    other  a fixed reply                                   0 model calls

This needs a real AI model, for example:
    MODEL=openai:gpt-4.1 python agent.py

Optional: a cheaper model for the router and the fact route:
    MODEL=openai:gpt-4.1 SMALL_MODEL=openai:gpt-4o-mini python agent.py
"""
import os
import sys

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

MAX_TURNS = 10  # Tool steps for the agent. Stops it if it never finishes.

ROUTES = ["fact", "times", "plan", "other"]

ROUTER_INSTRUCTIONS = (
    "You sort questions for a market-trip assistant. Do not answer the question. "
    "Reply with one word only, from this list:\n"
    "fact  - a simple fact about the market or the bus, like opening hours or the bus number\n"
    "times - asks for clock times of the bus trip, like when the user gets there or gets home\n"
    "plan  - asks whether or how to make the trip, like 'can I bike there tomorrow?'\n"
    "other - anything that is not about the trip to the market"
)

FACT_SHEET = (
    "The market opens at 08:00 and closes at 13:00. "
    "Bus 12 goes to Market Square. "
)

# The bus trip, from the evaluator-optimizer example.
LEAVE_HOME = "09:00"  # After the rain stops.
VISIT = 60            # Minutes at the market.
BUS_OUT = ["09:10", "09:25", "09:40", "09:55", "10:10"]
BUS_BACK = ["10:35", "10:50", "11:05", "11:20", "11:35"]
RIDE = 25
WALK = {"home to bus stop": 10, "bus stop to market": 5,
        "market to bus stop": 5, "bus stop to home": 5}

calls = 0  # Model calls for the current question.


def ask_model(model, messages: list):
    """Every model call goes through here, so we can count them."""
    global calls
    calls += 1
    return model.invoke(messages)


def route(small_model, question: str) -> str:
    """The router. One small model call that only picks a route."""
    reply = ask_model(small_model, [SystemMessage(ROUTER_INSTRUCTIONS), HumanMessage(question)])
    word = StrOutputParser().invoke(reply).strip().strip(".'\"`").lower()
    # Anything that is not on the list goes to 'other'. Never guess.
    return word if word in ROUTES else "other"


# Route 1: a small model answers from the fact sheet. No tools.
def answer_fact(small_model, big_model, question: str) -> str:
    reply = ask_model(small_model, [
        SystemMessage("Answer in one short sentence, using only these facts. "
                      "If the answer is not here, say you don't know. " + FACT_SHEET),
        HumanMessage(question),
    ])
    return StrOutputParser().invoke(reply)


# Route 2: time arithmetic, done in code so it is always right.
def minutes_of(hhmm: str) -> int:
    hours, mins = hhmm.split(":")
    return int(hours) * 60 + int(mins)


def clock(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def answer_times(small_model, big_model, question: str) -> str:
    at_stop = minutes_of(LEAVE_HOME) + WALK["home to bus stop"]
    bus_out = next(minutes_of(t) for t in BUS_OUT if minutes_of(t) >= at_stop)
    shop_start = bus_out + RIDE + WALK["bus stop to market"]
    shop_end = shop_start + VISIT
    at_stop = shop_end + WALK["market to bus stop"]
    bus_back = next(minutes_of(t) for t in BUS_BACK if minutes_of(t) >= at_stop)
    home = bus_back + RIDE + WALK["bus stop to home"]
    return (f"Leave at {LEAVE_HOME}, take the {clock(bus_out)} bus, shop "
            f"{clock(shop_start)}-{clock(shop_end)}, take the {clock(bus_back)} "
            f"bus back. You are home at {clock(home)}.")


# Route 3: the ReAct agent from Part 2, unchanged.
@tool
def check_weather(when: str) -> str:
    """Check the weather forecast for a time like 'tomorrow morning'."""
    return "Light rain until 9am, then sunny and 18C."


@tool
def check_calendar(day: str) -> str:
    """Check your calendar for a day like 'tomorrow' or 'saturday'."""
    return "You are free until noon."


@tool
def check_bike() -> str:
    """Check whether your bike is in working order."""
    return "Your bike has a flat tire."


@tool
def check_bus(destination: str) -> str:
    """Find a bus route to a destination."""
    return "Bus 12 runs every 15 minutes to Market Square."


TOOLS = {t.name: t for t in [check_weather, check_calendar, check_bike, check_bus]}

AGENT_INSTRUCTIONS = (
    "You are a helpful day-planning assistant. Think about what you still "
    "need to know, then call exactly one tool per step. Use only what the "
    "tools tell you. When you know enough, answer briefly."
)


def answer_plan(small_model, big_model, question: str) -> str:
    model = big_model.bind_tools(list(TOOLS.values()))
    messages = [SystemMessage(AGENT_INSTRUCTIONS), HumanMessage(question)]

    for turn in range(1, MAX_TURNS + 1):
        response = ask_model(model, messages)      # Thought
        messages.append(response)

        if not response.tool_calls:
            return StrOutputParser().invoke(response)

        # Run the first tool. Every other call still needs an answer, or the
        # API rejects the conversation - tell the model it was skipped.
        for i, call in enumerate(response.tool_calls):
            if i == 0:
                print(f"    Action: {call['name']}({call['args']})")
                try:
                    result = TOOLS[call["name"]].invoke(call["args"])
                except Exception as exc:
                    result = f"This tool failed: {type(exc).__name__}"
                print(f"    Observation: {result}")
            else:
                result = ("Skipped: call one tool at a time. "
                          "Try this again on your next turn.")
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    return "I could not figure this out in time. Please decide yourself."


# Route 4: everything else. Say what you can do. Don't make something up.
def answer_other(small_model, big_model, question: str) -> str:
    return "Sorry, I can only help with your trip to the market."


HANDLERS = {
    "fact": answer_fact,
    "times": answer_times,
    "plan": answer_plan,
    "other": answer_other,
}


def ask(small_model, big_model, question: str) -> None:
    global calls
    calls = 0
    print(f"You: {question}")
    name = route(small_model, question)                          # Step 1: pick a route.
    print(f"Router: -> {name}")
    answer = HANDLERS[name](small_model, big_model, question)    # Step 2: only that route works.
    print(f"Answer: {answer}")
    print(f"Model calls: {calls} (1 for the router, {calls - 1} for the route)\n")


def main():
    if not os.environ.get("MODEL"):
        sys.exit("The agent needs an AI model. Set MODEL and your API key first.")

    big_model = init_chat_model(os.environ["MODEL"])
    small_model = init_chat_model(os.environ.get("SMALL_MODEL", os.environ["MODEL"]))

    questions = sys.argv[1:] or [
        "What time does the market open?",
        "What time will I get home if I take the bus?",
        "Can I bike to the farmers market tomorrow morning?",
        "Can you book me a flight to Paris?",
        "Is cycling a good idea tomorrow?",
    ]
    for question in questions:
        ask(small_model, big_model, question)


if __name__ == "__main__":
    main()
