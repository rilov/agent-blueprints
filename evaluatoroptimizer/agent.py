"""Evaluator-optimizer as a real agent.

This is patterns/react/agent.py with one more step. The ReAct agent drafts
an answer. A separate model call, the evaluator, checks it against the
user's needs. If something is missing, the feedback goes back to the same
agent, which calls more tools and revises.

The evaluator has its own instructions and its own conversation. It sees
the question, the tool results and the draft - never the agent's messages.

This needs a real AI model, for example:
    MODEL=openai:gpt-4o-mini python agent.py
"""
import json
import os
import sys

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

MAX_TURNS = 10  # Tool steps per round. Stops the agent if it never finishes.
MAX_ROUNDS = 3  # Initial draft plus two revisions.

# What the user needs. These are demo settings, not part of the original question.
NEEDS = (
    "Taking the bus is fine if cycling is not possible. "
    "Leave home only after the rain stops. "
    "Spend at least 60 minutes at the market. "
    "Be home strictly before 12:00 (12:00 itself is too late)."
)


# The tools. The first four are the same as the ReAct example.
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


# New tools, for the details the evaluator will ask about.
@tool
def check_bus_times(direction: str) -> str:
    """Exact Bus 12 departure times. direction is 'to market' or 'home'."""
    if direction == "to market":
        return "Bus 12 to the market departs 09:10, 09:25, 09:40, 09:55, 10:10. Ride: 25 minutes."
    return "Bus 12 home departs 10:35, 10:50, 11:05, 11:20, 11:35. Ride: 25 minutes."


@tool
def check_walking_time(route: str) -> str:
    """Walking time for one of: 'home to bus stop', 'bus stop to market',
    'market to bus stop', 'bus stop to home'."""
    minutes = {"home to bus stop": 10, "bus stop to market": 5,
               "market to bus stop": 5, "bus stop to home": 5}
    return f"Walking {route}: {minutes[route]} minutes."


TOOLS = {t.name: t for t in [check_weather, check_calendar, check_bike, check_bus,
                             check_bus_times, check_walking_time]}

INSTRUCTIONS = (
    "You are a helpful day-planning assistant. Think about what you still "
    "need to know, then call exactly one tool per step. Use only what the "
    "tools tell you. When you know enough, answer briefly. If the plan uses "
    "the bus, give a timed itinerary from leaving home to arriving home. "
    f"The user's needs: {NEEDS}"
)

EVALUATOR_INSTRUCTIONS = (
    "You check a day plan written by another assistant. You cannot call tools. "
    "Compare the plan with the user's needs and the tool results. Check: does it "
    "answer the cycling question; does every time come from a tool result (not "
    "'every 15 minutes'); are walks at both ends of both bus rides counted; is the "
    "visit at least 60 minutes; is the arrival home strictly before 12:00; is "
    "every step's arithmetic right (11:59 or earlier is before 12:00). Do not "
    "invent facts. The assistant can look up: weather, calendar, bike status, bus "
    "routes, exact bus times 'to market' and 'home', and walking times for 'home to "
    "bus stop', 'bus stop to market', 'market to bus stop', 'bus stop to home'. "
    'Reply with only JSON: {"verdict": "PASS" or "REVISE", "feedback": ["..."]}. '
    "Use PASS only if every check passes. Each feedback item names one concrete "
    "gap and what to look up to fix it."
)

observations = []  # Every tool result, so the evaluator can check the facts.


def react(model, messages: list) -> str:
    """The ReAct loop from patterns/react/agent.py, on an ongoing conversation."""
    for turn in range(1, MAX_TURNS + 1):
        # Thought: the model reasons, then chooses a tool or the answer.
        response = model.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return StrOutputParser().invoke(response)

        # Action + Observation: run the first tool, give the result back.
        # Every other call still needs an answer, or the API rejects the
        # conversation - tell the model it was skipped so it can retry.
        for i, call in enumerate(response.tool_calls):
            if i == 0:
                print(f"Action: {call['name']}({call['args']})")
                try:
                    result = TOOLS[call["name"]].invoke(call["args"])
                except Exception as exc:
                    result = f"This tool failed: {type(exc).__name__}"
                print(f"Observation: {result}\n")
                observations.append(f"{call['name']}({call['args']}) -> {result}")
            else:
                result = ("Skipped: call one tool at a time. "
                          "Try this again on your next turn.")
            messages.append(
                ToolMessage(content=result, tool_call_id=call["id"]))

    return "I could not figure this out in time. Please decide yourself."


def evaluate(model, question: str, answer: str) -> dict:
    """A separate model call with its own instructions and a fresh conversation."""
    review = model.invoke([
        SystemMessage(EVALUATOR_INSTRUCTIONS),
        HumanMessage(f"Question: {question}\nUser's needs: {NEEDS}\n"
                     f"Tool results:\n" + "\n".join(observations) +
                     f"\n\nPlan to check:\n{answer}"),
    ])
    text = StrOutputParser().invoke(review).strip().strip("`").removeprefix("json").strip()
    try:
        result = json.loads(text)
        if result.get("verdict") in ("PASS", "REVISE"):
            return result
    except json.JSONDecodeError:
        pass
    return {"verdict": "REVISE", "feedback": ["The review could not be read. Check the plan again."]}


def run_agent(question: str) -> str:
    base = init_chat_model(os.environ["MODEL"])
    model = base.bind_tools(list(TOOLS.values()))
    messages = [SystemMessage(INSTRUCTIONS), HumanMessage(question)]

    for round_number in range(1, MAX_ROUNDS + 1):
        print(f"=== Round {round_number}: the ReAct agent ===\n")
        answer = react(model, messages)
        print(f"Draft answer:\n{answer}\n")

        # New: the evaluator checks the draft.
        review = evaluate(base, question, answer)
        print(f"=== Evaluator: {review['verdict']} ===")
        for item in review.get("feedback", []):
            print(f"    - {item}")
        print()

        if review["verdict"] == "PASS":
            return answer

        # The feedback goes back to the same agent, which revises.
        messages.append(HumanMessage(
            "A reviewer checked your answer and found these gaps:\n- "
            + "\n- ".join(review.get("feedback", []))
            + "\nUse tools to fill them, then give a complete revised answer."))

    return "I could not verify a plan in time. Please check the bus times yourself."


def main():
    if not os.environ.get("MODEL"):
        sys.exit("The agent needs an AI model. Set MODEL and your API key first.")

    question = sys.argv[1] if len(sys.argv) > 1 else (
        "Can I bike to the farmers market tomorrow morning?"
    )
    print(f"You: {question}\n")
    print(f"Answer: {run_agent(question)}")


if __name__ == "__main__":
    main()
