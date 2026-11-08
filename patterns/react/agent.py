"""ReAct as a real agent.

In example.py, a rule-based thinker decides the next step.
Here, the model decides. It reasons about what is still missing, picks
one tool, reads the result, and loops until it can answer.

This needs a real AI model, for example:
    MODEL=openai:gpt-5 python agent.py
    MODEL=openai:gpt-5 python agent.py "Can I bike to the market tomorrow?"
"""
import os
import sys

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

MAX_TURNS = 6  # Stops the agent if it never finishes.


# The tools. The model reads these descriptions to decide what to call.
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

INSTRUCTIONS = (
    "You are a helpful day-planning assistant. Think about what you still "
    "need to know, then call exactly one tool per step. Use only what the "
    "tools tell you. When you know enough, answer in two short sentences."
)


def run_agent(question: str) -> str:
    model = init_chat_model(os.environ["MODEL"]).bind_tools(list(TOOLS.values()))
    messages = [SystemMessage(INSTRUCTIONS), HumanMessage(question)]

    for turn in range(1, MAX_TURNS + 1):
        # Thought: the model reasons, then chooses a tool or the answer.
        response = model.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return StrOutputParser().invoke(response)

        # Action + Observation: run the tool, give the result back.
        for call in response.tool_calls[:1]:  # One step at a time: ReAct.
            print(f"Action {turn}: {call['name']}({call['args']})")
            try:
                result = TOOLS[call["name"]].invoke(call["args"])
            except Exception as exc:
                result = f"This tool failed: {type(exc).__name__}"
            print(f"Observation {turn}: {result}\n")
            messages.append(
                ToolMessage(content=result, tool_call_id=call["id"]))

    return "I could not figure this out in time. Please decide yourself."


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
