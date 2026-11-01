"""Fan-out/fan-in as a real agent.

In example.py, the code decides which systems to call.
Here, the model decides. It chooses the tools, runs them at the same time,
reads the results, and decides what to do next.

This needs a real AI model, for example:
    MODEL=openai:gpt-5 python agent.py
    MODEL=openai:gpt-5 python agent.py "Is order A-1042 paid?"
"""
import asyncio
import os
import sys
import time

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

MAX_TURNS = 5  # Stops the agent if it never finishes.
TOOL_TIMEOUT = 2  # Seconds to wait for one tool.


# The tools. The model reads these descriptions to decide what to call.
@tool
async def check_store(order_id: str) -> str:
    """Check whether an order is confirmed and paid."""
    await asyncio.sleep(1)
    return "Order A-1042 is confirmed and paid."


@tool
async def check_warehouse(order_id: str) -> str:
    """Check whether an order has been packed."""
    await asyncio.sleep(1)
    return "Order A-1042 was packed yesterday."


@tool
async def check_courier(order_id: str) -> str:
    """Check whether the courier has collected an order."""
    await asyncio.sleep(1)
    return "Order A-1042 was not collected. Today's pickup was missed. Next try is tomorrow."


@tool
async def check_cancellation_policy() -> str:
    """Look up the rules for cancelling an order."""
    await asyncio.sleep(1)
    return "An order can be cancelled for free until the courier collects it."


TOOLS = {t.name: t for t in [check_store, check_warehouse, check_courier, check_cancellation_policy]}

INSTRUCTIONS = (
    "You are an order support assistant. Use the tools to answer the customer. "
    "Call only the tools you need. When tools don't depend on each other, call "
    "them together in one step. Use only what the tools tell you, and say what "
    "you could not confirm."
)


async def run_tool(call: dict) -> ToolMessage:
    """Run one tool. A failure becomes a message the model can read."""
    try:
        chosen = TOOLS[call["name"]]
        result = await asyncio.wait_for(chosen.ainvoke(call["args"]), TOOL_TIMEOUT)
    except Exception as exc:
        result = f"This tool failed: {type(exc).__name__}"
    return ToolMessage(content=result, tool_call_id=call["id"])


async def run_agent(question: str) -> str:
    model = init_chat_model(os.environ["MODEL"]).bind_tools(list(TOOLS.values()))
    messages = [SystemMessage(INSTRUCTIONS), HumanMessage(question)]

    for turn in range(1, MAX_TURNS + 1):
        # The model decides: call some tools, or answer.
        response = await model.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return StrOutputParser().invoke(response)

        # Fan-out: run every tool the model chose at the same time.
        names = [call["name"] for call in response.tool_calls]
        print(f"Turn {turn}: the model chose {', '.join(names)}")
        started = time.perf_counter()
        results = await asyncio.gather(*(run_tool(call) for call in response.tool_calls))
        print(f"        {len(results)} tool(s) finished in {time.perf_counter() - started:.1f}s")

        # Fan-in: give every result back to the model.
        messages.extend(results)

    return "I could not finish this request. Please ask a person to help."


async def main():
    if not os.environ.get("MODEL"):
        sys.exit("The agent needs an AI model. Set MODEL and your API key first.")

    question = sys.argv[1] if len(sys.argv) > 1 else (
        "Where's my order A-1042, and can I still cancel it?"
    )
    print(f"Customer: {question}\n")
    print(f"\nAgent: {await run_agent(question)}")


if __name__ == "__main__":
    asyncio.run(main())
