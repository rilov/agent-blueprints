"""The simplest fan-out/fan-in example.

One customer question goes to three systems at the same time,
and their answers are combined into one reply.

Run it for free:   python example.py
Run it with AI:    MODEL=openai:gpt-5 python example.py
"""
import asyncio
import os
import time

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel


# Pretend systems. Each one knows only one part of the answer.
async def ask_store(order_id):
    await asyncio.sleep(1)
    return "The order is confirmed and paid."


async def ask_warehouse(order_id):
    await asyncio.sleep(1)
    return "The parcel was packed yesterday."


async def ask_courier(order_id):
    await asyncio.sleep(1)
    return "We missed today's pickup. We will collect it tomorrow."


# Fan-out: ask all three at the same time.
fan_out = RunnableParallel(
    store=RunnableLambda(ask_store),
    warehouse=RunnableLambda(ask_warehouse),
    courier=RunnableLambda(ask_courier),
)


# Fan-in: turn the three answers into one reply.
def fan_in(question, answers):
    model_name = os.environ.get("MODEL")
    if not model_name:
        return " ".join(answers.values())  # No AI: just join the answers.

    model = init_chat_model(model_name) | StrOutputParser()
    return model.invoke(
        f"A customer asked: {question}\n"
        f"Our systems said: {answers}\n"
        "Reply to the customer in two short sentences. Use only these facts."
    )


async def main():
    question = "I paid three days ago. Where's my order?"

    started = time.perf_counter()
    answers = await fan_out.ainvoke("A-1042")
    seconds = time.perf_counter() - started

    for system, answer in answers.items():
        print(f"{system:>10}: {answer}")
    print(f"\nAll three answered in {seconds:.1f} seconds, not 3.")
    print(f"\nReply to customer: {fan_in(question, answers)}")


if __name__ == "__main__":
    asyncio.run(main())
