# %% [markdown]
# # Try fan-out/fan-in on real orders
#
# This uses the public Brazilian E-Commerce dataset by Olist: about 99,000 orders.
# Its tables stand in for three separate systems, so the assistant has to fan out.
# It runs for free. Set MODEL and your provider's API key to let an AI model write
# the answers and to run the voting example.

# %% setup
# Install:  pip install -r requirements.txt
import argparse
import asyncio
import operator
import os
import time
from collections import Counter
from typing import Annotated, Literal, Optional, TypedDict

import kagglehub
import pandas as pd
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel

# Any provider works, for example MODEL=anthropic:claude-opus-5-5 or MODEL=openai:gpt-5.
model = init_chat_model(os.environ["MODEL"]) if os.environ.get("MODEL") else None

# %% load
# Load the dataset. No Kaggle account is needed, and the download is about 43 MB.
path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")


def read(table: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(f"{path}/olist_{table}_dataset.csv", **kwargs)


orders = read("orders", parse_dates=[
    "order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date",
]).set_index("order_id")
payments = read("order_payments")
items = read("order_items", parse_dates=["shipping_limit_date"])
sellers = read("sellers").set_index("seller_id")
reviews = read("order_reviews")


def day(value) -> Optional[str]:
    return None if pd.isna(value) else str(value.date())


# %% sources
# Each function reads only what one real system would know.
def store_system(order_id: str) -> dict:
    order = orders.loc[order_id]
    paid = payments[payments.order_id == order_id]
    return {
        "status": order.order_status,
        "approved_on": day(order.order_approved_at),
        "paid": round(float(paid.payment_value.sum()), 2),
    }


def warehouse_system(order_id: str) -> dict:
    lines = items[items.order_id == order_id]
    return {
        "items": len(lines),
        "sellers": lines.seller_id.nunique(),
        "ship_by": day(lines.shipping_limit_date.max()) if len(lines) else None,
    }


def courier_system(order_id: str) -> dict:
    order = orders.loc[order_id]
    return {
        "collected_on": day(order.order_delivered_carrier_date),
        "delivered_on": day(order.order_delivered_customer_date),
        "estimated_on": day(order.order_estimated_delivery_date),
    }


# Local lookups are instant, so this adds the delay a real API would have.
LATENCY = {"store": 0.2, "warehouse": 0.3, "courier": 0.5}
DOWN: set[str] = set()  # Add a source name here to simulate an outage.


def source(name: str, system, timeout: float = 2.0) -> RunnableLambda:
    """Wrap a system so one failure cannot sink the whole request."""

    async def call(inputs: dict) -> dict:
        try:
            if name in DOWN:
                raise ConnectionError(f"{name} is down")
            await asyncio.wait_for(asyncio.sleep(LATENCY[name]), timeout)
            return {"status": "ok", "data": system(inputs["order_id"])}
        except Exception as exc:
            return {"status": "unavailable", "reason": type(exc).__name__}

    return RunnableLambda(call)


# %% sectioning
# Sectioning: different questions, different systems.
fan_out = RunnableParallel(
    store=source("store", store_system),
    warehouse=source("warehouse", warehouse_system),
    courier=source("courier", courier_system),
)

fan_in_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an order support assistant. Answer the customer's question "
     "using only the results below. If a source is unavailable, say what "
     "you could not confirm. Never guess.\n\nResults:\n{results}"),
    ("human", "{question}"),
])


def rule_based_answer(results: dict) -> str:
    """A free fan-in with no model, so anyone can run this."""
    data = {name: r["data"] for name, r in results.items() if r["status"] == "ok"}
    store, warehouse, courier = (data.get(n) for n in ("store", "warehouse", "courier"))
    missing = [n for n in ("store", "warehouse", "courier") if n not in data]
    parts = []

    if store and store["status"] in ("canceled", "unavailable"):
        label = "was canceled" if store["status"] == "canceled" else "could not be fulfilled"
        parts.append(f"This order {label}.")
    elif courier and courier["delivered_on"]:
        days = (pd.Timestamp(courier["delivered_on"]) - pd.Timestamp(courier["estimated_on"])).days
        timing = f"{days} days late" if days > 0 else "on time"
        parts.append(f"Your order was delivered on {courier['delivered_on']}, {timing}.")
    elif courier and courier["collected_on"]:
        parts.append(f"The courier collected your order on {courier['collected_on']} "
                     f"and estimates delivery by {courier['estimated_on']}.")
    elif courier:
        parts.append("The courier has not collected your order yet.")
    elif store:
        parts.append(f"The store shows your order as {store['status']}.")

    if warehouse and warehouse["items"]:
        parts.append(f"It has {warehouse['items']} item(s) from "
                     f"{warehouse['sellers']} seller(s).")
    if missing:
        parts.append(f"I could not confirm details from the {' or '.join(missing)} system.")
    return " ".join(parts) or "I could not confirm anything about this order right now."


async def where_is_my_order(order_id: str) -> None:
    question = "I paid for my order. Where is it?"
    started = time.perf_counter()
    results = await fan_out.ainvoke({"order_id": order_id})
    elapsed = time.perf_counter() - started

    for name, result in results.items():
        print(f"{name:>10}: {result.get('data', result)}")
    print(f"Fan-out took {elapsed:.2f}s. One after another: {sum(LATENCY.values()):.2f}s.")

    if model:
        chain = fan_in_prompt | model | StrOutputParser()
        reply = await chain.ainvoke({"results": results, "question": question})
    else:
        reply = rule_based_answer(results)
    print(f"\nFan-in answer: {reply}")


# %% samples
def sample_orders() -> dict[str, str]:
    """One real order for each situation, so you have something to try."""
    multi = items.groupby("order_id").seller_id.nunique()
    delivered = orders[orders.order_status == "delivered"].dropna(
        subset=["order_delivered_customer_date", "order_estimated_delivery_date"])
    late = (delivered.order_delivered_customer_date.dt.normalize()
            > delivered.order_estimated_delivery_date)
    picks = {
        "Delivered on time": delivered[~late].index,
        "Delivered late": delivered[late].index,
        "With the courier": orders[orders.order_status == "shipped"].index,
        "Not collected yet": orders[orders.order_status == "processing"].index,
        "Canceled": orders[orders.order_status == "canceled"].index,
        "Several sellers": multi[multi >= 3].index,
    }
    return {label: sorted(ids)[0] for label, ids in picks.items()}


# %% voting
# Voting: same question, three reviewers, on real customer reviews in Portuguese.
class Vote(BaseModel):
    received: Literal["yes", "no", "unclear"]
    reason: str


review_prompt = ChatPromptTemplate.from_messages([
    ("system", "You review customer comments, which are in Portuguese. {lens}"),
    ("human", "Review: {review}\n\nDid the customer receive their order?"),
])

LENSES = [
    "Read the comment literally. Do not infer anything it does not state.",
    "Focus on whether any part of the order is said to be missing.",
    "Look for wording that could mean a late, partial or wrong delivery.",
]


def disputed_reviews(limit: int = 5) -> pd.DataFrame:
    """Orders marked delivered where the customer says they did not receive them."""
    said_no = reviews.review_comment_message.str.contains(
        "não recebi|nao recebi", case=False, na=False)
    marked = orders.index[orders.order_status == "delivered"]
    return reviews[said_no & reviews.order_id.isin(marked)].sort_values("review_id").head(limit)


async def resolve_by_vote(order_id: str, review: str) -> dict:
    if model is None:
        return {"skipped": "Voting needs an AI model. Set MODEL to turn it on."}
    reviewer = review_prompt | model.with_structured_output(Vote, method="json_schema")

    # Fan-out: same question, three reviewers.
    votes = await reviewer.abatch([{"lens": lens, "review": review} for lens in LENSES])

    # Fan-in: compare the votes, then compare them with the delivery record.
    tally = Counter(vote.received for vote in votes)
    answer, count = tally.most_common(1)[0]
    delivered = courier_system(order_id)["delivered_on"] is not None
    if count < len(votes):
        return {"votes": dict(tally), "received": delivered, "decided_by": "delivery record"}
    if answer == "no" and delivered:
        return {"votes": dict(tally), "received": "disputed", "decided_by": "send to a person"}
    return {"votes": dict(tally), "received": answer, "decided_by": "unanimous vote"}


# %% dynamic
# Dynamic fan-out: one task per seller, and only the data decides how many.
class OrderState(TypedDict):
    order_id: str
    seller_ids: list[str]
    seller_reports: Annotated[list[dict], operator.add]
    summary: str


class SellerState(TypedDict):
    order_id: str
    seller_id: str


def load_sellers(state: OrderState) -> dict:
    lines = items[items.order_id == state["order_id"]]
    return {"seller_ids": sorted(lines.seller_id.unique())}


def fan_out_sellers(state: OrderState) -> list[Send]:
    return [Send("check_seller", {"order_id": state["order_id"], "seller_id": s})
            for s in state["seller_ids"]]


async def check_seller(state: SellerState) -> dict:
    await asyncio.sleep(0.3)
    lines = items[(items.order_id == state["order_id"]) & (items.seller_id == state["seller_id"])]
    seller = sellers.loc[state["seller_id"]]
    return {"seller_reports": [{
        "city": seller.seller_city,
        "items": len(lines),
        "ship_by": day(lines.shipping_limit_date.max()),
    }]}


def combine_sellers(state: OrderState) -> dict:
    reports = sorted(state["seller_reports"], key=lambda r: r["ship_by"])
    lines = [f"{r['items']} item(s) from {r['city']}, ship by {r['ship_by']}" for r in reports]
    return {"summary": f"{len(reports)} sellers: " + "; ".join(lines)}


builder = StateGraph(OrderState)
builder.add_node("load_sellers", load_sellers)
builder.add_node("check_seller", check_seller)
builder.add_node("combine_sellers", combine_sellers)
builder.add_edge(START, "load_sellers")
builder.add_conditional_edges("load_sellers", fan_out_sellers, ["check_seller"])
builder.add_edge("check_seller", "combine_sellers")
builder.add_edge("combine_sellers", END)
graph = builder.compile()


# %% cli
async def main() -> None:
    parser = argparse.ArgumentParser(description="Fan-out/fan-in on real orders.")
    parser.add_argument("--order", help="An order_id from the dataset.")
    parser.add_argument("--down", nargs="*", default=[], choices=list(LATENCY),
                        help="Systems to take offline.")
    args = parser.parse_args()
    DOWN.update(args.down)
    samples = sample_orders()

    print("Sample orders:")
    for label, order_id in samples.items():
        print(f"  {label}: {order_id}")

    print("\n1. Sectioning\n")
    await where_is_my_order(args.order or samples["Delivered late"])

    print("\n2. Voting\n")
    case = disputed_reviews().iloc[0]
    print(f"Review: {case.review_comment_message}")
    print(await resolve_by_vote(case.order_id, case.review_comment_message))

    print("\n3. Dynamic fan-out\n")
    state = await graph.ainvoke({"order_id": samples["Several sellers"]})
    print(state["summary"])


if __name__ == "__main__":
    asyncio.run(main())
