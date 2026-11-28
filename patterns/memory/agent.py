"""Memory with a real AI model, written with LangChain (LCEL).

The Part 5 chain, with recall added at the start and remember at the end:

    recall  ->  gather  ->  decide  ->  draft  ->  check  ->  remember
    code        code        model       model      code       code

Answering a question still takes 2 model calls (decide and draft), like Part 5.
Recall is a lookup in the memory file, not a model call.

One more model call happens when someone *tells* the assistant something,
like "Bike's fixed, by the way." The model pulls out the facts worth keeping.

Run it (needs a real model and your API key):
    MODEL=openai:gpt-4.1 python agent.py
"""
import json  # Reads the facts the model sends back.
import os    # Reads the MODEL environment variable.
import re    # Finds times like 09:10 in the reply.
import sys   # Stops the program with a clear message if MODEL is missing.
from pathlib import Path  # Puts memory-agent.json next to this file.

from langchain_core.callbacks import BaseCallbackHandler  # Lets us count model calls.
from langchain_core.output_parsers import StrOutputParser  # Turns a model reply into plain text.
from langchain_core.prompts import ChatPromptTemplate      # A prompt with blanks like {facts}.
from langchain_core.runnables import RunnableLambda        # Wraps one of our functions as a chain step.

from memory_store import MemoryStore  # The long-term memory, saved as JSON.

LEAVE_HOME = "09:00"
BUS_OUT = ["09:10", "09:25", "09:40", "09:55", "10:10"]
BIKE_LAST_CHECKED = "2026-11-21"  # The bike check doesn't know about later repairs.
QUESTION = "Can I bike to the farmers market tomorrow morning?"

# Step 2 (decide). The model sees the four facts AND what it remembers.
DECIDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You help someone choose how to get to the farmers market. Use only the "
               "facts and memories you are given. Each has a date. If a memory is newer "
               "than a check, trust the memory. Respect the person's preferences.\n"
               "Reply on one line, exactly like this: bus: <short reason> "
               "or bike: <short reason>"),
    ("human", "Facts:\n{facts}\n\nWhat you remember about this person:\n{memory}"),
])

# Step 3 (draft). Lessons from past runs go into the instructions.
DRAFT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Write one or two short sentences to a person planning their morning. "
               "Use only these times, exactly as written, and no other times: {allowed}\n"
               "Lessons from earlier replies that went wrong:\n{lessons}"),
    ("human", "The choice is: {way}. The reason is: {reason}."),
])

# Used by remember_facts(). Turns something a person said into facts to keep.
EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Pull out lasting facts worth remembering from the message. "
               "Ignore small talk. Never keep passwords or keys.\n"
               "Reply with only a JSON list. Each item has:\n"
               '  "topic": "bike" for the bike\'s condition, "rain" for how they like '
               'to travel in rain, or another one-word topic\n'
               '  "text": the fact, in one short sentence\n'
               '  "shared": true if it is about something the household shares, like the bike; '
               "false if it is a personal preference\n"
               "If there is nothing worth keeping, reply with []"),
    ("human", "{message}"),
])


class CallCounter(BaseCallbackHandler):
    """Counts model calls, so the printout can show the real number."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def on_chat_model_start(self, serialized, messages, **kwargs):
        """LangChain calls this each time a model is about to be called."""
        self.count += 1


# ---- The facts (step 1 uses these). The same fakes as Part 5. ----

def check_weather() -> str:
    return "Light rain until 9am, then sunny and 18C."


def check_calendar() -> str:
    return "You are free until noon."


def check_bike() -> str:
    """Still says flat: it was last checked on 21 Nov. Only memory knows it was fixed."""
    return "Your bike has a flat tire."


def check_bus() -> str:
    return "Bus 12 runs every 15 minutes to Market Square."


# ---- Steps that are plain code ----

def recall(state: dict) -> dict:
    """Step 0. Read the facts and lessons this person may see."""
    state["memory"] = state["store"].facts_for(state["person"])
    state["lessons"] = state["store"].lessons_for(state["person"])
    print(f"  0. recall   -> {len(state['memory'])} fact(s), {len(state['lessons'])} lesson(s)")
    return state


def gather(state: dict) -> dict:
    """Step 1. Collect the four facts, the same way every time."""
    state["facts"] = {
        "weather": check_weather(),
        "calendar": check_calendar(),
        f"bike (checked {BIKE_LAST_CHECKED})": check_bike(),
        "bus": check_bus(),
    }
    print(f"  1. gather   -> {len(state['facts'])} facts")
    return state


def check(state: dict) -> dict:
    """Step 4. A gate: only send the reply if every time is on the timetable."""
    times = re.findall(r"\d{2}:\d{2}", state["reply"])
    wrong = [t for t in times if t not in set(BUS_OUT + [LEAVE_HOME])]
    if wrong:
        state["reply"] = None
        print(f"  4. check    -> STOP: times not on the timetable: {', '.join(wrong)}")
    else:
        print("  4. check    -> OK: every time is on the timetable.")
    return state


def remember(state: dict) -> dict:
    """Step 5. Write the run down as an episode, shared with the household."""
    scope = {"household": "home"} if state["person"] in ("me", "sam") else {"user": state["person"]}
    state["episode_id"] = state["store"].log_episode(
        QUESTION, state["decision"]["way"], state["reply"], scope, state["today"])
    print(f"  5. remember -> saved as {state['episode_id']}")
    return state


def build_chain(model):
    """Build the six-step chain once, and return it.

    Input:  a chat model (for example, one made with init_chat_model()).
    Output: a chain you can call with .invoke(state).
    Called by: main(), once.
    The baton passed between steps is one dictionary, the state.
    Each step adds its result to it.
    """
    decide = DECIDE_PROMPT | model | StrOutputParser()
    draft = DRAFT_PROMPT | model | StrOutputParser()

    def decide_step(state: dict, config) -> dict:
        """Step 2. The model picks bike or bus, using facts and memory."""
        facts = "\n".join(f"- {name}: {text}" for name, text in state["facts"].items())
        memory = "\n".join(f"- {f['text']} (told on {f['date']})" for f in state["memory"]) or "- nothing yet"
        line = decide.invoke({"facts": facts, "memory": memory}, config=config).strip()
        way, _, reason = line.partition(":")
        way = way.strip().strip(".'\"`*").lower()
        if way not in ("bike", "bus"):
            raise ValueError(f"The decide step answered '{line}', not bike or bus.")
        # Code finds the bus time. A model is not good at clock arithmetic.
        state["decision"] = {"way": way, "reason": reason.strip(),
                             "bus_time": BUS_OUT[0] if way == "bus" else None}
        print(f"  2. decide   -> {way}: {reason.strip()}")
        return state

    def draft_step(state: dict, config) -> dict:
        """Step 3. The model writes the reply, following past lessons."""
        d = state["decision"]
        if d["way"] == "bus":
            allowed = f"Leave home at {LEAVE_HOME}. Take Bus 12 at {d['bus_time']}."
        else:
            allowed = f"Leave home at {LEAVE_HOME} by bike."
        lessons = "\n".join(f"- {lesson}" for lesson in state["lessons"]) or "- none yet"
        state["reply"] = draft.invoke({"allowed": allowed, "way": d["way"],
                                       "reason": d["reason"], "lessons": lessons}, config=config)
        print(f"  3. draft    -> {state['reply']}")
        return state

    return (
        RunnableLambda(recall)
        | RunnableLambda(gather)
        | RunnableLambda(decide_step)
        | RunnableLambda(draft_step)
        | RunnableLambda(check)
        | RunnableLambda(remember)
    )


def ask(chain, store: MemoryStore, person: str, today: str):
    """Run the chain for one question. Return the episode id, or None if it failed."""
    counter = CallCounter()
    print(f"{person.capitalize()} ({today}): {QUESTION}")
    state = {"store": store, "person": person, "today": today}
    try:
        state = chain.invoke(state, config={"callbacks": [counter]})
    except ValueError as exc:
        print(f"  2. decide   -> {exc} Stopping.\nModel calls: {counter.count}\n")
        return None
    print(f"Answer: {state['reply'] or 'Nothing sent. The check stopped the chain.'}")
    print(f"Model calls: {counter.count} (1 to decide, 1 to draft)\n")
    return state["episode_id"]


def remember_facts(model, store: MemoryStore, person: str, message: str, today: str) -> None:
    """Ask the model which facts in a message are worth keeping, and save them.

    One model call. Bike facts are shared with the household; preferences
    stay with the person who said them.
    """
    counter = CallCounter()
    extract = EXTRACT_PROMPT | model | StrOutputParser()
    print(f"{person.capitalize()} ({today}): \"{message}\"")
    raw = extract.invoke({"message": message}, config={"callbacks": [counter]})
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        facts = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  The model's answer was not a JSON list, so nothing was saved: {raw}\n")
        return
    for fact in facts:
        scope = {"household": "home"} if fact.get("shared") else {"user": person}
        store.save_fact(fact["topic"], fact["text"], scope, person, today)
        who = "whole household" if fact.get("shared") else f"only {person}"
        print(f"  saved fact: {fact['text']} ({who})")
    if not facts:
        print("  nothing worth remembering")
    print(f"Model calls: {counter.count}\n")


def main():
    """Start here. Plays the same week as example.py, with a real model.

    Run it:  MODEL=openai:gpt-4.1 python agent.py
    """
    if not os.environ.get("MODEL"):
        sys.exit("This needs an AI model. Set MODEL and your API key first.")

    # Imported here, so the file can be read and tested without the full langchain package.
    from langchain.chat_models import init_chat_model

    model = init_chat_model(os.environ["MODEL"])
    chain = build_chain(model)
    store = MemoryStore(Path(__file__).with_name("memory-agent.json"), fresh=True)

    old = store.log_episode(QUESTION, "bus", "Take Bus 12 at 09:10.", {"household": "home"}, "2026-10-31")
    store.add_feedback(old, "Bus 12 didn't come. Roadworks.", cause="outside_world")

    print("=== Saturday: no memory yet ===")
    run_id = ask(chain, store, "me", "2026-11-21")
    if run_id:
        store.add_feedback(run_id, "The reply didn't say clearly why not the bike.",
                           cause="assistant_mistake",
                           lesson="Start with the decision and the reason. If it's the bus, name Bus 12.")
        print("Feedback saved: start with the decision and the reason.\n")
    remember_facts(model, store, "me", "Yes, the tire's flat. I'll fix it this week.", "2026-11-21")

    print("=== Wednesday: I tell it two things ===")
    remember_facts(model, store, "me", "Bike's fixed, by the way.", "2026-11-25")
    remember_facts(model, store, "me", "If it's raining, I'd rather take the bus.", "2026-11-25")

    print("=== Friday: same question, with memory ===")
    ask(chain, store, "me", "2026-11-27")

    print("=== Friday: Sam asks the same question ===")
    ask(chain, store, "sam", "2026-11-27")

    print(f"Open {store.path.name} to see what the model chose to remember.")


if __name__ == "__main__":
    main()  # Only runs when you run this file directly, not when it is imported.
