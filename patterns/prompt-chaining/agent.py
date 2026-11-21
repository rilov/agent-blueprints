"""Prompt chaining with a real AI model, written with LangChain (LCEL).

The chain is a LangChain pipeline. Each step is joined with | :

    gather  ->  decide  ->  draft  ->  check
    code        model       model      code

Two steps call the model: decide (one word) and draft (one reply).
The order is fixed. The model never picks a tool or the next step.
So this is a chain, not an agent. The ReAct agent from Part 2 made about
5 model calls for the same question. This chain makes 2.

Run it (needs a real model and your API key):
    MODEL=openai:gpt-4.1 python agent.py
"""
import os   # Reads the MODEL environment variable.
import re   # Finds times like 09:10 in the reply.
import sys  # Stops the program with a clear message if MODEL is missing.

from langchain_core.callbacks import BaseCallbackHandler  # Lets us count model calls.
from langchain_core.output_parsers import StrOutputParser  # Turns a model reply into plain text.
from langchain_core.prompts import ChatPromptTemplate      # A prompt with blanks like {facts}.
from langchain_core.runnables import RunnableLambda        # Wraps one of our functions as a chain step.

# The bus trip, from the routing example.
LEAVE_HOME = "09:00"  # When you leave home.
BUS_OUT = ["09:10", "09:25", "09:40", "09:55", "10:10"]  # Buses that leave the stop.

# The prompt for step 2 (decide). The {facts} blank is filled in by decide_step().
DECIDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You help someone choose how to get to the farmers market. Use only "
               "the facts you are given. Reply with one word: bike or bus. Nothing else."),
    ("human", "{facts}"),
])

# The prompt for step 3 (draft). {allowed} and {way} are filled in by draft_step().
DRAFT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Write one or two short, friendly sentences to a person planning their "
               "morning. Use only these times, exactly as written, and no other times: "
               "{allowed}"),
    ("human", "The choice is: {way}."),
])


class CallCounter(BaseCallbackHandler):
    """Counts model calls, so the printout can show the real number.

    Where it is used: ask() creates one for each question and passes it in
    when it runs the chain. LangChain calls on_chat_model_start() every time
    a model is called, and we add 1.
    """

    def __init__(self):
        super().__init__()
        self.count = 0  # How many model calls so far.

    def on_chat_model_start(self, serialized, messages, **kwargs):
        """LangChain calls this each time a model is about to be called."""
        self.count += 1


# ---- The facts (step 1 uses these) ----
# These four helpers fake a lookup. gather() calls all four, every time.

def check_weather() -> str:
    """Fake weather lookup. Used only by gather()."""
    return "Light rain until 9am, then sunny and 18C."


def check_calendar() -> str:
    """Fake calendar lookup. Used only by gather()."""
    return "You are free until noon."


def check_bike() -> str:
    """Fake bike check. Used only by gather()."""
    return "Your bike has a flat tire."


def check_bus() -> str:
    """Fake bus timetable lookup. Used only by gather()."""
    return "Bus 12 runs every 15 minutes to Market Square."


# ---- Step 1: gather (code, no model) ----

def gather(question: str) -> dict:
    """Step 1. Collect the same four facts, in the same order, every time.

    Input:  the question (not used yet, but kept so every step has the same shape).
    Output: a dictionary with the four facts.
    Called by: the first step in build_chain().
    """
    facts = {
        "weather": check_weather(),
        "calendar": check_calendar(),
        "bike": check_bike(),
        "bus": check_bus(),
    }
    print(f"  1. gather  -> {len(facts)} facts")
    return facts


def build_chain(model):
    """Build the whole chain once, and return it.

    Input:  a chat model (for example, one made with init_chat_model()).
    Output: a chain you can call with .invoke(question).
    Called by: main(), once. The chain is then reused for every question.

    How it works: each step is a function. RunnableLambda wraps it, and the
    | symbol joins the steps so the output of one becomes the input of the next.
    """
    # Each prompt is joined to the model and a parser: prompt | model | parser.
    # The parser turns the model's reply into plain text.
    decide = DECIDE_PROMPT | model | StrOutputParser()
    draft = DRAFT_PROMPT | model | StrOutputParser()

    # ---- Step 2: decide (one model call) ----
    def decide_step(facts: dict, config) -> dict:
        """Step 2. The model picks "bike" or "bus" from the facts.

        Input:  the facts from gather().
        Output: {"way": "bike" or "bus", "bus_time": a time or None}.
        If the model answers anything else, it raises an error, and the chain stops.
        The config argument passes the call counter into the model call.
        """
        text = "\n".join(f"{name}: {fact}" for name, fact in facts.items())  # Facts as lines of text.
        word = decide.invoke({"facts": text}, config=config).strip().strip(".'\"`").lower()
        if word not in ("bike", "bus"):
            raise ValueError(f"The decide step answered '{word}', not bike or bus.")
        print(f"  2. decide  -> {word}")
        # Code finds the bus time. A model is not good at clock arithmetic.
        return {"way": word, "bus_time": BUS_OUT[0] if word == "bus" else None}

    # ---- Step 3: draft (one model call) ----
    def draft_step(decision: dict, config) -> str:
        """Step 3. The model writes the reply, using only the allowed times.

        Input:  the decision from decide_step().
        Output: the reply as text.
        The model only sees the choice and the allowed times, not the whole question.
        """
        if decision["way"] == "bus":
            allowed = f"Leave home at {LEAVE_HOME}. Take the {decision['bus_time']} bus."
        else:
            allowed = f"Leave home at {LEAVE_HOME} by bike."
        reply = draft.invoke({"allowed": allowed, "way": decision["way"]}, config=config)
        print(f"  3. draft   -> {reply}")
        return reply

    # ---- Step 4: check (code, no model) ----
    def check(draft_text: str) -> dict:
        """Step 4. A gate: only send the reply if every time is on the timetable.

        Input:  the reply text from draft_step().
        Output: {"reply": the text or None, "verdict": a short message}.
        If a time is wrong, reply is None, so nothing is sent.
        """
        times = re.findall(r"\d{2}:\d{2}", draft_text)  # Finds text like "09:10".
        wrong = [t for t in times if t not in set(BUS_OUT + [LEAVE_HOME])]
        if wrong:
            return {"reply": None,
                    "verdict": f"STOP: the draft has times not on the timetable: {', '.join(wrong)}"}
        return {"reply": draft_text, "verdict": "OK: every time is on the timetable."}

    # Join the four steps in order. Each step gets the previous step's output.
    return (
        RunnableLambda(gather)
        | RunnableLambda(decide_step)
        | RunnableLambda(draft_step)
        | RunnableLambda(check)
    )


def ask(chain, question: str) -> None:
    """Run the chain for one question and print the result.

    Input:  the chain from build_chain(), and a question.
    Output: nothing is returned. It prints each step, the answer and the model call count.
    Called by: main(), once for each question.
    How: a new CallCounter is made for each question, so the counts start at zero.
    If decide_step() raises an error, the chain stops, and we print why.
    """
    counter = CallCounter()
    print(f"You: {question}")
    try:
        result = chain.invoke(question, config={"callbacks": [counter]})
    except ValueError as exc:
        print(f"  2. decide  -> {exc} Stopping.")
        print(f"Model calls: {counter.count}\n")
        return
    print(f"  4. check   -> {result['verdict']}")
    print(f"Answer: {result['reply'] or 'No answer sent. The check stopped the chain.'}")
    print(f"Model calls: {counter.count} (1 to decide, 1 to draft)\n")


def main():
    """Start here. Reads MODEL, builds the chain, and asks the questions.

    Run it:  MODEL=openai:gpt-4.1 python agent.py
    You can also pass questions in:  python agent.py "your question here"
    """
    if not os.environ.get("MODEL"):
        sys.exit("The chain needs an AI model. Set MODEL and your API key first.")

    # Imported here, so the file can be read and tested without the full langchain package.
    from langchain.chat_models import init_chat_model

    chain = build_chain(init_chat_model(os.environ["MODEL"]))  # Build the chain once.

    questions = sys.argv[1:] or ["Can I bike to the farmers market tomorrow morning?"]
    for question in questions:
        ask(chain, question)


if __name__ == "__main__":
    main()  # Only runs when you run this file directly, not when it is imported.
