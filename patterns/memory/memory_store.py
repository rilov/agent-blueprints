"""A tiny long-term memory for the market assistant, saved in a JSON file.

It keeps two kinds of memory:

    facts     what is true now        "The bike's tire is fixed."
    episodes  what happened before    "Last Saturday the reply said 'great plan'
                                       and never said the bike was flat."

Every memory has a scope, so it only reaches the right people:

    {"user": "me"}            only me
    {"household": "home"}     everyone who lives at home (me and Sam)

The file is plain JSON, so you can open it and read what was remembered.
A real app would use a database or a vector store. The ideas stay the same.

Used by: example.py and agent.py.
"""
import json         # Reads and writes the memory file.
from pathlib import Path  # Works with file paths on any computer.

# Who lives where. Sam shares my home and my bike. Ben is a neighbour.
HOUSEHOLD = {"me": "home", "sam": "home", "ben": "ben-house"}


def can_see(person: str, scope: dict) -> bool:
    """Return True if this person may read a memory with this scope.

    Input:  a person ("me", "sam" or "ben") and a memory's scope.
    Output: True or False.
    Called by: MemoryStore.facts_for() and MemoryStore.lessons_for().
    How: a "user" memory is only for that user. A "household" memory is for
    everyone in that household. Anything else is hidden, so a mistake hides
    a memory instead of leaking it.
    """
    if "user" in scope:
        return scope["user"] == person
    if "household" in scope:
        return scope["household"] == HOUSEHOLD.get(person)
    return False


class MemoryStore:
    """Facts and episodes, saved to one JSON file after every change.

    Created by: main() in example.py and agent.py.
    """

    def __init__(self, path: str, fresh: bool = False):
        """Open the memory file. With fresh=True, start with an empty memory."""
        self.path = Path(path)
        if fresh or not self.path.exists():
            self.data = {"facts": [], "episodes": []}
            self._write()
        else:
            self.data = json.loads(self.path.read_text())

    def _write(self) -> None:
        """Save everything to the file. Called after every change."""
        self.path.write_text(json.dumps(self.data, indent=2))

    # ---- Facts: what is true now ----

    def save_fact(self, topic: str, text: str, scope: dict, source: str, date: str) -> None:
        """Remember one fact. A new fact on the same topic replaces the old one.

        Input:  topic   a short label, like "bike" or "rain"
                text    the fact, like "The bike's tire is fixed."
                scope   who may see it, like {"household": "home"}
                source  who said it, like "me" or "tool:check_bike"
                date    when, like "2026-11-25"
        Called by: remember_facts() in example.py and agent.py.
        How: the old fact is kept, but marked as replaced, so you can still
        see the history in the file. Only facts without "replaced_by" are used.
        """
        new_id = f"fact-{len(self.data['facts']) + 1}"
        for old in self.data["facts"]:
            if old["topic"] == topic and old["scope"] == scope and "replaced_by" not in old:
                old["replaced_by"] = new_id
        self.data["facts"].append({"id": new_id, "topic": topic, "text": text,
                                   "scope": scope, "source": source, "date": date})
        self._write()

    def facts_for(self, person: str) -> list:
        """Return the current facts this person may see.

        Called by: the recall step in example.py and agent.py.
        """
        return [f for f in self.data["facts"]
                if "replaced_by" not in f and can_see(person, f["scope"])]

    # ---- Episodes: what happened before ----

    def log_episode(self, goal: str, decision: str, reply: str, scope: dict, date: str) -> str:
        """Write down one finished run. Return its id, so feedback can be added later.

        Called by: the remember step in example.py and agent.py.
        """
        episode_id = f"run-{len(self.data['episodes']) + 1}"
        self.data["episodes"].append({"id": episode_id, "date": date, "goal": goal,
                                      "decision": decision, "reply": reply,
                                      "scope": scope, "lesson": None, "cause": None})
        self._write()
        return episode_id

    def add_feedback(self, episode_id: str, what_happened: str, cause: str, lesson=None) -> None:
        """Add what happened after a run, and the lesson if there is one.

        Input:  cause is "assistant_mistake" or "outside_world".
                lesson is a short rule for next time, or None.
        Called by: main() in example.py and agent.py, after I tell the assistant
        how it went.
        """
        for ep in self.data["episodes"]:
            if ep["id"] == episode_id:
                ep.update({"what_happened": what_happened, "cause": cause, "lesson": lesson})
        self._write()

    def lessons_for(self, person: str) -> list:
        """Return lessons from past runs this person may see.

        Only the assistant's own mistakes become lessons. If the bus was late
        because of roadworks, that is the outside world, not a lesson.
        Called by: the recall step in example.py and agent.py.
        """
        return [ep["lesson"] for ep in self.data["episodes"]
                if ep.get("lesson") and ep.get("cause") == "assistant_mistake"
                and can_see(person, ep["scope"])]
