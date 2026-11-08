"""Try ReAct on a real question, with real weather.

"Can I bike to the farmers market tomorrow morning in <my city>?"

The weather tool calls the free Open-Meteo API (no key, no account), so
the agent reasons over a real forecast. The calendar, bike and bus tools
stay pretend - swap in your own data to make it yours.

Run it for free:      python try_it.py "Berlin"
Let an AI model loop: MODEL=openai:gpt-5 python try_it.py "Tokyo"
"""
import json
import os
import ssl
import sys
import urllib.request

MAX_TURNS = 6

# macOS Python installs sometimes ship without root certificates.
try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()


# --- Real tools: free APIs, no key needed ---------------------------------

def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "react-demo/1.0"})
    with urllib.request.urlopen(req, timeout=10, context=SSL_CONTEXT) as r:
        return json.load(r)


def check_weather(city: str) -> str:
    """Real forecast for tomorrow morning in any city."""
    geo = get_json(
        f"https://geocoding-api.open-meteo.com/v1/search"
        f"?name={city.replace(' ', '+')}&count=1"
    )
    if not geo.get("results"):
        return f"Could not find a place called {city}."
    place = geo["results"][0]

    fc = get_json(
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={place['latitude']}&longitude={place['longitude']}"
        f"&hourly=temperature_2m,precipitation_probability"
        f"&forecast_days=2&timezone=auto"
    )
    # Tomorrow morning = hours 24..35 (midnight to noon of the next day).
    temps = fc["hourly"]["temperature_2m"][24:36]
    rain = max(fc["hourly"]["precipitation_probability"][24:36])
    return (
        f"{place['name']}, {place.get('country', '')}: tomorrow morning "
        f"{min(temps)}-{max(temps)}C, up to {rain}% chance of rain."
    )


# --- Pretend tools: swap these for your real apps -------------------------

def check_calendar(day: str) -> str:
    """Check your calendar for a day like 'tomorrow'."""
    return "You are free until noon."


def check_bike() -> str:
    """Check whether your bike is in working order."""
    return "Your bike has a flat tire."


def check_bus(destination: str) -> str:
    """Find a bus route to a destination."""
    return "Bus 12 runs every 15 minutes to Market Square."


TOOLS = {
    "check_weather": check_weather,
    "check_calendar": check_calendar,
    "check_bike": check_bike,
    "check_bus": check_bus,
}

DESCRIPTIONS = {
    "check_weather": "check_weather('<city>') - real forecast",
    "check_calendar": "check_calendar('<day>') - your free time",
    "check_bike": "check_bike() - is your bike working",
    "check_bus": "check_bus('<place>') - a backup route",
}


# --- Option 1: a rule-based thinker, so it runs free ----------------------

def think(city: str, facts: list[str]) -> tuple[str, str]:
    known = " ".join(facts).lower()
    if "check_weather" not in known:
        return ("Is it even nice enough to bike there tomorrow?",
                f"check_weather('{city}')")
    if "check_bike" not in known:
        return ("I know the weather. Is the bike itself ready?", "check_bike()")
    if "flat tire" in known and "check_bus" not in known:
        return ("The bike is out. Is there a backup?", "check_bus('market')")
    return ("I know the weather, the bike, and the backup. Enough.", "finish")


# --- Option 2: a real model drives the same loop --------------------------

def react_with_model(city: str) -> str:
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import tool

    lc_tools = []
    for name, fn in TOOLS.items():
        lc_tools.append(tool(fn))  # Docstrings double as tool descriptions.
    model = init_chat_model(os.environ["MODEL"]).bind_tools(lc_tools)

    messages = [
        SystemMessage(
            "You are a day-planning assistant. Call one tool at a time, "
            "read the result, then decide the next step. Answer in two "
            "short sentences once you know enough."
        ),
        HumanMessage(f"Can I bike to the farmers market tomorrow morning "
                     f"in {city}?"),
    ]
    for turn in range(1, MAX_TURNS + 1):
        response = model.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return response.content
        for call in response.tool_calls[:1]:
            print(f"Action {turn}:      {call['name']}({call['args']})")
            try:
                result = TOOLS[call["name"]](*call["args"].values())
            except Exception as exc:
                result = f"This tool failed: {type(exc).__name__}"
            print(f"Observation {turn}: {result}\n")
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    return "I ran out of turns. Decide yourself."


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "Berlin"
    print(f"You: Can I bike to the farmers market tomorrow morning in "
          f"{city}?\n")

    if os.environ.get("MODEL"):
        print(f"Answer: {react_with_model(city)}")
        return

    facts = []
    for turn in range(1, MAX_TURNS + 1):
        thought, action = think(city, facts)
        print(f"Thought {turn}:     {thought}")
        if action == "finish":
            break
        print(f"Action {turn}:      {action}")
        name, args = action.split("(", 1)
        args = args.strip(")").strip("'\"")
        try:
            observation = TOOLS[name](args) if args else TOOLS[name]()
        except Exception as exc:
            observation = f"This tool failed: {type(exc).__name__}"
        facts.append(f"{name}: {observation}")
        print(f"Observation {turn}: {observation}\n")

    weather = facts[0].split(": ", 1)[1]
    print("Answer: Check the real forecast above, but plan on the bus - "
          "your bike has a flat tire.\n"
          f"(Weather: {weather})")


if __name__ == "__main__":
    main()
