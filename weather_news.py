
import asyncio
import os

from dotenv import load_dotenv
from ddgs import DDGS

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


# --------------------------------------------------
# 1. Load environment variables from the .env file
# --------------------------------------------------
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY is missing. "
        "Please check your .env file."
    )


# --------------------------------------------------
# 2. Define the DuckDuckGo weather news search tool
# --------------------------------------------------
def search_weather_news(city: str) -> str:
    """
    Search DuckDuckGo news for today's weather-related
    news in the specified city and return up to five results.
    """
    try:
        results = DDGS().news(
            query=f"{city} weather today",
            max_results=5
        )

        if not results:
            return f"No weather news results found for {city}."

        return str(results)

    except Exception as error:
        return f"Weather news search failed: {error}"


# --------------------------------------------------
# 3. Create the OpenAI model client
# --------------------------------------------------
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini"
)


# --------------------------------------------------
# 4. Create the AutoGen AI agent
# --------------------------------------------------
agent = AssistantAgent(
    name="weather_news",
    model_client=model_client,
    tools=[search_weather_news],
    system_message=(
        "You are a weather news assistant. "
        "Use the search_weather_news tool to find "
        "today's weather-related news for the requested city. "
        "Base your answer on the search results. "
        "Do not invent weather information. "
        "Reply in exactly three short lines. "
        "Each line must contain one concise point. "
        "Do not add headings, introductions, or conclusions."
    ),
    reflect_on_tool_use=True
)


# --------------------------------------------------
# 5. Ask the user for a city
# --------------------------------------------------
city = input("Which city? ").strip()


# --------------------------------------------------
# 6. Run the agent and close the model client
# --------------------------------------------------
async def main():
    try:
        if not city:
            print("Please enter a valid city name.")
            return

        task = (
            f"Find today's weather-related news for {city}. "
            "Use the search tool and provide exactly "
            "three short lines based on the search results."
        )

        await Console(
            agent.run_stream(task=task)
        )

    finally:
        await model_client.close()


# --------------------------------------------------
# 7. Start the application
# --------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())

