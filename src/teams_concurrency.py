import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main():
    
    client = OpenAIChatCompletionClient(model="gpt-4o", temperature=0.7)
    agent = AssistantAgent(name="assistant", model_client=client)
    team = RoundRobinGroupChat([agent], max_turns=10)

    async for msg in team.run_stream(task="Create a plan to find latest news about AutoGen."):
        print(str(msg)[:80])


if __name__ == "__main__":
    asyncio.run(main())