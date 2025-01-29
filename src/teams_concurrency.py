import asyncio
from asyncio import Semaphore
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main(semaphore):
    async with semaphore:
        client = OpenAIChatCompletionClient(model="gpt-4o", temperature=0.7)
        agent = AssistantAgent(name="assistant", model_client=client, system_message="Generate short beautiful poems")
        critic = AssistantAgent(name="critic", model_client=client, system_message="Review the poem")
        team = RoundRobinGroupChat([agent, critic], max_turns=10)

        async for msg in team.run_stream(task="Generate poem."):
            print(str(msg)[:80])

async def run_multiple_times(n_tasks, pool_size=2):
    semaphore = Semaphore(pool_size)
    tasks = [main(semaphore) for _ in range(n_tasks)]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(run_multiple_times(n_tasks=5))