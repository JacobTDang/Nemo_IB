"""Nemo_IB Financial Analysis Agent. Ticker detected automatically from query."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.MCP_manager import MCPConnectionManager
from agent.workflows.analysis_workflow import WorkFlow


async def _run(user_query: str) -> str:
    async with MCPConnectionManager() as mcp:
        w = WorkFlow(mcp=mcp)
        return await w.run(user_query=user_query, ticker='')


def main():
    print("Nemo_IB Financial Analysis Agent")
    print("Ticker is detected automatically from your query.")
    user_query = input("\nYou: ").strip()
    if not user_query or user_query.lower() == 'quit':
        return
    print("\nStarting analysis...", flush=True)
    report = asyncio.run(_run(user_query))
    print("\n" + "=" * 80)
    print("FINAL ANALYSIS")
    print("=" * 80)
    print(report or "No analysis generated.")


if __name__ == "__main__":
    main()
