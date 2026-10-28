#!/usr/bin/env python3
"""Same agent as agent.py, with GitHub Copilot as the brain.

Difference: the Copilot SDK runs the plan -> act -> observe loop itself.
You only supply the tools and the guardrails, which are identical to agent.py.

    pip install github-copilot-sdk     # Python 3.11+, needs a Copilot seat
    # sign in once with the Copilot CLI, or: export COPILOT_GITHUB_TOKEN=...
    python3 agent_copilot.py "Make 'make test' pass" --workdir ./my-repo

Guardrails: Copilot can only use the three tools below (none of its built-in
shell/edit tools), asks before any command or file write, cannot touch files
outside --workdir, stops after --max-steps. Use --yes only in a sandbox.
"""
import argparse
import asyncio
import os
import subprocess
from pathlib import Path

from copilot import CopilotClient, Tool, ToolInvocation, ToolResult
from copilot.rpc import PermissionDecisionReject
from copilot.session_events import AssistantMessageData

MODEL = os.environ.get("AGENT_MODEL")  # None = your Copilot default model
MAX_CHARS = 10_000  # max tool output sent back to the model per call

SYSTEM = (
    "You are a build/CI agent working inside a repository. Investigate before "
    "changing anything, make the smallest fix that works, and verify it by "
    "re-running the relevant command. When the goal is met, or you are blocked, "
    "stop calling tools and give a short summary."
)

# ---- Tools and guardrails: unchanged from agent.py --------------------------
TOOLS = [
    {
        "name": "run_command",
        "description": "Run a shell command in the repository root. Returns exit code, stdout and stderr.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a text file. Path is relative to the repository root.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a text file with its full new content.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
]


def clip(text: str) -> str:
    """Keep the head and tail of long output; build errors are usually at the end."""
    if len(text) <= MAX_CHARS:
        return text
    head = MAX_CHARS // 5
    return text[:head] + "\n...[truncated]...\n" + text[-(MAX_CHARS - head):]


def approved(action: str, auto: bool) -> bool:
    """Human-in-the-loop check for anything that can change state."""
    return auto or input(f"  Allow {action}? [y/N] ").strip().lower() == "y"


def in_workspace(root: Path, rel: str) -> Path:
    """Resolve a path and refuse anything that escapes the workspace."""
    path = (root / rel).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{rel} is outside the workspace")
    return path


def run_tool(name: str, inp: dict, root: Path, auto: bool) -> tuple[str, bool]:
    """Execute one tool call. Returns (output, is_error)."""
    try:
        if name == "run_command":
            cmd = inp["command"]
            if not approved(f"command `{cmd}`", auto):
                return "The user denied this command. Choose another approach or stop.", True
            r = subprocess.run(cmd, shell=True, cwd=root, capture_output=True,
                               text=True, timeout=300)
            return clip(f"exit code: {r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"), False
        if name == "read_file":
            return clip(in_workspace(root, inp["path"]).read_text()), False
        if name == "write_file":
            path = in_workspace(root, inp["path"])
            if not approved(f"write to {inp['path']}", auto):
                return "The user denied this write.", True
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(inp["content"])
            return f"Wrote {len(inp['content'])} characters to {inp['path']}", False
        return f"Unknown tool: {name}", True
    except Exception as e:  # failures go back to the model as observations
        return f"{type(e).__name__}: {e}", True


# ---- Copilot glue: replaces the hand-written loop ---------------------------
def make_handler(root: Path, auto: bool, max_steps: int):
    """Copilot calls this for every tool call it decides to make."""
    lock = asyncio.Lock()  # one tool at a time, so approval prompts never overlap
    steps = 0

    async def handler(inv: ToolInvocation) -> ToolResult:
        nonlocal steps
        async with lock:
            steps += 1
            if steps > max_steps:
                return ToolResult(text_result_for_llm=f"Step limit ({max_steps}) reached. "
                                  "Stop calling tools and summarize.", result_type="denied")
            print(f"\n[step {steps}] {inv.tool_name}: {inv.arguments}")
            # run_tool blocks (input(), subprocess), so keep it off the event loop
            output, is_error = await asyncio.to_thread(
                run_tool, inv.tool_name, inv.arguments or {}, root, auto)
            print("  -> " + output[:200].replace("\n", "\n     "))
            return ToolResult(text_result_for_llm=output,
                              result_type="failure" if is_error else "success")

    return handler


async def main() -> None:
    p = argparse.ArgumentParser(description="Minimal agent on GitHub Copilot")
    p.add_argument("goal", help="what the agent should achieve")
    p.add_argument("--workdir", default=".")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--timeout", type=float, default=1800, help="seconds before giving up")
    p.add_argument("--yes", action="store_true", help="auto-approve every action (sandbox only)")
    args = p.parse_args()

    root = Path(args.workdir).resolve()
    handler = make_handler(root, args.yes, args.max_steps)
    tools = [Tool(name=t["name"], description=t["description"], parameters=t["input_schema"],
                  handler=handler, skip_permission=True)  # run_tool does its own approvals
             for t in TOOLS]

    # No token set = use the account you signed in with in the Copilot CLI
    async with CopilotClient(github_token=os.environ.get("COPILOT_GITHUB_TOKEN")) as client:
        async with await client.create_session(
            model=MODEL,
            tools=tools,
            available_tools=[t.name for t in tools],  # ONLY our tools: no built-in shell/edit
            system_message={"mode": "append", "content": SYSTEM},
            working_directory=str(root),
            on_permission_request=lambda req, inv: PermissionDecisionReject(
                feedback="Only the provided tools are allowed."),  # refuse anything else
        ) as session:

            def on_event(event):
                if event.agent_id:  # ignore sub-agents
                    return
                if isinstance(event.data, AssistantMessageData) and event.data.content.strip():
                    print(f"\n[agent] {event.data.content}")

            session.on(on_event)
            try:
                # The whole plan -> act -> observe loop runs inside this one call
                await session.send_and_wait(args.goal, timeout=args.timeout)
            except Exception as e:
                print(f"\n[agent] Stopped: {e}")


if __name__ == "__main__":
    asyncio.run(main())
