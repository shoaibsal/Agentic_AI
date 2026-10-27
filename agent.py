#!/usr/bin/env python3
"""Minimal agentic loop: plan -> act (tools) -> observe -> repeat.

    pip install anthropic          # Python 3.9+
    export ANTHROPIC_API_KEY=sk-ant-...
    python agent.py "Make 'make test' pass" --workdir ./my-repo

Guardrails: asks before any command or file write, cannot touch files
outside --workdir, stops after --max-steps. Use --yes only in a sandbox
(e.g. a throwaway Docker container).
"""
import argparse
import os
import subprocess
from pathlib import Path

import anthropic

MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")
MAX_CHARS = 10_000  # max tool output sent back to the model per call

SYSTEM = (
    "You are a build/CI agent working inside a repository. Investigate before "
    "changing anything, make the smallest fix that works, and verify it by "
    "re-running the relevant command. When the goal is met, or you are blocked, "
    "stop calling tools and give a short summary."
)

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


def main() -> None:
    p = argparse.ArgumentParser(description="Minimal agent loop")
    p.add_argument("goal", help="what the agent should achieve")
    p.add_argument("--workdir", default=".")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--yes", action="store_true", help="auto-approve every action (sandbox only)")
    args = p.parse_args()

    root = Path(args.workdir).resolve()
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    messages = [{"role": "user", "content": args.goal}]

    for step in range(1, args.max_steps + 1):
        # PLAN: the model reads the history so far and decides the next action
        resp = client.messages.create(model=MODEL, max_tokens=4096, system=SYSTEM,
                                      tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        for block in resp.content:
            if block.type == "text" and block.text.strip():
                print(f"\n[agent] {block.text}")

        if resp.stop_reason != "tool_use":  # no tool requested: goal met or blocked
            return

        # ACT + OBSERVE: run each requested tool and send the results back
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                print(f"\n[step {step}] {block.name}: {block.input}")
                output, is_error = run_tool(block.name, block.input, root, args.yes)
                print("  -> " + output[:200].replace("\n", "\n     "))
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": output, "is_error": is_error})
        messages.append({"role": "user", "content": results})  # ...and REPEAT

    print(f"\n[agent] Stopped after {args.max_steps} steps without finishing.")


if __name__ == "__main__":
    main()
