from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


class PipelineError(RuntimeError):
    pass


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


class Runner:
    def __init__(self, work: Path, dry_run: bool = False):
        self.work = work
        self.dry_run = dry_run

    def run(self, command, *, cwd=None, env=None):
        command = [str(x) for x in command]
        print("+ " + shlex.join(command), flush=True)
        if self.dry_run:
            return
        logs = self.work / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        record = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "argv": command,
            "cwd": str(cwd) if cwd else None,
        }
        with (logs / "commands.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        merged = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        if env:
            merged.update(env)
        with (logs / "console.log").open("a") as handle:
            handle.write("\n+ " + shlex.join(command) + "\n")
            handle.flush()
            process = subprocess.Popen(
                command, cwd=cwd, env=merged, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    handle.write(line)
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
            if code:
                raise PipelineError(f"Command failed ({code}); see {logs / 'console.log'}")


def checked_output(command, *, include_stderr=False):
    """Return stdout, optionally including stderr for tools that print help there."""
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise PipelineError(f"Command failed: {shlex.join(map(str, command))}\n{result.stderr[-2000:]}")
    return result.stdout + result.stderr if include_stderr else result.stdout


def stage(runner, name, signature, outputs, action):
    """Skip complete stages only when their inputs/config match; never reuse stale outputs."""
    if runner.dry_run:
        print(f"\n[{name}] planned")
        action()
        return
    stamp = runner.work / "state" / f"{name}.json"
    fingerprint = digest(signature)
    if stamp.exists():
        previous = read_json(stamp)
        if previous.get("fingerprint") != fingerprint:
            raise PipelineError(
                f"{name} inputs/settings changed. Use a NEW --workspace to keep results consistent."
            )
        if all(path.exists() for path in outputs):
            print(f"[{name}] already complete")
            return
        raise PipelineError(f"{name} outputs are missing. Use a new workspace; do not trust its stamp.")
    print(f"\n[{name}] running", flush=True)
    action()
    missing = [str(path) for path in outputs if not path.exists()]
    if missing:
        raise PipelineError(f"{name} completed without required outputs: {missing}")
    write_json(stamp, {"fingerprint": fingerprint, "signature": signature})
