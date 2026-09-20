#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  FORGE v4.0 — Autonomous + Streaming Agent                               ║
║                                                                          ║
║  Hermes Agent + Claude Code arxitekturası əsasında:                      ║
║    • Streaming (token-by-token, interruptible)                           ║
║    • Autonomous Loop (Plan → Execute → Reflect)                          ║
║    • Memory (MEMORY.md) + Skills (auto-create)                           ║
║    • Parallel Tools (ThreadPoolExecutor)                                 ║
║    • Session Persistence (JSONL)                                         ║
║    • Context Compaction                                                  ║
║    • Cron Scheduler                                                      ║
║    • No Model Rotation                                                   ║
║                                                                          ║
║  İstifadə:                                                               ║
║    python forge.py                       # interactive                    ║
║    python forge.py --autonomous "task"   # autonomous                     ║
║    python forge.py --stream "prompt"     # streaming                      ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import logging.handlers
import os
import random
import re
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import tool as lc_tool
from langchain_openai import ChatOpenAI

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table

# ══════════════════════════════════════════════════════════════════════════
# 1) CONFIG
# ══════════════════════════════════════════════════════════════════════════
load_dotenv()
console = Console()

API_KEY = os.getenv("NVIDIA_API_KEY")
BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")

if not API_KEY:
    console.print(Panel("[red]❌ NVIDIA_API_KEY yoxdur[/red]", border_style="red"))
    sys.exit(1)

# Dirs
FORGE_HOME = Path.home() / ".forge"
FORGE_HOME.mkdir(exist_ok=True)
for sub in ("logs", "sessions", "snapshots", "skills", "memory"):
    (FORGE_HOME / sub).mkdir(exist_ok=True)
HISTORY_FILE = FORGE_HOME / "history.txt"
MEMORY_FILE = FORGE_HOME / "memory" / "MEMORY.md"
SKILLS_DB = FORGE_HOME / "skills" / "skills.db"
SESSIONS_DB = FORGE_HOME / "sessions" / "sessions.db"

WORK_DIR = Path.cwd().resolve()

# Limits
MAX_CONTEXT_TOKENS = 100_000
AUTOCOMPACT_BUFFER = 10_000
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_BASH_OUTPUT = 30_000
LLM_TIMEOUT = 120
MAX_RETRIES = 4
DEFAULT_RPM = 20
MAX_WORKERS = 8
MAX_TURNS = 50
AUTONOMOUS_MAX_STEPS = 30

# ══════════════════════════════════════════════════════════════════════════
# 2) LOGGING
# ══════════════════════════════════════════════════════════════════════════
class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "ts": datetime.now().isoformat(),
            "lvl": record.levelname,
            "msg": record.getMessage(),
        }, ensure_ascii=False)

fh = logging.handlers.RotatingFileHandler(
    FORGE_HOME / "logs" / "forge.log",
    maxBytes=5_000_000, backupCount=3, encoding="utf-8")
fh.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[fh], force=True)
log = logging.getLogger("forge")

# ══════════════════════════════════════════════════════════════════════════
# 3) CANCEL TOKEN (interruptible streaming)
# ══════════════════════════════════════════════════════════════════════════
class CancelToken:
    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    def reset(self):
        self._event.clear()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def check(self):
        if self._event.is_set():
            raise KeyboardInterrupt("Cancelled")


CANCEL = CancelToken()


def install_signal_handler():
    def handler(sig, frame):
        if CANCEL.is_cancelled():
            console.print("\n[red]Force exit[/red]")
            sys.exit(1)
        console.print("\n[yellow]⏸  Interrupting (Ctrl+C again to force)...[/yellow]")
        CANCEL.cancel()
    try:
        signal.signal(signal.SIGINT, handler)
    except (ValueError, OSError):
        pass


# ══════════════════════════════════════════════════════════════════════════
# 4) CONTEXT VARS
# ══════════════════════════════════════════════════════════════════════════
_BASE_VAR: contextvars.ContextVar[Path] = contextvars.ContextVar(
    "base", default=WORK_DIR)


def get_base() -> Path:
    return _BASE_VAR.get()


def set_base(p: Path):
    _BASE_VAR.set(Path(p).resolve())


class BaseContext:
    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self.token = None

    def __enter__(self):
        self.token = _BASE_VAR.set(self.path)
        return self.path

    def __exit__(self, *args):
        if self.token is not None:
            _BASE_VAR.reset(self.token)


# ══════════════════════════════════════════════════════════════════════════
# 5) FILE SAFETY
# ══════════════════════════════════════════════════════════════════════════
def safe_path(path: str, allow_work: bool = True) -> Optional[Path]:
    if not path or not isinstance(path, str) or "\x00" in path:
        return None
    try:
        base = get_base()
        p = Path(path)
        if not p.is_absolute():
            p = base / p
        p = p.resolve()
        try:
            p.relative_to(base)
            return p
        except ValueError:
            pass
        if allow_work:
            try:
                p.relative_to(WORK_DIR)
                return p
            except ValueError:
                pass
        return None
    except Exception:
        return None


def atomic_write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.stem + "_", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


# ══════════════════════════════════════════════════════════════════════════
# 6) SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════
class Snapshot:
    @staticmethod
    def create(name: str, paths: List[str]) -> Optional[str]:
        try:
            snap_id = f"{int(time.time())}_{os.urandom(3).hex()}"
            snap_dir = FORGE_HOME / "snapshots" / snap_id
            snap_dir.mkdir(parents=True, exist_ok=True)
            base = get_base()
            for rel in paths:
                p = safe_path(rel)
                if not p or not p.is_file():
                    continue
                try:
                    rel_path = p.relative_to(base)
                    dest = snap_dir / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dest)
                except Exception:
                    pass
            snaps = sorted(
                [d for d in (FORGE_HOME / "snapshots").iterdir() if d.is_dir()],
                key=lambda d: d.name)
            for old in snaps[:-20]:
                shutil.rmtree(old, ignore_errors=True)
            return snap_id
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════════════
# 7) RATE LIMITER
# ══════════════════════════════════════════════════════════════════════════
class RateLimiter:
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.rate = rpm / 60.0
        self.tokens = 5.0
        self.last = time.time()
        self._lock = threading.Lock()
        self._errors = 0

    def wait(self):
        with self._lock:
            now = time.time()
            self.tokens = min(5.0, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens < 1:
                wait = (1 - self.tokens) / self.rate
                time.sleep(wait)
                self.tokens = 0
                self.last = time.time()
            else:
                self.tokens -= 1

    def report_429(self):
        with self._lock:
            self._errors += 1
            if self._errors >= 3:
                self.rpm = max(3, int(self.rpm * 0.7))
                self.rate = self.rpm / 60.0
                self._errors = 0

    def report_success(self):
        with self._lock:
            self._errors = 0


RATE = RateLimiter(DEFAULT_RPM)


# ══════════════════════════════════════════════════════════════════════════
# 8) LLM CLIENT (no rotation, streaming)
# ══════════════════════════════════════════════════════════════════════════
class LLMClient:
    def __init__(self):
        self._local = threading.local()

    def get(self) -> ChatOpenAI:
        if not hasattr(self._local, "llm"):
            self._local.llm = ChatOpenAI(
                openai_api_key=API_KEY,
                openai_api_base=BASE_URL,
                model=MODEL,
                temperature=0.2,
                max_tokens=8192,
                timeout=LLM_TIMEOUT,
                max_retries=0,
            )
        return self._local.llm

    def get_with_tools(self, tools: List) -> ChatOpenAI:
        if not hasattr(self._local, "llm_tools"):
            self._local.llm_tools = self.get().bind_tools(tools)
        return self._local.llm_tools


LLM_CLIENT = LLMClient()


# ══════════════════════════════════════════════════════════════════════════
# 9) STREAMING ENGINE (token-by-token + Rich Live)
# ══════════════════════════════════════════════════════════════════════════
class StreamingEngine:
    """
    Token-by-token streaming with Rich Live.
    Interruptible — ESC/Ctrl+C ilə dayandırıla bilər.
    Tool calls ayrıca göstərilir.
    """

    def __init__(self, show_reasoning: bool = False):
        self.show_reasoning = show_reasoning
        self.buffer = ""
        self.tool_calls: List[Dict] = []
        self.live: Optional[Live] = None

    def stream_llm(self, messages: List, tools: Optional[List] = None) -> AIMessageChunk:
        """
        Stream LLM response. Returns final accumulated chunk.
        Async generator deyil — sync loop üçün.
        """
        CANCEL.check()
        RATE.wait()

        llm = LLM_CLIENT.get_with_tools(tools) if tools else LLM_CLIENT.get()

        self.buffer = ""
        self.tool_calls = []
        accumulated: Optional[AIMessageChunk] = None

        # Rich Live — canlı markdown
        self.live = Live(
            console=console,
            refresh_per_second=15,
            vertical_overflow="visible",
        )

        with self.live:
            try:
                for chunk in llm.stream(messages):
                    CANCEL.check()

                    # Content delta
                    if chunk.content:
                        self.buffer += chunk.content
                        try:
                            self.live.update(Markdown(self.buffer))
                        except Exception:
                            # Markdown parse xətası → plain text
                            self.live.update(self.buffer)

                    # Tool call chunks
                    if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                        for tc in chunk.tool_call_chunks:
                            self._accumulate_tool_call(tc)

                    # Accumulate
                    if accumulated is None:
                        accumulated = chunk
                    else:
                        accumulated = accumulated + chunk

                    RATE.report_success()

            except KeyboardInterrupt:
                self.live.stop()
                console.print("\n[yellow]⏸  Stream interrupted[/yellow]")
                CANCEL.reset()

        return accumulated

    def _accumulate_tool_call(self, tc: Dict):
        """Tool call chunks birləşdir."""
        idx = tc.get("index", 0)
        while len(self.tool_calls) <= idx:
            self.tool_calls.append({"id": "", "name": "", "args": ""})

        if tc.get("id"):
            self.tool_calls[idx]["id"] = tc["id"]
        if tc.get("name"):
            self.tool_calls[idx]["name"] = tc["name"]
        if tc.get("args"):
            self.tool_calls[idx]["args"] += tc["args"]

    def get_final_content(self) -> str:
        return self.buffer

    def get_tool_calls(self) -> List[Dict]:
        """Parse edilmiş tool calls."""
        result = []
        for tc in self.tool_calls:
            if not tc["name"]:
                continue
            try:
                args = json.loads(tc["args"]) if tc["args"] else {}
            except json.JSONDecodeError:
                args = {}
            result.append({
                "id": tc["id"] or f"call_{random.randint(1000, 9999)}",
                "name": tc["name"],
                "args": args,
            })
        return result


STREAM = StreamingEngine()


# ══════════════════════════════════════════════════════════════════════════
# 10) CONTEXT MANAGER
# ══════════════════════════════════════════════════════════════════════════
class ContextManager:
    def __init__(self):
        self.messages: List = []

    def total_tokens(self) -> int:
        return sum(max(1, len(str(getattr(m, "content", "") or "")) // 4)
                   for m in self.messages)

    def add(self, msg):
        self.messages.append(msg)

    def extend(self, msgs):
        self.messages.extend(msgs)

    def should_compact(self) -> bool:
        return self.total_tokens() > (MAX_CONTEXT_TOKENS - AUTOCOMPACT_BUFFER)

    def compact(self) -> bool:
        try:
            system = [m for m in self.messages if isinstance(m, SystemMessage)]
            others = [m for m in self.messages if not isinstance(m, SystemMessage)]
            if len(others) <= 10:
                return True
            old, recent = others[:-10], others[-10:]
            summary = "[COMPACTED]\n" + "\n".join(
                f"- {str(m.content or '')[:100]}" for m in old[-30:])
            recent.insert(0, HumanMessage(content=summary))
            self.messages = system + recent
            return True
        except Exception:
            return False


# ══════════════════════════════════════════════════════════════════════════
# 11) TOOLS
# ══════════════════════════════════════════════════════════════════════════
def _is_binary(p: Path) -> bool:
    try:
        with p.open("rb") as f:
            return b"\x00" in f.read(1024)
    except Exception:
        return False


def h_read_file(path: str, offset: int = 1, limit: int = 2000) -> str:
    p = safe_path(path)
    if not p or not p.is_file():
        return f"❌ Fayl yoxdur: {path}"
    try:
        if p.stat().st_size > MAX_FILE_SIZE:
            return "❌ Fayl böyük"
        if _is_binary(p):
            return "❌ Binary"
        content = p.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        start = max(0, offset - 1)
        chunk = lines[start:start + limit]
        return "\n".join(f"{start + i + 1:6d}│ {line}"
                        for i, line in enumerate(chunk)) or "(boş)"
    except Exception as e:
        return f"❌ {e}"


def h_write_file(path: str, content: str) -> str:
    p = safe_path(path)
    if not p:
        return f"❌ Təhlükəsiz olmayan yol: {path}"
    try:
        if p.exists():
            Snapshot.create(f"pre_{p.name}", [str(p.relative_to(get_base()))])
        atomic_write(p, content)
        return f"✅ {p} ({len(content)}b)"
    except Exception as e:
        return f"❌ {e}"


def h_edit_file(path: str, old_string: str, new_string: str) -> str:
    p = safe_path(path)
    if not p or not p.is_file():
        return "❌ Fayl yoxdur"
    try:
        content = p.read_text(encoding="utf-8")
        count = content.count(old_string)
        if count == 0:
            return "❌ `old_string` tapılmadı"
        if count > 1:
            return f"❌ `old_string` {count} dəfə təkrar"
        Snapshot.create(f"pre_{p.name}", [str(p.relative_to(get_base()))])
        atomic_write(p, content.replace(old_string, new_string, 1))
        return f"✅ {p}"
    except Exception as e:
        return f"❌ {e}"


def h_list_files(directory: str = ".", pattern: str = "*") -> str:
    p = safe_path(directory)
    if not p or not p.is_dir():
        return f"❌ Qovluq yoxdur"
    try:
        files = sorted(p.glob(pattern))
        if not files:
            return f"❌ `{pattern}` uyğun yoxdur"
        return "\n".join(str(f.relative_to(p)) for f in files[:200])
    except Exception as e:
        return f"❌ {e}"


def h_tree_view(directory: str = ".", max_depth: int = 3) -> str:
    p = safe_path(directory)
    if not p or not p.is_dir():
        return "❌ Qovluq yoxdur"
    SKIP = {".git", "__pycache__", "node_modules", ".venv", "venv",
            "dist", "build", ".turbo"}
    lines = [str(p) + "/"]

    def walk(d: Path, prefix: str = "", depth: int = 0):
        if depth >= max_depth:
            return
        try:
            items = sorted(
                [x for x in d.iterdir()
                 if x.name not in SKIP and not x.name.startswith(".")],
                key=lambda x: (x.is_file(), x.name))[:40]
        except PermissionError:
            return
        for i, it in enumerate(items):
            last = i == len(items) - 1
            marker = "└── " if last else "├── "
            lines.append(f"{prefix}{marker}{it.name}" +
                        ("/" if it.is_dir() else ""))
            if it.is_dir():
                walk(it, prefix + ("    " if last else "│   "), depth + 1)

    walk(p)
    return "```\n" + "\n".join(lines) + "\n```"


def h_verify_file(path: str) -> str:
    p = safe_path(path)
    if not p or not p.is_file():
        return "❌ Fayl yoxdur"
    ext = p.suffix.lower()
    try:
        content = p.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            return "❌ Boş"
        if ext == ".json":
            json.loads(content)
            return "✅ JSON"
        if ext == ".py":
            compile(content, str(p), "exec")
            return "✅ Python"
        if ext in (".ts", ".tsx", ".js", ".jsx"):
            if content.count("{") != content.count("}"):
                return "❌ Braces"
            return f"✅ {ext[1:].upper()}"
        return f"✅ {ext or 'txt'}"
    except SyntaxError as e:
        return f"❌ Syntax: {e.msg} (sətir {e.lineno})"
    except Exception as e:
        return f"❌ {str(e)[:100]}"


def h_grep_search(pattern: str, path: str = ".", max_results: int = 50) -> str:
    p = safe_path(path)
    if not p:
        return "❌"
    try:
        if shutil.which("rg"):
            cmd = ["rg", "-n", "--no-heading", pattern, str(p)]
        else:
            cmd = ["grep", "-rn", "-E", pattern, str(p)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        out = r.stdout or ""
        return "\n".join(out.splitlines()[:max_results]) or "🔍 yoxdur"
    except Exception as e:
        return f"❌ {e}"


def h_run_bash(command: str, timeout: int = 60) -> str:
    CANCEL.check()
    if not command:
        return "❌ Boş"
    FORBIDDEN = {";", "&&", "||", "|", ">", "<", "`", "$(", "${",
                 "&", "\\", "\n", "\r"}
    for sym in FORBIDDEN:
        if sym in command:
            return f"❌ Qadağan: {sym}"
    try:
        parts = shlex.split(command)
    except ValueError as e:
        return f"❌ Parse: {e}"
    if not parts:
        return "❌ Boş"
    SAFE = {"ls", "pwd", "cat", "head", "tail", "wc", "grep", "find",
            "file", "stat", "tree", "sort", "uniq", "tr", "cut", "awk",
            "sed", "jq", "du", "df", "which", "type", "mkdir", "touch",
            "echo", "date", "whoami", "uname", "ps", "env", "git",
            "python3", "python", "pip", "pip3", "npm", "node",
            "pnpm", "yarn", "make", "pytest", "ruff", "mypy"}
    if parts[0] not in SAFE:
        return f"❌ `{parts[0]}` icazəli deyil"
    DANGEROUS = {"-rf", "-fr", "--no-preserve-root"}
    for arg in parts[1:]:
        if arg in DANGEROUS:
            return f"❌ Təhlükəli: {arg}"
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
           "HOME": os.environ.get("HOME", "/tmp"),
           "LANG": "en_US.UTF-8"}
    try:
        r = subprocess.run(parts, capture_output=True, text=True,
                          timeout=timeout, shell=False,
                          cwd=str(get_base()), env=env)
        out = (r.stdout or "")[:6000]
        if r.stderr:
            out += "\n[stderr]\n" + (r.stderr or "")[:2000]
        return f"```\n{out}\n```\n_(rc={r.returncode})_"
    except subprocess.TimeoutExpired:
        return "⏱ Timeout"
    except Exception as e:
        return f"❌ {e}"


def h_delete_file(path: str) -> str:
    p = safe_path(path)
    if not p or not p.is_file():
        return "❌ Fayl yoxdur"
    try:
        Snapshot.create(f"pre_del_{p.name}", [str(p.relative_to(get_base()))])
        p.unlink()
        return f"✅ Silindi: {path}"
    except Exception as e:
        return f"❌ {e}"


def h_run_tests(path: str = ".") -> str:
    p = safe_path(path)
    if not p:
        return "❌"
    if shutil.which("pytest") and ((p / "tests").exists() or list(p.glob("test_*.py"))):
        try:
            r = subprocess.run(["pytest", "-q", "--tb=short"],
                             capture_output=True, text=True,
                             timeout=120, cwd=str(p))
            return f"```\n{(r.stdout or '')[:3000]}\n```"
        except Exception as e:
            return f"❌ {e}"
    return "ℹ️ Test yoxdur"


def h_web_search(query: str, max_results: int = 5) -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return "\n\n".join(
            f"**{i}. {r['title']}**\n{r['href']}\n{r['body'][:200]}..."
            for i, r in enumerate(results, 1)) or "🔍 yoxdur"
    except ImportError:
        return "❌ duckduckgo-search yoxdur"
    except Exception as e:
        return f"❌ {e}"


def h_memory_save(key: str, value: str) -> str:
    """MEMORY.md-ə yaz."""
    try:
        with MEMORY_FILE.open("a", encoding="utf-8") as f:
            f.write(f"\n## {key}\n_{datetime.now().isoformat()}_\n\n{value}\n")
        return f"✅ Yadda saxlanıldı: {key}"
    except Exception as e:
        return f"❌ {e}"


def h_memory_read() -> str:
    """MEMORY.md oxu."""
    if not MEMORY_FILE.exists():
        return "(boş)"
    try:
        return MEMORY_FILE.read_text(encoding="utf-8")[:5000]
    except Exception as e:
        return f"❌ {e}"


def h_skill_save(name: str, description: str, code: str) -> str:
    """Skill yarat (SQLite)."""
    try:
        with sqlite3.connect(SKILLS_DB) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE, description TEXT, code TEXT,
                uses INTEGER DEFAULT 0, created TEXT)""")
            c.execute("""INSERT OR REPLACE INTO skills
                (name, description, code, created) VALUES (?,?,?,?)""",
                (name, description, code, datetime.now().isoformat()))
        return f"✅ Skill: {name}"
    except Exception as e:
        return f"❌ {e}"


def h_skill_list() -> str:
    """Skill siyahısı."""
    try:
        with sqlite3.connect(SKILLS_DB) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE, description TEXT, code TEXT,
                uses INTEGER DEFAULT 0, created TEXT)""")
            rows = c.execute("SELECT name, description, uses FROM skills "
                           "ORDER BY uses DESC LIMIT 20").fetchall()
        if not rows:
            return "(skill yoxdur)"
        return "\n".join(f"• **{n}** ({u}x) — {d[:60]}" for n, d, u in rows)
    except Exception as e:
        return f"❌ {e}"


def h_skill_use(name: str) -> str:
    """Skill-i yüklə və istifadə sayını artır."""
    try:
        with sqlite3.connect(SKILLS_DB) as c:
            c.execute("UPDATE skills SET uses = uses + 1 WHERE name = ?", (name,))
            row = c.execute("SELECT code FROM skills WHERE name = ?",
                          (name,)).fetchone()
        return row[0] if row else f"❌ Skill tapılmadı: {name}"
    except Exception as e:
        return f"❌ {e}"


# ══════════════════════════════════════════════════════════════════════════
# 12) LANGCHAIN TOOLS
# ══════════════════════════════════════════════════════════════════════════
@lc_tool
def read_file(path: str, offset: int = 1, limit: int = 2000) -> str:
    """Faylı oxuyur. Sətir nömrələri ilə."""
    return h_read_file(path, offset, limit)


@lc_tool
def write_file(path: str, content: str) -> str:
    """Fayl yaradır/əvəz edir. Qovluqlar avtomatik yaranır."""
    return h_write_file(path, content)


@lc_tool
def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Faylda unikal mətn əvəzləməsi."""
    return h_edit_file(path, old_string, new_string)


@lc_tool
def list_files(directory: str = ".", pattern: str = "*") -> str:
    """Qovluqdakı faylları siyahıla."""
    return h_list_files(directory, pattern)


@lc_tool
def tree_view(directory: str = ".", max_depth: int = 3) -> str:
    """Qovluq strukturunu ağac kimi göstər."""
    return h_tree_view(directory, max_depth)


@lc_tool
def verify_file(path: str) -> str:
    """Faylın sintaksisini yoxla."""
    return h_verify_file(path)


@lc_tool
def grep_search(pattern: str, path: str = ".") -> str:
    """Regex axtarışı."""
    return h_grep_search(pattern, path)


@lc_tool
def run_bash(command: str, timeout: int = 60) -> str:
    """Təhlükəsiz shell əmri (whitelist)."""
    return h_run_bash(command, timeout)


@lc_tool
def delete_file(path: str) -> str:
    """Faylı silir (snapshot ilə)."""
    return h_delete_file(path)


@lc_tool
def run_tests(path: str = ".") -> str:
    """Testləri işə sal (pytest)."""
    return h_run_tests(path)


@lc_tool
def web_search(query: str, max_results: int = 5) -> str:
    """Web axtarışı (DuckDuckGo)."""
    return h_web_search(query, max_results)


@lc_tool
def memory_save(key: str, value: str) -> str:
    """Persistent yaddaşa yaz (MEMORY.md)."""
    return h_memory_save(key, value)


@lc_tool
def memory_read() -> str:
    """Yaddaşı oxu."""
    return h_memory_read()


@lc_tool
def skill_save(name: str, description: str, code: str) -> str:
    """Yeni skill yarat (təcrübədən öyrənmə)."""
    return h_skill_save(name, description, code)


@lc_tool
def skill_list() -> str:
    """Mövcud skill-ləri siyahıla."""
    return h_skill_list()


@lc_tool
def skill_use(name: str) -> str:
    """Skill-i yüklə."""
    return h_skill_use(name)


TOOLS = [
    read_file, write_file, edit_file, list_files, tree_view,
    verify_file, grep_search, run_bash, delete_file, run_tests,
    web_search, memory_save, memory_read,
    skill_save, skill_list, skill_use,
]

TOOL_HANDLERS = {
    t.name: globals()[f"h_{t.name}"] for t in TOOLS
    if f"h_{t.name}" in globals()
}


# ══════════════════════════════════════════════════════════════════════════
# 13) PERMISSION
# ══════════════════════════════════════════════════════════════════════════
class PermissionMode(str, Enum):
    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    PLAN = "plan"
    AUTO = "auto"
    BYPASS = "bypass"


READONLY = {"read_file", "list_files", "tree_view", "verify_file",
            "grep_search", "web_search", "memory_read", "skill_list"}
WRITE = {"write_file", "edit_file", "delete_file", "memory_save", "skill_save"}


class Permission:
    def __init__(self, mode: PermissionMode = PermissionMode.DEFAULT):
        self.mode = mode

    def check(self, tool_name: str, args: dict) -> Tuple[Optional[bool], str]:
        if self.mode == PermissionMode.BYPASS:
            return True, "bypass"
        if self.mode == PermissionMode.AUTO:
            return True, "auto"
        if self.mode == PermissionMode.PLAN:
            if tool_name in READONLY:
                return True, "plan-readonly"
            return False, "plan mode: write denied"
        if self.mode == PermissionMode.ACCEPT_EDITS:
            if tool_name in READONLY or tool_name in WRITE:
                return True, "acceptEdits"
            return False, "acceptEdits: shell denied"
        return None, "ask"


# ══════════════════════════════════════════════════════════════════════════
# 14) SESSION (JSONL)
# ══════════════════════════════════════════════════════════════════════════
class Session:
    def __init__(self, session_id: str, project_dir: Path):
        self.session_id = session_id
        self.project_dir = project_dir
        safe = hashlib.md5(str(project_dir).encode()).hexdigest()[:16]
        self.path = FORGE_HOME / "sessions" / f"{safe}.jsonl"
        self._lock = threading.Lock()

    def append(self, event: dict):
        with self._lock:
            event["ts"] = datetime.now().isoformat()
            try:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════
# 15) PLANNER
# ══════════════════════════════════════════════════════════════════════════
PLANNER_PROMPT = """Sən layihə planlayıcısı-san. Verilən məqsəd üçün ATOMİK addımlar planı yarat.

MƏQSƏD: {goal}

CAVAB FORMATI (yalnız JSON):
{{
  "understanding": "məqsədin 1 cümləlik izahı",
  "steps": [
    {{"n": 1, "action": "konkret addım", "tools": ["tool1"], "expected": "nəticə"}},
    ...
  ],
  "risks": ["risk 1", "risk 2"]
}}

QAYDALAR:
1. Hər addım KONKRET və İCRA EDİLƏ BİLƏN
2. 2-15 addım arası
3. Tool-ları addımlara uyğun seç
4. Addımlar ardıcıl ola bilər
"""


def plan_task(goal: str) -> dict:
    """Məqsədi addımlara böl."""
    console.print()
    console.rule("[cyan]📋 Planlama[/cyan]")

    safe_goal = goal[:2000]
    prompt = PLANNER_PROMPT.format(goal=safe_goal)

    try:
        r = LLM_CLIENT.get().invoke([
            SystemMessage(content="Yalnız JSON qaytar."),
            HumanMessage(content=prompt),
        ])
        text = r.content or ""
    except Exception as e:
        return {"understanding": goal, "steps": [], "risks": [str(e)]}

    # Parse JSON
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"understanding": goal, "steps": [], "risks": ["JSON parse xətası"]}

    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        # Robust parse
        fixed = re.sub(r",(\s*[}\]])", r"\1", m.group(0))
        try:
            data = json.loads(fixed)
        except Exception:
            return {"understanding": goal, "steps": [], "risks": ["JSON parse xətası"]}

    steps = data.get("steps", [])
    console.print(f"[green]✅ {len(steps)} addım planlandı[/green]")

    # Planı göstər
    if steps:
        t = Table(title="📋 Plan", box=box.ROUNDED, header_style="bold")
        t.add_column("#", style="cyan", justify="right")
        t.add_column("Addım", style="white")
        t.add_column("Tool-lar", style="dim")
        for s in steps[:20]:
            t.add_row(str(s.get("n", "?")),
                     str(s.get("action", ""))[:60],
                     ", ".join(s.get("tools", []))[:30])
        console.print(t)

    return data


# ══════════════════════════════════════════════════════════════════════════
# 16) EXECUTOR (parallel tools)
# ══════════════════════════════════════════════════════════════════════════
class Executor:
    def __init__(self, permission: Permission, session: Session):
        self.permission = permission
        self.session = session

    def execute_tool(self, name: str, args: dict) -> str:
        """Bir tool icra et."""
        # Permission
        allowed, reason = self.permission.check(name, args)
        if allowed is False:
            return f"🚫 {reason}"
        if allowed is None:
            console.print(f"\n[yellow]🔧 {name}[/yellow]")
            for k, v in args.items():
                console.print(f"   [dim]{k}: {str(v)[:80]}[/dim]")
            if not Confirm.ask("[yellow]İcazə?[/yellow]", default=True):
                return "❌ İstifadəçi imtina etdi"

        handler = TOOL_HANDLERS.get(name)
        if not handler:
            return f"❌ Tool yoxdur: {name}"

        try:
            result = handler(**args)
        except TypeError as e:
            return f"❌ Argument: {e}"
        except Exception as e:
            return f"❌ Tool: {e}"

        s = str(result)
        return s[:MAX_BASH_OUTPUT] if len(s) > MAX_BASH_OUTPUT else s

    def execute_parallel(self, tool_calls: List[Dict]) -> List[ToolMessage]:
        """Tool-ları paralel icra et (Claude Code üsulu)."""
        if not tool_calls:
            return []

        results: List[Optional[ToolMessage]] = [None] * len(tool_calls)

        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tool_calls))) as ex:
            futures = {}
            for i, tc in enumerate(tool_calls):
                fut = ex.submit(self.execute_tool, tc["name"], tc["args"])
                futures[fut] = i

            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    result = f"❌ {e}"
                results[i] = ToolMessage(
                    content=str(result),
                    tool_call_id=tool_calls[i]["id"])

        return [r for r in results if r is not None]


# ══════════════════════════════════════════════════════════════════════════
# 17) REFLECTOR
# ══════════════════════════════════════════════════════════════════════════
REFLECT_PROMPT = """Sən tənqidçi-agentsən. Verilən nəticəni qiymətləndir.

MƏQSƏD: {goal}
NƏTİCƏ: {result}

Qiymətləndir və YALNIZ JSON qaytar:
{{
  "score": 0-10,
  "complete": true|false,
  "issues": ["problem 1"],
  "suggestion": "nə etməli (əgər yenidən cəhd lazımdırsa)"
}}
"""


def reflect(goal: str, result: str) -> dict:
    """Nəticəni tənqid et."""
    try:
        r = LLM_CLIENT.get().invoke([
            SystemMessage(content="Yalnız JSON."),
            HumanMessage(content=REFLECT_PROMPT.format(
                goal=goal[:500], result=result[:2000])),
        ])
        text = r.content or ""
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return {"score": 5, "complete": True, "issues": [], "suggestion": ""}


# ══════════════════════════════════════════════════════════════════════════
# 18) SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════
def build_system_prompt() -> str:
    tools_list = "\n".join(
        f"• `{t.name}`: {(t.description or '').split(chr(10))[0]}"
        for t in TOOLS)

    memory_context = ""
    if MEMORY_FILE.exists():
        try:
            mem = MEMORY_FILE.read_text(encoding="utf-8")[:2000]
            if mem.strip():
                memory_context = f"\n\nPERSISTENT MEMORY:\n{mem}\n"
        except Exception:
            pass

    return f"""Sən Forge adlı autonomous kodlaşdırma agentsən.

PRİNSİPLƏR:
• Azərbaycan dilində cavab ver
• Markdown istifadə et
• Alətlərdən istifadə et — təxmin etmə
• Yalnız təsdiqlənmiş məlumat ver
• Kod bloklarını ``` ilə göstər
• Uzun tapşırıqları addımlara böl

⚠️  VACİB:
• Qovluq yaratmaq üçün `mkdir` İSTİFADƏ ETMƏ
• `write_file('qovluq/fayl.py', '...')` → qovluqlar AVTOMATİK yaranır

MÖVCUD ALƏTLƏR:
{tools_list}

İŞ PRİNSİPİ:
1. İstifadəçinin nə istədiyini anla
2. Lazım olan tool-ları seç
3. Tool nəticələrini yoxla
4. Final cavabı ver
{memory_context}
"""


# ══════════════════════════════════════════════════════════════════════════
# 19) AUTONOMOUS AGENT
# ══════════════════════════════════════════════════════════════════════════
class ForgeAgent:
    """
    Agentic loop:
      Plan → Execute → Reflect
    Streaming ilə token-by-token.
    """

    def __init__(self, session: Session, permission: Permission,
                 max_turns: int = MAX_TURNS):
        self.session = session
        self.permission = permission
        self.executor = Executor(permission, session)
        self.max_turns = max_turns
        self.ctx = ContextManager()
        self.system_prompt = build_system_prompt()

    def _run_turn(self, stream: bool = True) -> Tuple[str, List[Dict]]:
        """
        Bir turn icra et.
        Returns: (final_content, tool_calls)
        """
        if self.ctx.should_compact():
            console.print("[dim]🗜️  Compacting context...[/dim]")
            self.ctx.compact()

        if stream:
            chunk = STREAM.stream_llm(self.ctx.messages, tools=TOOLS)
            content = STREAM.get_final_content()
            tool_calls = STREAM.get_tool_calls()

            # Add to context
            self.ctx.add(AIMessage(content=content))

            return content, tool_calls
        else:
            # Non-streaming (fallback)
            RATE.wait()
            try:
                llm = LLM_CLIENT.get_with_tools(TOOLS)
                r = llm.invoke(self.ctx.messages)
                content = r.content or ""
                tool_calls = []
                if hasattr(r, "tool_calls") and r.tool_calls:
                    for tc in r.tool_calls:
                        tool_calls.append({
                            "id": tc.get("id", f"call_{random.randint(1000,9999)}"),
                            "name": tc.get("name", ""),
                            "args": tc.get("args", {}),
                        })
                self.ctx.add(r)
                return content, tool_calls
            except Exception as e:
                return f"❌ {e}", []

    def run(self, user_input: str, stream: bool = True) -> str:
        """Əsas agentic loop."""
        self.session.append({"type": "user", "content": user_input})

        # Add to context
        self.ctx.add(SystemMessage(content=self.system_prompt))
        self.ctx.add(HumanMessage(content=user_input))

        final_content = ""

        for turn in range(self.max_turns):
            CANCEL.check()

            # Turn icra
            content, tool_calls = self._run_turn(stream=stream)
            final_content = content

            # Tool calls yoxdursa → final
            if not tool_calls:
                self.session.append({"type": "assistant", "content": content})
                return content

            # Tool-ları paralel icra et
            console.print(f"\n[bold cyan]🔧 {len(tool_calls)} tool icra olunur...[/bold cyan]")
            tool_results = self.executor.execute_parallel(tool_calls)

            # Context-ə əlavə
            self.ctx.extend(tool_results)

            # Session
            self.session.append({
                "type": "tool_use",
                "turn": turn,
                "count": len(tool_calls),
            })

        return final_content or "⚠️ Max turns"

    def run_autonomous(self, goal: str, max_steps: int = AUTONOMOUS_MAX_STEPS) -> str:
        """
        Autonomous rejim: Plan → Execute → Reflect.
        İstifadəçi müdaxiləsi olmadan.
        """
        console.print(Panel.fit(
            f"[bold]🎯 MƏQSƏD:[/bold] {goal}\n"
            f"[dim]Rejim: Plan → Execute → Reflect[/dim]",
            border_style="cyan"))

        # 1. PLAN
        plan = plan_task(goal)

        if not plan.get("steps"):
            console.print("[red]❌ Plan yaradılmadı[/red]")
            return self.run(goal, stream=True)

        steps = plan["steps"]
        results = []

        # 2. EXECUTE (hər addım)
        for step in steps:
            CANCEL.check()
            n = step.get("n", "?")
            action = step.get("action", "")
            tools = step.get("tools", [])

            console.print()
            console.rule(f"[cyan]Addım {n}: {action[:50]}[/cyan]")

            # Addımı agent kimi icra et
            step_prompt = f"""Məqsəd: {goal}

Plan addımı #{n}: {action}
Gözlənilən: {step.get('expected', '')}
Tövsiyə olunan tool-lar: {', '.join(tools)}

Bu addımı icra et."""

            # Hər addım ayrıca context
            step_ctx = ContextManager()
            step_ctx.add(SystemMessage(content=self.system_prompt))
            step_ctx.add(HumanMessage(content=step_prompt))

            step_result = ""
            for inner_turn in range(10):
                CANCEL.check()
                if step_ctx.should_compact():
                    step_ctx.compact()

                chunk = STREAM.stream_llm(step_ctx.messages, tools=TOOLS)
                content = STREAM.get_final_content()
                tool_calls = STREAM.get_tool_calls()

                step_ctx.add(AIMessage(content=content))
                step_result = content

                if not tool_calls:
                    break

                console.print(f"[dim]   🔧 {len(tool_calls)} tool[/dim]")
                tool_results = self.executor.execute_parallel(tool_calls)
                step_ctx.extend(tool_results)

            results.append({
                "step": n,
                "action": action,
                "result": step_result[:500],
            })

        # 3. REFLECT
        console.print()
        console.rule("[cyan]🧐 Refleksiya[/cyan]")

        summary = "\n\n".join(
            f"### Addım {r['step']}: {r['action']}\n{r['result']}"
            for r in results)

        reflection = reflect(goal, summary)
        score = reflection.get("score", 5)

        console.print(f"[cyan]Bal: {score}/10[/cyan]")

        if reflection.get("issues"):
            for issue in reflection["issues"][:3]:
                console.print(f"   [yellow]⚠️  {issue}[/yellow]")

        # 4. FINAL SUMMARY
        console.print()
        console.rule("[bold cyan]🎉 Yekun[/bold cyan]")

        # Yadda saxla
        h_memory_save(f"Task: {goal[:60]}",
                     f"Score: {score}/10\n\n{summary[:1000]}")

        final = f"""## 🎯 Nəticə

**Məqsəd:** {goal}

**Addımlar:** {len(results)}/{len(steps)}

**Bal:** {score}/10

### İcra olunanlar
{summary[:3000]}

### Refleksiya
{reflection.get('suggestion', '(yoxdur)')}
"""
        self.session.append({"type": "assistant", "content": final})
        return final


# ══════════════════════════════════════════════════════════════════════════
# 20) CLI
# ══════════════════════════════════════════════════════════════════════════
BANNER = r"""
[bold cyan]
  ██████╗  ██████╗ ██████╗  ██████╗ ███████╗
  ██╔══██╗██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
  ██████╔╝██║   ██║██████╔╝██║  ███╗█████╗  
  ██╔══██╗██║   ██║██╔══██╗██║   ██║██╔══╝  
  ██║  ██║╚██████╔╝██║  ██║╚██████╔╝███████╗
  ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
[/bold cyan]
[bold yellow]  v4.0 — Autonomous + Streaming Agent[/bold yellow]
"""

HELP = """
[bold cyan]🔨 Forge v4.0 — Autonomous Agent[/bold cyan]

[bold]Əsas əmrlər:[/bold]
  [green]/help[/green]      — Bu kömək
  [green]/mode[/green]      — İcazə rejimi
  [green]/tools[/green]     — Tool siyahısı (16)
  [green]/stream[/green]    — Streaming on/off
  [green]/memory[/green]    — Yaddaşı göstər
  [green]/skills[/green]    — Skill-ləri göstər
  [green]/clear[/green]     — Ekran
  [green]/exit[/green]      — Çıxış

[bold]Rejimlər:[/bold]
  [cyan]default[/cyan]      — Hər tool üçün soruşur
  [cyan]acceptEdits[/cyan]  — Fayl yazmağa avtomatik icazə
  [cyan]plan[/cyan]         — Yalnız oxuma
  [cyan]auto[/cyan]         — Hamısına icazə
  [cyan]bypass[/cyan]       — Tam bypass

[bold]İstifadə:[/bold]
  • Normal: "Bu qovluqdakı python fayllarını say"
  • Autonomous: /auto <məqsəd>
  • Streaming: /stream on

[dim]💡 Hermes + Claude Code arxitekturası:
   • Streaming (token-by-token)
   • Plan → Execute → Reflect
   • Memory (MEMORY.md)
   • Skills (auto-create)
   • Parallel tools (8 workers)[/dim]
"""

STYLE = Style.from_dict({"prompt": "bold #00d4ff"})


def interactive():
    console.print(BANNER)
    console.print(Panel(HELP, border_style="blue"))

    session_id = f"forge_{int(time.time())}"
    session = Session(session_id, WORK_DIR)
    permission = Permission(PermissionMode.DEFAULT)
    agent = ForgeAgent(session, permission)
    streaming = True

    console.print(
        f"[bold]Model:[/bold] [green]{MODEL}[/green]  |  "
        f"[bold]Tools:[/bold] [yellow]{len(TOOLS)}[/yellow]  |  "
        f"[bold]Mode:[/bold] [cyan]{permission.mode.value}[/cyan]  |  "
        f"[bold]Stream:[/bold] [cyan]{'on' if streaming else 'off'}[/cyan]  |  "
        f"[bold]Dir:[/bold] [dim]{WORK_DIR}[/dim]\n")

    pt = PromptSession(history=FileHistory(str(HISTORY_FILE)), style=STYLE)

    while True:
        try:
            user_input = pt.prompt("\n💬 Sən: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n👋")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/exit", "/quit"):
                console.print("👋 Görüşənədək!")
                break
            elif cmd == "/help":
                console.print(Panel(HELP, border_style="blue"))
            elif cmd == "/tools":
                t = Table(title=f"🔧 Tools ({len(TOOLS)})", box=box.ROUNDED)
                t.add_column("#", style="cyan", justify="right")
                t.add_column("Ad", style="green")
                t.add_column("Təsvir", style="dim")
                for i, tool_obj in enumerate(TOOLS, 1):
                    desc = (tool_obj.description or "").split("\n")[0][:60]
                    t.add_row(str(i), tool_obj.name, desc)
                console.print(t)
            elif cmd == "/mode":
                modes = [m.value for m in PermissionMode]
                if arg in modes:
                    permission.mode = PermissionMode(arg)
                    console.print(f"[green]✅ Mode: {arg}[/green]")
                else:
                    console.print(f"Mövcud: {', '.join(modes)}")
                    console.print(f"Cari: {permission.mode.value}")
            elif cmd == "/stream":
                if arg == "on":
                    streaming = True
                    console.print("[green]✅ Streaming ON[/green]")
                elif arg == "off":
                    streaming = False
                    console.print("[yellow]⏸ Streaming OFF[/yellow]")
                else:
                    console.print(f"Streaming: {streaming}")
            elif cmd == "/memory":
                mem = h_memory_read()
                console.print(Panel(mem[:2000], title="🧠 Memory"))
            elif cmd == "/skills":
                skills = h_skill_list()
                console.print(Panel(skills, title="🎯 Skills"))
            elif cmd == "/clear":
                console.clear()
            elif cmd == "/auto":
                if not arg:
                    console.print("[yellow]İstifadə: /auto <məqsəd>[/yellow]")
                    continue
                try:
                    result = agent.run_autonomous(arg)
                    console.print("\n[bold cyan]🤖 Forge:[/bold cyan]")
                    console.print(Markdown(result))
                except KeyboardInterrupt:
                    console.print("\n[yellow]⏸[/yellow]")
            else:
                console.print("[red]❌ Naməlum əmr[/red]")
            continue

        # Normal agent run
        try:
            result = agent.run(user_input, stream=streaming)
            if not streaming and result:
                console.print("\n[bold cyan]🤖 Forge:[/bold cyan]")
                console.print(Markdown(result))
        except KeyboardInterrupt:
            console.print("\n[yellow]⏸  Dayandırıldı[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ {e}[/red]")
            log.exception("agent_run")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Forge v4.0")
    ap.add_argument("--prompt", "-p", help="Birbaşa prompt")
    ap.add_argument("--autonomous", "-a", help="Autonomous rejim")
    ap.add_argument("--mode", default="default",
                    choices=["default", "acceptEdits", "plan", "auto", "bypass"])
    ap.add_argument("--no-stream", action="store_true")
    args = ap.parse_args()

    install_signal_handler()

    permission = Permission(PermissionMode(args.mode))
    session_id = f"forge_{int(time.time())}"
    session = Session(session_id, WORK_DIR)
    agent = ForgeAgent(session, permission)

    if args.autonomous:
        console.print(BANNER)
        result = agent.run_autonomous(args.autonomous)
        console.print("\n[bold cyan]🤖 Forge:[/bold cyan]")
        console.print(Markdown(result))
        return

    if args.prompt:
        console.print(BANNER)
        result = agent.run(args.prompt, stream=not args.no_stream)
        if args.no_stream:
            console.print("\n[bold cyan]🤖 Forge:[/bold cyan]")
            console.print(Markdown(result))
        return

    interactive()


if __name__ == "__main__":
    main()
