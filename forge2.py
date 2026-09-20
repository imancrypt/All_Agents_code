#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  FORGE v7.0 — Sub-Agent Mesh (Optimized)                                 ║
║                                                                          ║
║  Konsept:                                                                ║
║    • N sub-agent paralel işləyir                                         ║
║    • Ortaq task board (SQLite)                                           ║
║    • Hər agent özünə task götürür                                        ║
║    • Review/verification ortaq                                           ║
║    • Mesajlaşma (SQLite bus)                                             ║
║                                                                          ║
║  İstifadə:                                                               ║
║    python forge.py --mesh "böyük məqsəd" --workers 8                     ║
║    python forge.py                       # chatbot                       ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import contextvars
import fcntl
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
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage,
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
from rich.prompt import Confirm
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
    console.print(Panel("[red]❌ NVIDIA_API_KEY yoxdur[/red]",
                       border_style="red"))
    sys.exit(1)

# ─── Directories ───
FORGE_HOME = Path.home() / ".forge"
FORGE_HOME.mkdir(exist_ok=True)
for sub in ("logs", "sessions", "memory"):
    (FORGE_HOME / sub).mkdir(exist_ok=True)

HISTORY_FILE = FORGE_HOME / "history.txt"
MEMORY_FILE = FORGE_HOME / "memory" / "MEMORY.md"
MESH_DB = FORGE_HOME / "mesh.db"

WORK_DIR = Path.cwd().resolve()

# ─── Limits ───
MAX_CONTEXT_TOKENS = 100_000
AUTOCOMPACT_BUFFER = 10_000
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_BASH_OUTPUT = 30_000
LLM_TIMEOUT = 120
MAX_RETRIES = 6
RETRY_BASE_DELAY = 2.0
DEFAULT_RPM = 30
MAX_WORKERS = 16

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
logging.basicConfig(level=logging.WARNING, handlers=[fh], force=True)
log = logging.getLogger("forge")

# ══════════════════════════════════════════════════════════════════════════
# 3) CANCEL
# ══════════════════════════════════════════════════════════════════════════
class CancelToken:
    def __init__(self): self._event = threading.Event()
    def cancel(self): self._event.set()
    def reset(self): self._event.clear()
    def is_cancelled(self): return self._event.is_set()
    def check(self):
        if self._event.is_set(): raise KeyboardInterrupt("Cancelled")

CANCEL = CancelToken()

def install_signal_handler():
    def handler(sig, frame):
        if CANCEL.is_cancelled():
            console.print("\n[red]Force exit[/red]")
            sys.exit(1)
        console.print("\n[yellow]⏸  Interrupting...[/yellow]")
        CANCEL.cancel()
    try: signal.signal(signal.SIGINT, handler)
    except (ValueError, OSError): pass

# ══════════════════════════════════════════════════════════════════════════
# 4) CONTEXT VARS
# ══════════════════════════════════════════════════════════════════════════
_BASE_VAR = contextvars.ContextVar("base", default=WORK_DIR)
def get_base(): return _BASE_VAR.get()
def set_base(p): _BASE_VAR.set(Path(p).resolve())

# ══════════════════════════════════════════════════════════════════════════
# 5) SQLITE POOL (optimized thread-safe)
# ══════════════════════════════════════════════════════════════════════════
class DBPool:
    """Thread-local SQLite connections with WAL mode."""
    def __init__(self, path: Path):
        self.path = path
        self._local = threading.local()
        self._init_schema()

    def _init_schema(self):
        with sqlite3.connect(str(self.path), timeout=30) as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT,
                    files TEXT,
                    depends_on TEXT,
                    status TEXT DEFAULT 'pending',
                    assigned_to TEXT,
                    attempts INTEGER DEFAULT 0,
                    max_attempts INTEGER DEFAULT 3,
                    priority INTEGER DEFAULT 5,
                    created TEXT,
                    started TEXT,
                    completed TEXT,
                    result TEXT,
                    error TEXT,
                    checkpoint TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_status ON tasks(status);
                CREATE INDEX IF NOT EXISTS idx_priority ON tasks(priority DESC, id ASC);
                CREATE INDEX IF NOT EXISTS idx_assigned ON tasks(assigned_to);

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_agent TEXT, to_agent TEXT,
                    subject TEXT, body TEXT,
                    ts TEXT, read INTEGER DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_msg_to ON messages(to_agent, read);

                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT, value TEXT, author TEXT, ts TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_knowledge_key ON knowledge(key);
            """)

    def get(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(str(self.path), timeout=30)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=10000")
        return self._local.conn

DB = DBPool(MESH_DB)

# ══════════════════════════════════════════════════════════════════════════
# 6) FILE SAFETY
# ══════════════════════════════════════════════════════════════════════════
def safe_path(path: str, allow_work: bool = True) -> Optional[Path]:
    if not path or not isinstance(path, str) or "\x00" in path:
        return None
    try:
        base = get_base()
        p = Path(path)
        if not p.is_absolute(): p = base / p
        p = p.resolve()
        try:
            p.relative_to(base); return p
        except ValueError: pass
        if allow_work:
            try:
                p.relative_to(WORK_DIR); return p
            except ValueError: pass
        return None
    except Exception: return None

def atomic_write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp",
                              dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try: os.unlink(tmp)
        except Exception: pass
        raise

# ══════════════════════════════════════════════════════════════════════════
# 7) RATE LIMITER
# ══════════════════════════════════════════════════════════════════════════
class RateLimiter:
    def __init__(self, rpm: int = DEFAULT_RPM):
        self.rpm = rpm; self.rate = rpm / 60.0
        self.tokens = 5.0; self.last = time.time()
        self._lock = threading.Lock()
    def wait(self):
        with self._lock:
            now = time.time()
            self.tokens = min(5.0, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens < 1:
                time.sleep((1 - self.tokens) / self.rate)
                self.tokens = 0; self.last = time.time()
            else: self.tokens -= 1

RATE = RateLimiter()

# ══════════════════════════════════════════════════════════════════════════
# 8) LLM CLIENT
# ══════════════════════════════════════════════════════════════════════════
class LLMClient:
    def __init__(self): self._local = threading.local()
    def get(self):
        if not hasattr(self._local, "llm"):
            self._local.llm = ChatOpenAI(
                openai_api_key=API_KEY, openai_api_base=BASE_URL,
                model=MODEL, temperature=0.2, max_tokens=8192,
                timeout=LLM_TIMEOUT, max_retries=0)
        return self._local.llm
    def with_tools(self, tools):
        if not hasattr(self._local, "llm_tools"):
            self._local.llm_tools = self.get().bind_tools(tools)
        return self._local.llm_tools

LLM_CLIENT = LLMClient()

# ══════════════════════════════════════════════════════════════════════════
# 9) RETRY
# ══════════════════════════════════════════════════════════════════════════
def is_retryable(err):
    s = str(err).lower()
    if "429" in s or "too many" in s: return True, "rate_limit"
    if "overloaded" in s or "503" in s: return True, "overloaded"
    if "500" in s or "internal server" in s: return True, "server_error"
    if "timeout" in s or "timed out" in s: return True, "timeout"
    if "connection" in s or "network" in s: return True, "network"
    if "401" in s or "403" in s: return False, "auth"
    if "context" in s and "length" in s: return False, "context"
    return True, "unknown"

def retry_delay(kind, attempt):
    return {
        "rate_limit": (5.0 * (attempt + 1) + random.uniform(0, 2), "🚦"),
        "overloaded": (3.0 * (attempt + 1) + random.uniform(0, 1), "🔥"),
        "server_error": (2.0 * (attempt + 1) + random.uniform(0, 1), "💥"),
        "timeout": (3.0 * (attempt + 1), "⏱"),
        "network": (2.0 * (attempt + 1), "🌐"),
    }.get(kind, (RETRY_BASE_DELAY * (attempt + 1), "⚠️"))

def call_llm(func, *args, max_retries=MAX_RETRIES, show=False, **kwargs):
    last_err = None
    for attempt in range(max_retries):
        CANCEL.check()
        try:
            RATE.wait()
            return True, func(*args, **kwargs)
        except KeyboardInterrupt: raise
        except Exception as e:
            last_err = e
            should, kind = is_retryable(e)
            if not should: return False, e
            delay, icon = retry_delay(kind, attempt)
            if show:
                console.print(f"[yellow]   {icon} Cəhd {attempt+1}/{max_retries} ({kind}) — {delay:.1f}s[/yellow]")
            time.sleep(delay)
    return False, last_err

def stream_llm(messages, tools=None, show=False, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        CANCEL.check()
        llm = LLM_CLIENT.with_tools(tools) if tools else LLM_CLIENT.get()
        buf = ""; tc_list = []
        try:
            RATE.wait()
            if show:
                live = Live(console=console, refresh_per_second=15,
                          vertical_overflow="visible")
                with live:
                    for chunk in llm.stream(messages):
                        CANCEL.check()
                        if chunk.content:
                            buf += chunk.content
                            try: live.update(Markdown(buf))
                            except Exception: live.update(buf)
                        if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                            for tc in chunk.tool_call_chunks:
                                _accum_tc(tc_list, tc)
            else:
                for chunk in llm.stream(messages):
                    CANCEL.check()
                    if chunk.content: buf += chunk.content
                    if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                        for tc in chunk.tool_call_chunks:
                            _accum_tc(tc_list, tc)
            parsed = []
            for tc in tc_list:
                if not tc["name"]: continue
                try: args = json.loads(tc["args"]) if tc["args"] else {}
                except json.JSONDecodeError: args = {}
                parsed.append({"id": tc["id"] or f"call_{random.randint(1000,9999)}",
                              "name": tc["name"], "args": args})
            return buf, parsed
        except KeyboardInterrupt: raise
        except Exception as e:
            should, kind = is_retryable(e)
            if not should:
                if show: console.print(f"[red]   ❌ {kind}: {str(e)[:100]}[/red]")
                return "", []
            delay, icon = retry_delay(kind, attempt)
            if show:
                console.print(f"[yellow]   {icon} Stream cəhd {attempt+1}/{max_retries} ({kind}) — {delay:.1f}s[/yellow]")
            time.sleep(delay)
    return "", []

def _accum_tc(tc_list, tc):
    idx = tc.get("index", 0)
    while len(tc_list) <= idx:
        tc_list.append({"id": "", "name": "", "args": ""})
    if tc.get("id"): tc_list[idx]["id"] = tc["id"]
    if tc.get("name"): tc_list[idx]["name"] = tc["name"]
    if tc.get("args"): tc_list[idx]["args"] += tc["args"]

# ══════════════════════════════════════════════════════════════════════════
# 10) CONTEXT
# ══════════════════════════════════════════════════════════════════════════
class ContextManager:
    def __init__(self): self.messages = []
    def total_tokens(self):
        return sum(max(1, len(str(getattr(m, "content", "") or "")) // 4)
                  for m in self.messages)
    def add(self, msg): self.messages.append(msg)
    def extend(self, msgs): self.messages.extend(msgs)
    def should_compact(self):
        return self.total_tokens() > (MAX_CONTEXT_TOKENS - AUTOCOMPACT_BUFFER)
    def compact(self):
        try:
            system = [m for m in self.messages if isinstance(m, SystemMessage)]
            others = [m for m in self.messages if not isinstance(m, SystemMessage)]
            if len(others) <= 10: return True
            old, recent = others[:-10], others[-10:]
            summary = "[COMPACTED]\n" + "\n".join(
                f"- {str(m.content or '')[:100]}" for m in old[-30:])
            recent.insert(0, HumanMessage(content=summary))
            self.messages = system + recent
            return True
        except Exception: return False

# ══════════════════════════════════════════════════════════════════════════
# 11) TOOLS
# ══════════════════════════════════════════════════════════════════════════
def _is_binary(p):
    try:
        with p.open("rb") as f: return b"\x00" in f.read(1024)
    except Exception: return False

def h_read_file(path, offset=1, limit=2000):
    p = safe_path(path)
    if not p or not p.is_file(): return f"❌ Fayl yoxdur: {path}"
    try:
        if p.stat().st_size > MAX_FILE_SIZE: return "❌ Fayl böyük"
        if _is_binary(p): return "❌ Binary"
        content = p.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        start = max(0, offset - 1)
        chunk = lines[start:start + limit]
        return "\n".join(f"{start + i + 1:6d}│ {line}"
                        for i, line in enumerate(chunk)) or "(boş)"
    except Exception as e: return f"❌ {e}"

def h_write_file(path, content):
    p = safe_path(path)
    if not p: return f"❌ Təhlükəsiz olmayan yol: {path}"
    try:
        atomic_write(p, content)
        return f"✅ {p} ({len(content)}b)"
    except Exception as e: return f"❌ {e}"

def h_edit_file(path, old_string, new_string):
    p = safe_path(path)
    if not p or not p.is_file(): return "❌ Fayl yoxdur"
    try:
        content = p.read_text(encoding="utf-8")
        count = content.count(old_string)
        if count == 0: return "❌ `old_string` tapılmadı"
        if count > 1: return f"❌ `old_string` {count} dəfə təkrar"
        atomic_write(p, content.replace(old_string, new_string, 1))
        return f"✅ {p}"
    except Exception as e: return f"❌ {e}"

def h_list_files(directory=".", pattern="*"):
    p = safe_path(directory)
    if not p or not p.is_dir(): return "❌ Qovluq yoxdur"
    try:
        files = sorted(p.glob(pattern))
        if not files: return f"❌ `{pattern}` uyğun yoxdur"
        return "\n".join(str(f.relative_to(p)) for f in files[:200])
    except Exception as e: return f"❌ {e}"

def h_tree_view(directory=".", max_depth=3):
    p = safe_path(directory)
    if not p or not p.is_dir(): return "❌ Qovluq yoxdur"
    SKIP = {".git", "__pycache__", "node_modules", ".venv", "venv",
            "dist", "build", ".turbo"}
    lines = [str(p) + "/"]
    def walk(d, prefix="", depth=0):
        if depth >= max_depth: return
        try:
            items = sorted([x for x in d.iterdir()
                           if x.name not in SKIP and not x.name.startswith(".")],
                          key=lambda x: (x.is_file(), x.name))[:40]
        except PermissionError: return
        for i, it in enumerate(items):
            last = i == len(items) - 1
            marker = "└── " if last else "├── "
            lines.append(f"{prefix}{marker}{it.name}" + ("/" if it.is_dir() else ""))
            if it.is_dir(): walk(it, prefix + ("    " if last else "│   "), depth + 1)
    walk(p)
    return "```\n" + "\n".join(lines) + "\n```"

def h_verify_file(path):
    p = safe_path(path)
    if not p or not p.is_file(): return "❌ Fayl yoxdur"
    ext = p.suffix.lower()
    try:
        content = p.read_text(encoding="utf-8", errors="replace")
        if not content.strip(): return "❌ Boş"
        if ext == ".json": json.loads(content); return "✅ JSON"
        if ext == ".py": compile(content, str(p), "exec"); return "✅ Python"
        if ext in (".ts", ".tsx", ".js", ".jsx"):
            if content.count("{") != content.count("}"): return "❌ Braces"
            return f"✅ {ext[1:].upper()}"
        return f"✅ {ext or 'txt'}"
    except SyntaxError as e: return f"❌ Syntax: {e.msg} (sətir {e.lineno})"
    except Exception as e: return f"❌ {str(e)[:100]}"

def h_grep_search(pattern, path=".", max_results=50):
    p = safe_path(path)
    if not p: return "❌"
    try:
        cmd = ["rg", "-n", "--no-heading", pattern, str(p)] \
              if shutil.which("rg") else ["grep", "-rn", "-E", pattern, str(p)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        out = r.stdout or ""
        return "\n".join(out.splitlines()[:max_results]) or "🔍 yoxdur"
    except Exception as e: return f"❌ {e}"

def h_run_bash(command, timeout=60):
    CANCEL.check()
    if not command: return "❌ Boş"
    FORBIDDEN = {";", "&&", "||", "|", ">", "<", "`", "$(", "${",
                 "&", "\\", "\n", "\r"}
    for sym in FORBIDDEN:
        if sym in command: return f"❌ Qadağan: {sym}"
    try: parts = shlex.split(command)
    except ValueError as e: return f"❌ Parse: {e}"
    if not parts: return "❌ Boş"
    SAFE = {"ls", "pwd", "cat", "head", "tail", "wc", "grep", "find",
            "file", "stat", "tree", "sort", "uniq", "tr", "cut", "awk",
            "sed", "jq", "du", "df", "which", "type", "mkdir", "touch",
            "echo", "date", "whoami", "uname", "ps", "env", "git",
            "python3", "python", "pip", "pip3", "npm", "node",
            "pnpm", "yarn", "make", "pytest", "ruff", "mypy"}
    if parts[0] not in SAFE: return f"❌ `{parts[0]}` icazəli deyil"
    DANGEROUS = {"-rf", "-fr", "--no-preserve-root"}
    for arg in parts[1:]:
        if arg in DANGEROUS: return f"❌ Təhlükəli: {arg}"
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
           "HOME": os.environ.get("HOME", "/tmp"),
           "LANG": "en_US.UTF-8"}
    try:
        r = subprocess.run(parts, capture_output=True, text=True,
                          timeout=timeout, shell=False,
                          cwd=str(get_base()), env=env)
        out = (r.stdout or "")[:6000]
        if r.stderr: out += "\n[stderr]\n" + (r.stderr or "")[:2000]
        return f"```\n{out}\n```\n_(rc={r.returncode})_"
    except subprocess.TimeoutExpired: return "⏱ Timeout"
    except Exception as e: return f"❌ {e}"

def h_run_tests(path="."):
    p = safe_path(path)
    if not p: return "❌"
    if shutil.which("pytest") and ((p / "tests").exists() or list(p.glob("test_*.py"))):
        try:
            r = subprocess.run(["pytest", "-q", "--tb=short"],
                             capture_output=True, text=True,
                             timeout=120, cwd=str(p))
            return f"```\n{(r.stdout or '')[:3000]}\n```"
        except Exception as e: return f"❌ {e}"
    return "ℹ️ Test yoxdur"

def h_web_search(query, max_results=5):
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return "\n\n".join(f"**{i}. {r['title']}**\n{r['href']}\n{r['body'][:200]}..."
                          for i, r in enumerate(results, 1)) or "🔍 yoxdur"
    except Exception as e: return f"❌ {e}"

def h_memory_save(key, value):
    try:
        with MEMORY_FILE.open("a", encoding="utf-8") as f:
            f.write(f"\n## {key}\n_{datetime.now().isoformat()}_\n\n{value}\n")
        return f"✅ Yadda saxlanıldı"
    except Exception as e: return f"❌ {e}"

def h_memory_read():
    if not MEMORY_FILE.exists(): return "(boş)"
    try: return MEMORY_FILE.read_text(encoding="utf-8")[:5000]
    except Exception as e: return f"❌ {e}"

# ─── LangChain tool wrappers ───
@lc_tool
def read_file(path: str, offset: int = 1, limit: int = 2000) -> str:
    """Faylı oxuyur."""
    return h_read_file(path, offset, limit)

@lc_tool
def write_file(path: str, content: str) -> str:
    """Fayl yaradır. Qovluqlar avtomatik."""
    return h_write_file(path, content)

@lc_tool
def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Faylda mətn əvəzləməsi."""
    return h_edit_file(path, old_string, new_string)

@lc_tool
def list_files(directory: str = ".", pattern: str = "*") -> str:
    """Faylları siyahıla."""
    return h_list_files(directory, pattern)

@lc_tool
def tree_view(directory: str = ".", max_depth: int = 3) -> str:
    """Ağac struktur."""
    return h_tree_view(directory, max_depth)

@lc_tool
def verify_file(path: str) -> str:
    """Sintaksis yoxla."""
    return h_verify_file(path)

@lc_tool
def grep_search(pattern: str, path: str = ".") -> str:
    """Regex axtarışı."""
    return h_grep_search(pattern, path)

@lc_tool
def run_bash(command: str, timeout: int = 60) -> str:
    """Shell əmri."""
    return h_run_bash(command, timeout)

@lc_tool
def run_tests(path: str = ".") -> str:
    """Testlər."""
    return h_run_tests(path)

@lc_tool
def web_search(query: str, max_results: int = 5) -> str:
    """Web axtarışı."""
    return h_web_search(query, max_results)

@lc_tool
def memory_save(key: str, value: str) -> str:
    """Yaddaşa yaz."""
    return h_memory_save(key, value)

@lc_tool
def memory_read() -> str:
    """Yaddaşı oxu."""
    return h_memory_read()

TOOLS = [read_file, write_file, edit_file, list_files, tree_view,
         verify_file, grep_search, run_bash, run_tests, web_search,
         memory_save, memory_read]
TOOL_HANDLERS = {t.name: globals()[f"h_{t.name}"] for t in TOOLS
                 if f"h_{t.name}" in globals()}

# ══════════════════════════════════════════════════════════════════════════
# 12) TASK BOARD (optimized)
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class Task:
    id: int
    title: str
    description: str
    files: List[str]
    depends_on: List[int]
    status: str
    assigned_to: Optional[str]
    attempts: int
    max_attempts: int
    priority: int
    created: str
    started: Optional[str]
    completed: Optional[str]
    result: Optional[str]
    error: Optional[str]
    checkpoint: Optional[str]

class TaskBoard:
    """Optimized task board with dependency check."""

    def add(self, title: str, description: str = "",
            files: List[str] = None, depends_on: List[int] = None,
            priority: int = 5) -> int:
        conn = DB.get()
        with conn:
            cur = conn.execute(
                """INSERT INTO tasks
                   (title, description, files, depends_on, priority, created)
                   VALUES (?,?,?,?,?,?)""",
                (title, description, json.dumps(files or []),
                 json.dumps(depends_on or []), priority,
                 datetime.now().isoformat()))
            return cur.lastrowid

    def claim(self, agent: str) -> Optional[Task]:
        """Atomic claim. Busy timeout ilə."""
        conn = DB.get()
        # BEGIN IMMEDIATE — lock bütün transaction üçün
        try:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """SELECT * FROM tasks WHERE status = 'pending'
                   ORDER BY priority DESC, id ASC LIMIT 30""").fetchall()

            for row in rows:
                task = self._row_to_task(row)

                # Dependency yoxla
                if task.depends_on:
                    deps_ok = True
                    for dep_id in task.depends_on:
                        dep = conn.execute(
                            "SELECT status FROM tasks WHERE id = ?",
                            (dep_id,)).fetchone()
                        if not dep or dep["status"] != "done":
                            deps_ok = False
                            break
                    if not deps_ok:
                        continue

                # Claim
                cur = conn.execute(
                    """UPDATE tasks SET status = 'running',
                       assigned_to = ?, started = ?,
                       attempts = attempts + 1
                       WHERE id = ? AND status = 'pending'""",
                    (agent, datetime.now().isoformat(), task.id))
                if cur.rowcount > 0:
                    conn.commit()
                    task.status = "running"
                    task.assigned_to = agent
                    return task

            conn.commit()
            return None
        except Exception as e:
            log.exception(f"claim error: {e}")
            try: conn.rollback()
            except Exception: pass
            return None

    def _row_to_task(self, row) -> Task:
        return Task(
            id=row["id"], title=row["title"],
            description=row["description"] or "",
            files=json.loads(row["files"] or "[]"),
            depends_on=json.loads(row["depends_on"] or "[]"),
            status=row["status"], assigned_to=row["assigned_to"],
            attempts=row["attempts"], max_attempts=row["max_attempts"],
            priority=row["priority"], created=row["created"],
            started=row["started"], completed=row["completed"],
            result=row["result"], error=row["error"],
            checkpoint=row["checkpoint"])

    def complete(self, task_id: int, result: str):
        conn = DB.get()
        with conn:
            conn.execute(
                """UPDATE tasks SET status = 'done',
                   completed = ?, result = ? WHERE id = ?""",
                (datetime.now().isoformat(), result[:3000], task_id))

    def fail(self, task_id: int, error: str):
        conn = DB.get()
        with conn:
            row = conn.execute(
                "SELECT attempts, max_attempts FROM tasks WHERE id = ?",
                (task_id,)).fetchone()
            if row:
                if row["attempts"] >= row["max_attempts"]:
                    conn.execute(
                        """UPDATE tasks SET status = 'failed',
                           error = ?, completed = ? WHERE id = ?""",
                        (error[:2000], datetime.now().isoformat(), task_id))
                else:
                    conn.execute(
                        """UPDATE tasks SET status = 'pending',
                           error = ?, assigned_to = NULL,
                           started = NULL WHERE id = ?""",
                        (error[:2000], task_id))

    def save_checkpoint(self, task_id: int, checkpoint: dict):
        conn = DB.get()
        with conn:
            conn.execute("UPDATE tasks SET checkpoint = ? WHERE id = ?",
                        (json.dumps(checkpoint), task_id))

    def list(self, status=None, limit=200) -> List[Task]:
        conn = DB.get()
        q = "SELECT * FROM tasks"
        params = []
        if status:
            q += " WHERE status = ?"
            params.append(status)
        q += " ORDER BY priority DESC, id ASC LIMIT ?"
        params.append(limit)
        rows = conn.execute(q, params).fetchall()
        return [self._row_to_task(r) for r in rows]

    def stats(self) -> Dict[str, int]:
        conn = DB.get()
        rows = conn.execute(
            "SELECT status, COUNT(*) as n FROM tasks GROUP BY status").fetchall()
        return {r["status"]: r["n"] for r in rows}

    def all_done(self) -> bool:
        s = self.stats()
        return (s.get("pending", 0) == 0 and
                s.get("running", 0) == 0)

BOARD = TaskBoard()

# ══════════════════════════════════════════════════════════════════════════
# 13) AGENT BUS
# ══════════════════════════════════════════════════════════════════════════
class AgentBus:
    def send(self, from_agent, to_agent, subject, body):
        conn = DB.get()
        with conn:
            conn.execute(
                """INSERT INTO messages
                   (from_agent, to_agent, subject, body, ts)
                   VALUES (?,?,?,?,?)""",
                (from_agent, to_agent, subject, body[:2000],
                 datetime.now().isoformat()))

    def inbox(self, agent, unread_only=True) -> List[dict]:
        conn = DB.get()
        q = "SELECT * FROM messages WHERE to_agent = ?"
        if unread_only: q += " AND read = 0"
        q += " ORDER BY id DESC LIMIT 20"
        rows = conn.execute(q, (agent,)).fetchall()
        return [dict(r) for r in rows]

    def mark_read(self, msg_id):
        conn = DB.get()
        with conn:
            conn.execute("UPDATE messages SET read = 1 WHERE id = ?", (msg_id,))

    def stats(self):
        conn = DB.get()
        row = conn.execute("SELECT COUNT(*), COALESCE(SUM(read),0) FROM messages").fetchone()
        return {"total": row[0], "read": row[1]}

BUS = AgentBus()

# ══════════════════════════════════════════════════════════════════════════
# 14) PLANNER
# ══════════════════════════════════════════════════════════════════════════
PLANNER_PROMPT = """Sən layihə planlayıcısı-san. Məqsədi ATOMİK task-lara böl.

MƏQSƏD: {goal}

ÇIXIŞ (yalnız JSON):
{{
  "understanding": "1 cümlə",
  "tasks": [
    {{"title": "qısa başlıq", "description": "nə ediləcək",
      "files": ["path/file.ext"], "priority": 8}}
  ]
}}

QAYDALAR:
1. Hər task MÜSTƏQİL (paralel işləyə bilər)
2. 3-30 task
3. Eyni fayl 2 task-da olmasın
4. Priority 1-10 (10 ən vacib)
"""

def plan_tasks(goal: str, show: bool = True) -> List[dict]:
    if show:
        console.print()
        console.rule("[cyan]📋 Planlama[/cyan]")

    prompt = PLANNER_PROMPT.format(goal=goal[:2000])

    def _call():
        return LLM_CLIENT.get().invoke([
            SystemMessage(content="Yalnız JSON qaytar."),
            HumanMessage(content=prompt),
        ])

    ok, result = call_llm(_call, show=show)
    if not ok:
        console.print(f"[red]❌ Planlama xətası: {result}[/red]")
        return []

    text = result.content or ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m: return []

    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        fixed = re.sub(r",(\s*[}\]])", r"\1", m.group(0))
        try: data = json.loads(fixed)
        except Exception: return []

    tasks = data.get("tasks", [])
    if show:
        console.print(f"[green]✅ {len(tasks)} task planlandı[/green]")
        if tasks:
            t = Table(title="📋 Plan", box=box.ROUNDED)
            t.add_column("#", style="cyan", justify="right")
            t.add_column("Başlıq", style="white")
            t.add_column("Fayllar", style="dim")
            t.add_column("P", style="yellow", justify="right")
            for i, task in enumerate(tasks[:20], 1):
                t.add_row(str(i), task.get("title", "")[:55],
                         ", ".join(task.get("files", []))[:30],
                         str(task.get("priority", 5)))
            console.print(t)

    return tasks

# ══════════════════════════════════════════════════════════════════════════
# 15) SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════
def build_worker_prompt(worker_name: str) -> str:
    tools_list = "\n".join(
        f"• `{t.name}`: {(t.description or '').split(chr(10))[0]}"
        for t in TOOLS)

    memory_context = ""
    if MEMORY_FILE.exists():
        try:
            mem = MEMORY_FILE.read_text(encoding="utf-8")[:1500]
            if mem.strip():
                memory_context = f"\n\nYADDAŞ:\n{mem}\n"
        except Exception: pass

    return f"""Sən {worker_name} — sub-agent-sən.

PRİNSİPLƏR:
• Azərbaycan dilində cavab ver
• Markdown istifadə et
• Alətlərdən istifadə et — təxmin etmə
• Kod yazmaq üçün `write_file` İSTİFADƏ ET
• Qovluqlar AVTOMATİK yaranır (mkdir lazım deyil)

MÖVCUD ALƏTLƏR:
{tools_list}
{memory_context}
"""

# ══════════════════════════════════════════════════════════════════════════
# 16) WORKER AGENT
# ══════════════════════════════════════════════════════════════════════════
class WorkerAgent:
    """Bir sub-agent — task board-dan iş götürür."""
    def __init__(self, worker_id: int):
        self.id = worker_id
        self.name = f"W{worker_id}"
        self.system_prompt = build_worker_prompt(self.name)
        self._stop = False
        self.completed = 0
        self.failed = 0

    def _read_inbox(self) -> str:
        msgs = BUS.inbox(self.name, unread_only=True)
        if not msgs: return ""
        lines = ["MESAJLAR:"]
        for m in msgs[:3]:
            lines.append(f"  • [{m['from_agent']}] {m['subject']}: {m['body'][:100]}")
            BUS.mark_read(m["id"])
        return "\n".join(lines)

    def execute_task(self, task: Task) -> bool:
        """Task-ı icra et."""
        ctx = ContextManager()
        ctx.add(SystemMessage(content=self.system_prompt))
        ctx.add(HumanMessage(content=(
            f"TASK: {task.title}\n\n"
            f"{task.description}\n\n"
            f"Fayllar: {', '.join(task.files) if task.files else '(sən seç)'}\n\n"
            f"İcra et.")))

        inbox = self._read_inbox()
        if inbox:
            ctx.add(HumanMessage(content=inbox))

        for iteration in range(15):
            CANCEL.check()
            if ctx.should_compact(): ctx.compact()

            content, tool_calls = stream_llm(
                ctx.messages, tools=TOOLS, show=False, max_retries=3)

            if not content and not tool_calls:
                BOARD.fail(task.id, "LLM cavab vermədi")
                return False

            ctx.add(AIMessage(content=content))

            if not tool_calls:
                BOARD.complete(task.id, content[:1000])
                return True

            # Tool-ları paralel icra et
            results = self._exec_tools_parallel(tool_calls)
            for tc, r in results:
                ctx.add(ToolMessage(content=str(r), tool_call_id=tc["id"]))

        BOARD.fail(task.id, "Max iterations")
        return False

    def _exec_tools_parallel(self, tool_calls) -> List[Tuple[dict, str]]:
        """Tool-ları paralel icra et."""
        if not tool_calls: return []

        results = [None] * len(tool_calls)

        def _do(i, tc):
            try:
                handler = TOOL_HANDLERS.get(tc["name"])
                if not handler:
                    return f"❌ Tool yoxdur: {tc['name']}"
                result = handler(**tc["args"])
                s = str(result)
                return s[:MAX_BASH_OUTPUT] if len(s) > MAX_BASH_OUTPUT else s
            except TypeError as e:
                return f"❌ Argument: {e}"
            except Exception as e:
                return f"❌ Tool: {e}"

        with ThreadPoolExecutor(max_workers=min(8, len(tool_calls))) as ex:
            futures = {ex.submit(_do, i, tc): i for i, tc in enumerate(tool_calls)}
            for fut in as_completed(futures):
                i = futures[fut]
                try: results[i] = fut.result()
                except Exception as e: results[i] = f"❌ {e}"

        return [(tool_calls[i], results[i]) for i in range(len(tool_calls))]

    def loop(self):
        """Əsas worker dövrü."""
        while not self._stop:
            CANCEL.check()
            try:
                task = BOARD.claim(self.name)
                if not task:
                    time.sleep(2)
                    continue

                console.print(f"[dim]   🔨 {self.name}: {task.title[:50]}[/dim]")

                try:
                    ok = self.execute_task(task)
                    if ok: self.completed += 1
                    else: self.failed += 1
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    log.exception(f"worker {self.name} task error")
                    BOARD.fail(task.id, str(e)[:200])

            except KeyboardInterrupt:
                break
            except Exception as e:
                log.exception(f"worker {self.name} loop error")
                time.sleep(3)

    def stop(self):
        self._stop = True

# ══════════════════════════════════════════════════════════════════════════
# 17) MESH — orkestrator
# ══════════════════════════════════════════════════════════════════════════
class Mesh:
    """
    Sub-agent mesh.
    N işçi paralel işləyir, ortaq task board-dan iş götürür.
    """
    def __init__(self, num_workers: int = 8):
        self.num_workers = max(1, min(num_workers, MAX_WORKERS))
        self.workers: List[WorkerAgent] = []
        for i in range(self.num_workers):
            self.workers.append(WorkerAgent(i + 1))

    def run(self, goal: str, max_runtime: int = 1800):
        """Mesh-i işə sal."""
        start = time.time()

        # Plan
        tasks = plan_tasks(goal, show=True)
        if not tasks:
            console.print("[red]❌ Heç bir task planlanmadı[/red]")
            return

        # Board-a əlavə et
        added = 0
        for t in tasks:
            BOARD.add(
                title=t.get("title", "Untitled"),
                description=t.get("description", ""),
                files=t.get("files", []),
                priority=t.get("priority", 5),
            )
            added += 1
        console.print(f"[green]✅ {added} task board-a əlavə edildi[/green]")

        # Start
        console.print()
        console.print(Panel.fit(
            f"[bold]🕸️  MESH İŞƏ BAŞLAYIR[/bold]\n\n"
            f"İşçilər: {len(self.workers)}\n"
            f"Task-lar: {added}",
            border_style="cyan"))

        # Workers paralel işləsin
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            futures = [executor.submit(w.loop) for w in self.workers]

            # Progress monitor
            try:
                last_stats = None
                while time.time() - start < max_runtime:
                    CANCEL.check()
                    time.sleep(5)

                    stats = BOARD.stats()
                    if stats != last_stats:
                        self._print_progress(stats, time.time() - start)
                        last_stats = dict(stats)

                    if BOARD.all_done():
                        console.print()
                        console.print("[green]✅ Bütün task-lar tamamlandı![/green]")
                        break
            except KeyboardInterrupt:
                console.print("\n[yellow]⏸  Dayandırılır...[/yellow]")
            finally:
                for w in self.workers:
                    w.stop()
                for f in futures:
                    f.cancel()

        # Yekun
        console.print()
        self._final_report(start)

    def _print_progress(self, stats: dict, elapsed: float):
        total = sum(stats.values())
        done = stats.get("done", 0)
        running = stats.get("running", 0)
        pending = stats.get("pending", 0)
        failed = stats.get("failed", 0)

        bar_len = 30
        progress = done / max(1, total)
        filled = int(bar_len * progress)
        bar = "█" * filled + "░" * (bar_len - filled)

        console.print(
            f"[cyan]⏱ {elapsed:.0f}s[/cyan]  "
            f"[bold]{bar}[/bold] {progress*100:.0f}%  "
            f"[green]✅ {done}[/green] "
            f"[yellow]🔄 {running}[/yellow] "
            f"[dim]⏳ {pending}[/dim] "
            f"[red]❌ {failed}[/red]")

    def _final_report(self, start: float):
        elapsed = time.time() - start
        stats = BOARD.stats()
        worker_stats = [(w.name, w.completed, w.failed) for w in self.workers]

        console.print()
        console.rule("[bold cyan]🎉 MESH YEKUN[/bold cyan]")

        t = Table(box=box.ROUNDED, show_header=False)
        t.add_column("", style="cyan")
        t.add_column("", style="white")
        t.add_row("Müddət", f"{elapsed:.0f}s")
        t.add_row("İşçilər", str(len(self.workers)))
        t.add_row("Tamamlandı", f"[green]{stats.get('done', 0)}[/green]")
        t.add_row("Uğursuz", f"[red]{stats.get('failed', 0)}[/red]")
        t.add_row("Pending", f"[yellow]{stats.get('pending', 0)}[/yellow]")
        t.add_row("Mesajlar", str(BUS.stats()["total"]))
        console.print(t)

        # Worker statistikası
        console.print()
        wt = Table(title="👥 İşçi Statistikası", box=box.ROUNDED)
        wt.add_column("Ad", style="cyan")
        wt.add_column("✅", style="green", justify="right")
        wt.add_column("❌", style="red", justify="right")
        for name, ok, fail in worker_stats:
            wt.add_row(name, str(ok), str(fail))
        console.print(wt)

        console.print()
        console.print(Markdown(h_tree_view(".", 3)))

# ══════════════════════════════════════════════════════════════════════════
# 18) CLI
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
[bold yellow]  v7.0 — Sub-Agent Mesh (Optimized)[/bold yellow]
"""

HELP = """
[bold cyan]🔨 Forge v7.0 — Sub-Agent Mesh[/bold cyan]

[bold]Mesh:[/bold]
  [green]/mesh <məqsəd>[/green]   — Mesh işə sal (paralel sub-agentlər)
  [green]/workers <N>[/green]     — İşçi sayı (1-16, default 8)
  [green]/board[/green]           — Task board
  [green]/messages[/green]        — Agent mesajları
  [green]/reset[/green]           — Board-u təmizlə

[bold]Chat:[/bold]
  Birbaşa yaz → cavab

[bold]Digər:[/bold]
  [green]/tools[/green] / [green]/clear[/green] / [green]/help[/green] / [green]/exit[/green]

[bold]Nümunə:[/bold]
  /workers 10
  /mesh FastAPI REST API: users, posts, comments + tests + docs

[dim]💡 Hər worker ortaq board-dan task götürür. Paralel, avtomatik.[/dim]
"""

STYLE = Style.from_dict({"prompt": "bold #00d4ff"})


def cmd_board():
    tasks = BOARD.list()
    if not tasks:
        console.print("[dim]Board boşdur[/dim]")
        return
    t = Table(title=f"📋 Task Board ({len(tasks)})", box=box.ROUNDED)
    t.add_column("ID", style="cyan", justify="right")
    t.add_column("Status", style="yellow")
    t.add_column("P", justify="right", style="dim")
    t.add_column("Title", style="white")
    t.add_column("Assigned", style="dim")
    for task in tasks[:30]:
        icon = {"pending": "⏳", "running": "🔄", "done": "✅",
                "failed": "❌"}.get(task.status, "?")
        t.add_row(str(task.id), f"{icon} {task.status}",
                 str(task.priority), task.title[:50],
                 (task.assigned_to or "-")[:20])
    console.print(t)


def cmd_messages():
    rows = DB.get().execute(
        "SELECT * FROM messages ORDER BY id DESC LIMIT 20").fetchall()
    if not rows:
        console.print("[dim]Mesaj yoxdur[/dim]")
        return
    t = Table(title="💬 Mesajlar", box=box.ROUNDED)
    t.add_column("Time", style="dim")
    t.add_column("From", style="cyan")
    t.add_column("To", style="green")
    t.add_column("Subject", style="white")
    for r in rows:
        t.add_row(r["ts"][11:19], r["from_agent"], r["to_agent"],
                 r["subject"][:40])
    console.print(t)


def cmd_tools():
    t = Table(title=f"🔧 Tools ({len(TOOLS)})", box=box.ROUNDED)
    t.add_column("#", style="cyan", justify="right")
    t.add_column("Ad", style="green")
    t.add_column("Təsvir", style="dim")
    for i, tool_obj in enumerate(TOOLS, 1):
        desc = (tool_obj.description or "").split("\n")[0][:60]
        t.add_row(str(i), tool_obj.name, desc)
    console.print(t)


def interactive():
    console.print(BANNER)
    console.print(Panel(HELP, border_style="blue"))

    num_workers = 8

    console.print(
        f"[bold]Model:[/bold] [green]{MODEL}[/green]  |  "
        f"[bold]Workers:[/bold] [yellow]{num_workers}[/yellow]  |  "
        f"[bold]Dir:[/bold] [dim]{WORK_DIR}[/dim]\n")

    pt = PromptSession(history=FileHistory(str(HISTORY_FILE)), style=STYLE)

    while True:
        CANCEL.reset()
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

            try:
                if cmd in ("/exit", "/quit"):
                    console.print("👋")
                    break
                elif cmd == "/help":
                    console.print(Panel(HELP, border_style="blue"))
                elif cmd == "/tools":
                    cmd_tools()
                elif cmd == "/board":
                    cmd_board()
                elif cmd == "/messages":
                    cmd_messages()
                elif cmd == "/workers":
                    if arg.isdigit():
                        num_workers = max(1, min(int(arg), MAX_WORKERS))
                        console.print(f"[green]✅ Workers: {num_workers}[/green]")
                    else:
                        console.print(f"Cari: {num_workers}")
                elif cmd == "/reset":
                    conn = DB.get()
                    with conn:
                        conn.execute("DELETE FROM tasks")
                        conn.execute("DELETE FROM messages")
                    console.print("[green]✅ Board təmizləndi[/green]")
                elif cmd == "/clear":
                    console.clear()
                elif cmd == "/mesh":
                    if not arg:
                        console.print("[yellow]/mesh <məqsəd>[/yellow]")
                        continue
                    try:
                        mesh = Mesh(num_workers=num_workers)
                        mesh.run(arg)
                    except KeyboardInterrupt:
                        console.print("\n[yellow]⏸[/yellow]")
                        CANCEL.reset()
                else:
                    console.print("[red]❌ Naməlum əmr[/red]")
            except KeyboardInterrupt:
                console.print("\n[yellow]⏸  Dayandırıldı[/yellow]")
                CANCEL.reset()
            except Exception as e:
                console.print(f"[red]❌ Əmr xətası: {e}[/red]")
                log.exception("cmd_error")
            continue

        # Sadə chat (tək agent)
        console.print("[dim]İpucu: Mesh üçün /mesh <məqsəd> istifadə et[/dim]")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Forge v7.0 — Sub-Agent Mesh")
    ap.add_argument("--mesh", "-m", help="Mesh məqsədi")
    ap.add_argument("--workers", "-w", type=int, default=8,
                    help="İşçi sayı (1-16)")
    ap.add_argument("--board", action="store_true", help="Board göstər")
    ap.add_argument("--messages", action="store_true", help="Mesajlar")
    ap.add_argument("--reset", action="store_true", help="Board təmizlə")
    args = ap.parse_args()

    install_signal_handler()
    CANCEL.reset()

    if args.reset:
        conn = DB.get()
        with conn:
            conn.execute("DELETE FROM tasks")
            conn.execute("DELETE FROM messages")
        console.print("[green]✅ Board təmizləndi[/green]")
        return

    if args.board:
        cmd_board()
        return

    if args.messages:
        cmd_messages()
        return

    if args.mesh:
        console.print(BANNER)
        mesh = Mesh(num_workers=args.workers)
        try:
            mesh.run(args.mesh)
        except KeyboardInterrupt:
            console.print("\n[yellow]⏸[/yellow]")
        return

    interactive()


if __name__ == "__main__":
    main()
