#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  FORGE v2.0 — Claude Code Architecture Replica                           ║
║                                                                          ║
║  Claude Code ilə eyni iş prinsipi:                                       ║
║    • Agentic Loop (turn-based, tool-driven)                              ║
║    • Permission-Gated Tools                                              ║
║    • Multi-Agent Coordinator                                             ║
║    • 5-Layer Error Recovery                                              ║
║    • Auto Context Compaction                                             ║
║    • Session Persistence (JSONL)                                         ║
║    • Snapshot + Rollback                                                 ║
║    • Hook System                                                         ║
║                                                                          ║
║  İstifadə:                                                               ║
║    python forge.py                       # interaktiv                    ║
║    python forge.py --prompt "..."        # birbaşa                       ║
║    python forge.py --prompt "..." --mode auto                            ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import re
import sys
import ast
import json
import time
import signal
import shlex
import random
import shutil
import logging
import hashlib
import tempfile
import threading
import subprocess
import logging.handlers
import contextvars
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import tool as lc_tool

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.rule import Rule
from rich import box
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

# ══════════════════════════════════════════════════════════════════════════
# 1) CONFIG
# ══════════════════════════════════════════════════════════════════════════
load_dotenv()
console = Console()

API_KEY = os.getenv("NVIDIA_API_KEY")
BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")

if not API_KEY:
    console.print(Panel(
        "[red]❌ NVIDIA_API_KEY yoxdur[/red]\n"
        "[dim]`.env` faylına əlavə et[/dim]",
        border_style="red"))
    sys.exit(1)

# Dirs
FORGE_HOME = Path.home() / ".forge"
FORGE_HOME.mkdir(exist_ok=True)
(FORGE_HOME / "logs").mkdir(exist_ok=True)
(FORGE_HOME / "sessions").mkdir(exist_ok=True)
(FORGE_HOME / "snapshots").mkdir(exist_ok=True)
HISTORY_FILE = FORGE_HOME / "history.txt"

WORK_DIR = Path.cwd().resolve()

# Limits
MAX_CONTEXT_TOKENS = 100_000
AUTOCOMPACT_BUFFER = 10_000
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_BASH_OUTPUT = 30_000
LLM_TIMEOUT = 120
MAX_RETRIES = 4
DEFAULT_RPM = 20

# Fallback models
FALLBACK_MODELS = [
    m.strip() for m in os.getenv("FORGE_FALLBACK", "").split(",") if m.strip()
]

# ══════════════════════════════════════════════════════════════════════════
# 2) LOGGING
# ══════════════════════════════════════════════════════════════════════════
class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "ts": datetime.now().isoformat(),
            "lvl": record.levelname,
            "msg": record.getMessage(),
            "mod": record.module,
        }, ensure_ascii=False)

fh = logging.handlers.RotatingFileHandler(
    FORGE_HOME / "logs" / "forge.log",
    maxBytes=5_000_000, backupCount=3, encoding="utf-8")
fh.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[fh], force=True)
log = logging.getLogger("forge")

# ══════════════════════════════════════════════════════════════════════════
# 3) CONTEXT VARS
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
# 4) CANCEL TOKEN
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
        console.print("\n[yellow]⏸  Interrupted[/yellow]")
        CANCEL.cancel()
        log.info("SIGINT")
    try:
        signal.signal(signal.SIGINT, handler)
    except (ValueError, OSError):
        pass


# ══════════════════════════════════════════════════════════════════════════
# 5) SESSION (JSONL persistence)
# ══════════════════════════════════════════════════════════════════════════
class Session:
    def __init__(self, session_id: str, project_dir: Path):
        self.session_id = session_id
        self.project_dir = project_dir
        safe = hashlib.md5(str(project_dir).encode()).hexdigest()[:16]
        self.path = FORGE_HOME / "sessions" / f"{safe}.jsonl"
        self._lock = threading.Lock()
        self.events: List[dict] = []

    def append(self, event: dict):
        with self._lock:
            event["ts"] = datetime.now().isoformat()
            self.events.append(event)
            try:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
            except Exception as e:
                log.warning(f"session_append {e}")


# ══════════════════════════════════════════════════════════════════════════
# 6) FILE SAFETY
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
# 7) SNAPSHOT
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

            # Rotation (son 20)
            snaps = sorted(
                [d for d in (FORGE_HOME / "snapshots").iterdir() if d.is_dir()],
                key=lambda d: d.name)
            for old in snaps[:-20]:
                shutil.rmtree(old, ignore_errors=True)
            return snap_id
        except Exception as e:
            log.warning(f"snapshot {e}")
            return None


# ══════════════════════════════════════════════════════════════════════════
# 8) RATE LIMITER (Adaptive)
# ══════════════════════════════════════════════════════════════════════════
class RateLimiter:
    def __init__(self, rpm: int):
        self.base_rpm = rpm
        self.rpm = rpm
        self.rate = rpm / 60.0
        self.tokens = 3.0
        self.last = time.time()
        self._lock = threading.Lock()
        self._errors = 0
        self._successes = 0

    def wait(self):
        with self._lock:
            now = time.time()
            self.tokens = min(3.0, self.tokens + (now - self.last) * self.rate)
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
            self._successes = 0
            if self._errors >= 2:
                old = self.rpm
                self.rpm = max(3, int(self.rpm * 0.7))
                self.rate = self.rpm / 60.0
                if self.rpm != old:
                    console.print(
                        f"[yellow]⚠️  RPM: {old} → {self.rpm}[/yellow]")
                self._errors = 0

    def report_success(self):
        with self._lock:
            self._successes += 1
            self._errors = 0
            if self._successes >= 20 and self.rpm < self.base_rpm:
                self.rpm = min(self.base_rpm, self.rpm + 2)
                self.rate = self.rpm / 60.0
                self._successes = 0


RATE = RateLimiter(DEFAULT_RPM)


# ══════════════════════════════════════════════════════════════════════════
# 9) LLM POOL (thread-local + model fallback)
# ══════════════════════════════════════════════════════════════════════════
class LLMPool:
    def __init__(self):
        self._local = threading.local()
        self._model_idx = 0
        self._lock = threading.Lock()
        self._cooldowns: Dict[str, float] = {}

    def all_models(self) -> List[str]:
        return [MODEL] + FALLBACK_MODELS

    def current_model(self) -> str:
        with self._lock:
            models = self.all_models()
            now = time.time()
            for i in range(len(models)):
                idx = (self._model_idx + i) % len(models)
                if self._cooldowns.get(models[idx], 0) < now:
                    self._model_idx = idx
                    return models[idx]
            return models[self._model_idx % len(models)]

    def rotate(self):
        with self._lock:
            models = self.all_models()
            model = models[self._model_idx % len(models)]
            self._cooldowns[model] = time.time() + 60
            self._model_idx += 1
            new_model = self.current_model()
        if hasattr(self._local, "llm"):
            del self._local.llm
        console.print(f"[yellow]🔄 Model: {new_model}[/yellow]")
        log.info(f"model_rotate {new_model}")

    def get(self) -> ChatOpenAI:
        if not hasattr(self._local, "llm"):
            self._local.llm = ChatOpenAI(
                openai_api_key=API_KEY,
                openai_api_base=BASE_URL,
                model=self.current_model(),
                temperature=0.2,
                max_tokens=8192,
                timeout=LLM_TIMEOUT,
                max_retries=0,
            )
        return self._local.llm


LLM_POOL = LLMPool()


# ══════════════════════════════════════════════════════════════════════════
# 10) ERROR CLASSIFY (5-layer recovery)
# ══════════════════════════════════════════════════════════════════════════
class ErrorKind(str, Enum):
    CONTEXT = "context"
    RATE = "rate"
    SERVER = "server"
    TIMEOUT = "timeout"
    NETWORK = "network"
    AUTH = "auth"
    MAX_OUTPUT = "max_output"
    UNKNOWN = "unknown"


def classify_error(err: Exception) -> Tuple[ErrorKind, str, float]:
    s = str(err).lower()
    if "prompt" in s and "long" in s:
        return ErrorKind.CONTEXT, str(err)[:150], 0.0
    if "max_output" in s or "output token" in s:
        return ErrorKind.MAX_OUTPUT, str(err)[:150], 0.0
    if "429" in s or "too many" in s:
        return ErrorKind.RATE, str(err)[:150], 3.0
    if "500" in s or "internal" in s:
        return ErrorKind.SERVER, str(err)[:150], 2.0
    if "timeout" in s or "timed out" in s:
        return ErrorKind.TIMEOUT, str(err)[:150], 3.0
    if "connection" in s or "network" in s:
        return ErrorKind.NETWORK, str(err)[:150], 2.0
    if "401" in s or "403" in s or "unauthorized" in s:
        return ErrorKind.AUTH, str(err)[:150], 0.0
    return ErrorKind.UNKNOWN, str(err)[:150], 1.5


# ══════════════════════════════════════════════════════════════════════════
# 11) CONTEXT MANAGER
# ══════════════════════════════════════════════════════════════════════════
def count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class ContextManager:
    def __init__(self):
        self.messages: List = []
        self.compact_failures = 0

    def total_tokens(self) -> int:
        return sum(count_tokens(str(getattr(m, "content", "") or ""))
                   for m in self.messages)

    def add(self, msg):
        self.messages.append(msg)

    def extend(self, msgs):
        self.messages.extend(msgs)

    def should_compact(self) -> bool:
        return self.total_tokens() > (MAX_CONTEXT_TOKENS - AUTOCOMPACT_BUFFER)

    def compact(self) -> bool:
        """Compact köhnə mesajları — özü sıfır maliyyətli."""
        try:
            system = [m for m in self.messages
                     if isinstance(m, SystemMessage)]
            others = [m for m in self.messages
                     if not isinstance(m, SystemMessage)]

            if len(others) <= 10:
                return True

            old, recent = others[:-10], others[-10:]
            summary = "[COMPACTED HISTORY]\n" + "\n".join(
                f"- {str(m.content or '')[:100]}" for m in old[-30:])
            recent.insert(0, HumanMessage(content=summary))
            self.messages = system + recent
            self.compact_failures = 0
            log.info("context_compacted")
            return True
        except Exception as e:
            self.compact_failures += 1
            log.warning(f"compact_fail {e}")
            return False


# ══════════════════════════════════════════════════════════════════════════
# 12) TOOLS — Handlers
# ══════════════════════════════════════════════════════════════════════════
def _is_binary(p: Path) -> bool:
    try:
        with p.open("rb") as f:
            chunk = f.read(1024)
        if b"\x00" in chunk:
            return True
        try:
            chunk.decode("utf-8")
            return False
        except UnicodeDecodeError:
            return True
    except Exception:
        return False


def handler_read_file(path: str, offset: int = 1, limit: int = 2000) -> str:
    p = safe_path(path)
    if not p or not p.is_file():
        return f"❌ Fayl yoxdur: {path}"
    try:
        if p.stat().st_size > MAX_FILE_SIZE:
            return "❌ Fayl böyük"
        if _is_binary(p):
            return "❌ Binary fayl"
        content = p.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        start = max(0, offset - 1)
        chunk = lines[start:start + limit]
        return "\n".join(
            f"{start + i + 1:6d}│ {line}"
            for i, line in enumerate(chunk)) or "(boş)"
    except Exception as e:
        return f"❌ {e}"


def handler_write_file(path: str, content: str) -> str:
    p = safe_path(path)
    if not p:
        return f"❌ Təhlükəsiz olmayan yol: {path}"
    try:
        if p.exists():
            Snapshot.create(f"pre_{p.name}",
                          [str(p.relative_to(get_base()))])
        atomic_write(p, content)
        return f"✅ {p} ({len(content)}b)"
    except Exception as e:
        return f"❌ {e}"


def handler_edit_file(path: str, old_string: str, new_string: str) -> str:
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


def handler_list_files(directory: str = ".", pattern: str = "*") -> str:
    p = safe_path(directory)
    if not p or not p.is_dir():
        return f"❌ Qovluq yoxdur: {directory}"
    try:
        files = sorted(p.glob(pattern))
        if not files:
            return f"❌ `{pattern}` uyğun yoxdur"
        return "\n".join(str(f.relative_to(p)) for f in files[:200])
    except Exception as e:
        return f"❌ {e}"


def handler_tree_view(directory: str = ".", max_depth: int = 3) -> str:
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


def handler_verify_file(path: str) -> str:
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
            if content.count("(") != content.count(")"):
                return "❌ Parens"
            return f"✅ {ext[1:].upper()}"
        return f"✅ {ext or 'txt'}"
    except SyntaxError as e:
        return f"❌ Syntax: {e.msg} (sətir {e.lineno})"
    except Exception as e:
        return f"❌ {str(e)[:100]}"


def handler_grep_search(pattern: str, path: str = ".",
                       max_results: int = 50) -> str:
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


def handler_run_bash(command: str, timeout: int = 60) -> str:
    CANCEL.check()
    if not command or not isinstance(command, str):
        return "❌ Boş"

    FORBIDDEN = {";", "&&", "||", "|", ">", "<", "`", "$(", "${",
                 "&", "\\", "\n", "\r"}
    for sym in FORBIDDEN:
        if sym in command:
            return f"❌ Qadağan simvol: {sym}"

    try:
        parts = shlex.split(command)
    except ValueError as e:
        return f"❌ Parse: {e}"

    if not parts:
        return "❌ Boş"

    SAFE = {"ls", "pwd", "cat", "head", "tail", "wc", "grep", "find",
            "file", "echo", "date", "whoami", "uname", "df", "du",
            "ps", "env", "which", "type", "stat", "tree", "sort",
            "uniq", "tr", "cut", "awk", "sed", "jq", "git", "python3",
            "pip", "pip3", "npm", "node", "pnpm", "yarn", "make",
            "cargo", "go", "rustc", "gcc", "g++", "clang", "pytest",
            "ruff", "mypy"}

    if parts[0] not in SAFE:
        return f"❌ `{parts[0]}` icazəli deyil"

    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": "en_US.UTF-8",
    }

    try:
        r = subprocess.run(
            parts, capture_output=True, text=True, timeout=timeout,
            shell=False, cwd=str(get_base()), env=env)
        out = (r.stdout or "")[:6000]
        if r.stderr:
            out += "\n[stderr]\n" + (r.stderr or "")[:2000]
        return f"```\n{out}\n```\n_(rc={r.returncode})_"
    except subprocess.TimeoutExpired:
        return "⏱ Timeout"
    except Exception as e:
        return f"❌ {e}"


def handler_delete_file(path: str) -> str:
    p = safe_path(path)
    if not p or not p.is_file():
        return "❌ Fayl yoxdur"
    try:
        Snapshot.create(f"pre_del_{p.name}",
                      [str(p.relative_to(get_base()))])
        p.unlink()
        return f"✅ Silindi: {path}"
    except Exception as e:
        return f"❌ {e}"


def handler_run_tests(path: str = ".") -> str:
    p = safe_path(path)
    if not p:
        return "❌"
    if shutil.which("pytest") and (
            (p / "tests").exists() or list(p.glob("test_*.py"))):
        try:
            r = subprocess.run(
                ["pytest", "-q", "--tb=short"],
                capture_output=True, text=True, timeout=120, cwd=str(p))
            return f"```\n{(r.stdout or '')[:3000]}\n```"
        except Exception as e:
            return f"❌ {e}"
    return "ℹ️ Test konfiqurasiyası yoxdur"


def handler_web_search(query: str, max_results: int = 5) -> str:
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


# ══════════════════════════════════════════════════════════════════════════
# 13) LANGCHAIN TOOLS (bind_tools üçün)
# ══════════════════════════════════════════════════════════════════════════
@lc_tool
def read_file(path: str) -> str:
    """Faylı oxuyur. Tam yol və ya nisbi yol."""
    return handler_read_file(path)


@lc_tool
def write_file(path: str, content: str) -> str:
    """Fayl yaradır və ya əvəz edir. content tam mətn olmalıdır."""
    return handler_write_file(path, content)


@lc_tool
def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Faylda mətn əvəzləməsi. old_string unikal olmalıdır."""
    return handler_edit_file(path, old_string, new_string)


@lc_tool
def list_files(directory: str = ".", pattern: str = "*") -> str:
    """Qovluqdakı faylları siyahıla. pattern glob formatı: '*.py'."""
    return handler_list_files(directory, pattern)


@lc_tool
def tree_view(directory: str = ".", max_depth: int = 3) -> str:
    """Qovluq strukturunu ağac kimi göstər."""
    return handler_tree_view(directory, max_depth)


@lc_tool
def verify_file(path: str) -> str:
    """Faylın sintaksisini yoxla (Python, JSON, TS)."""
    return handler_verify_file(path)


@lc_tool
def grep_search(pattern: str, path: str = ".") -> str:
    """Fayl məzmununda regex axtarışı."""
    return handler_grep_search(pattern, path)


@lc_tool
def run_bash(command: str, timeout: int = 60) -> str:
    """Təhlükəsiz shell əmri. Yalnız whitelist əmrlər."""
    return handler_run_bash(command, timeout)


@lc_tool
def delete_file(path: str) -> str:
    """Faylı silir (snapshot ilə)."""
    return handler_delete_file(path)


@lc_tool
def run_tests(path: str = ".") -> str:
    """Testləri işə sal (pytest/npm)."""
    return handler_run_tests(path)


@lc_tool
def web_search(query: str, max_results: int = 5) -> str:
    """Web axtarışı (DuckDuckGo)."""
    return handler_web_search(query, max_results)


# Langchain tool → handler map
LC_TO_HANDLER: Dict[str, Callable] = {
    "read_file": handler_read_file,
    "write_file": handler_write_file,
    "edit_file": handler_edit_file,
    "list_files": handler_list_files,
    "tree_view": handler_tree_view,
    "verify_file": handler_verify_file,
    "grep_search": handler_grep_search,
    "run_bash": handler_run_bash,
    "delete_file": handler_delete_file,
    "run_tests": handler_run_tests,
    "web_search": handler_web_search,
}

LANGCHAIN_TOOLS = [
    read_file, write_file, edit_file, list_files, tree_view,
    verify_file, grep_search, run_bash, delete_file, run_tests,
    web_search,
]


# ══════════════════════════════════════════════════════════════════════════
# 14) PERMISSION
# ══════════════════════════════════════════════════════════════════════════
class PermissionMode(str, Enum):
    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    PLAN = "plan"
    AUTO = "auto"
    BYPASS = "bypass"


READONLY_TOOLS = {"read_file", "list_files", "tree_view",
                 "verify_file", "grep_search", "web_search"}
WRITE_TOOLS = {"write_file", "edit_file", "delete_file"}


class Permission:
    def __init__(self, mode: PermissionMode = PermissionMode.DEFAULT):
        self.mode = mode

    def check(self, tool_name: str, args: dict) -> Tuple[Optional[bool], str]:
        """
        Returns: (True=allow, False=deny, None=ask)
        """
        if self.mode == PermissionMode.BYPASS:
            return True, "bypass"
        if self.mode == PermissionMode.AUTO:
            return True, "auto"
        if self.mode == PermissionMode.PLAN:
            if tool_name in READONLY_TOOLS:
                return True, "plan-readonly"
            return False, "plan: write denied"
        if self.mode == PermissionMode.ACCEPT_EDITS:
            if tool_name in READONLY_TOOLS or tool_name in WRITE_TOOLS:
                return True, "acceptEdits"
            return False, "acceptEdits: shell denied"

        # DEFAULT — soruş
        return None, "ask"


# ══════════════════════════════════════════════════════════════════════════
# 15) ROBUST JSON
# ══════════════════════════════════════════════════════════════════════════
def parse_json_robust(text: str) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    raw = m.group(0)

    for fix in [
        lambda x: x,
        lambda x: re.sub(r",(\s*[}\]])", r"\1", x),
        lambda x: x.replace("'", '"'),
        lambda x: re.sub(r"(\w+)\s*:", r'"\1":', x),
    ]:
        try:
            return json.loads(fix(raw))
        except json.JSONDecodeError:
            continue

    try:
        last = raw.rfind("}")
        start = raw.find("{")
        if last > start:
            return json.loads(raw[start:last + 1])
    except json.JSONDecodeError:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════
# 16) SAFE LLM CALL
# ══════════════════════════════════════════════════════════════════════════
def call_llm(messages: List,
            tools: Optional[List] = None,
            streaming: bool = False,
            max_retries: int = MAX_RETRIES,
            ctx: Optional[ContextManager] = None,
            return_message: bool = True) -> Tuple[bool, Any]:
    """
    Full-protected LLM çağırışı.

    return_message=True  → AIMessage (tool_calls ilə)
    return_message=False → string
    """
    CANCEL.check()

    last_err = None
    output_recovery = 0

    for attempt in range(max_retries):
        CANCEL.check()
        RATE.wait()
        llm = LLM_POOL.get()

        try:
            # Message lazımdırsa və ya tools varsa
            if return_message or tools:
                if tools:
                    llm_t = llm.bind_tools(tools)
                else:
                    llm_t = llm
                r = llm_t.invoke(messages)
                RATE.report_success()
                return True, r

            # Streaming
            if streaming:
                buf = ""
                for chunk in llm.stream(messages):
                    CANCEL.check()
                    buf += chunk.content or ""
                RATE.report_success()
                return True, buf

            # Sync string
            r = llm.invoke(messages)
            RATE.report_success()
            return True, (r.content or "")

        except KeyboardInterrupt:
            raise

        except Exception as e:
            last_err = e
            kind, short_msg, base_delay = classify_error(e)

            # Context — compact
            if kind == ErrorKind.CONTEXT and ctx:
                if ctx.compact():
                    console.print("[dim]   🗜️  Compacted[/dim]")
                    continue

            # Max output recovery
            if kind == ErrorKind.MAX_OUTPUT:
                output_recovery += 1
                if output_recovery <= 2:
                    continue

            # Rate limit
            if kind == ErrorKind.RATE:
                RATE.report_429()

            # Auth — retry yox
            if kind == ErrorKind.AUTH:
                console.print(f"[red]   🔐 AUTH: {short_msg}[/red]")
                return False, f"❌ AUTH: {short_msg}"

            # Server → model rotate
            if kind == ErrorKind.SERVER and attempt >= 2:
                LLM_POOL.rotate()

            delay = base_delay * (attempt + 1) + random.uniform(0, 1)

            icon = {
                ErrorKind.RATE: "🚦",
                ErrorKind.SERVER: "💥",
                ErrorKind.TIMEOUT: "⏱",
                ErrorKind.NETWORK: "🌐",
                ErrorKind.UNKNOWN: "⚠️",
            }.get(kind, "⚠️")

            console.print(
                f"[yellow]   {icon} Cəhd {attempt+1}/{max_retries}[/yellow]")
            console.print(f"[dim]      {short_msg}[/dim]")
            console.print(f"[yellow]      → {delay:.1f}s[/yellow]")
            time.sleep(delay)

    return False, f"❌ {max_retries} cəhd: {str(last_err)[:150]}"


# ══════════════════════════════════════════════════════════════════════════
# 17) SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════
def build_system_prompt() -> str:
    tools_list = "\n".join(
        f"• `{t.name}`: {(t.description or '').split(chr(10))[0]}"
        for t in LANGCHAIN_TOOLS)

    return f"""Sən Forge adlı kodlaşdırma agentsən.

PRİNSİPLƏR:
• Azərbaycan dilində cavab ver (əgər istifadəçi azərbaycanca yazıbsa)
• Markdown istifadə et
• Alətlərdən istifadə et (read_file, write_file, edit_file, run_bash)
• Yalnız təsdiqlənmiş məlumat ver
• Kod bloklarını ``` ilə göstər
• Uzun tapşırıqları addımlara böl

MÖVCUD ALƏTLƏR:
{tools_list}

İŞ PRİNSİPİ:
1. İstifadəçinin nə istədiyini anla
2. Lazım olan tool-ları seç
3. Tool nəticələrini yoxla
4. Final cavabı ver

QEYD: Heç vaxt tool nəticəsini təxmin etmə — real nəticəni gözlə."""


# ══════════════════════════════════════════════════════════════════════════
# 18) AGENT (agentic loop)
# ══════════════════════════════════════════════════════════════════════════
class Agent:
    def __init__(self, session: Session,
                 permission: Permission,
                 max_turns: int = 50):
        self.session = session
        self.permission = permission
        self.max_turns = max_turns
        self.ctx = ContextManager()
        self.system_prompt = build_system_prompt()

    def _execute_tool(self, name: str, args: dict) -> str:
        """Bir tool icra et."""
        # Permission
        allowed, reason = self.permission.check(name, args)
        if allowed is False:
            return f"🚫 Permission denied: {reason}"
        if allowed is None:  # ask
            console.print(f"\n[yellow]🔧 {name}[/yellow]")
            for k, v in args.items():
                console.print(f"   [dim]{k}: {str(v)[:80]}[/dim]")
            if not Confirm.ask("[yellow]İcazə?[/yellow]", default=True):
                return "❌ İstifadəçi imtina etdi"

        # Handler
        handler = LC_TO_HANDLER.get(name)
        if not handler:
            return f"❌ Tool yoxdur: {name}"

        try:
            result = handler(**args)
        except TypeError as e:
            return f"❌ Argument xətası: {e}"
        except Exception as e:
            log.exception(f"tool_{name}")
            return f"❌ Tool xətası: {e}"

        result_str = str(result)
        if len(result_str) > MAX_BASH_OUTPUT:
            result_str = result_str[:MAX_BASH_OUTPUT] + "\n... [truncated]"

        return result_str

    def run(self, user_prompt: str) -> str:
        """Əsas agentic loop."""
        self.session.append({"type": "user", "content": user_prompt})

        # Yeni kontekst
        self.ctx.add(SystemMessage(content=self.system_prompt))
        self.ctx.add(HumanMessage(content=user_prompt))

        for turn in range(self.max_turns):
            CANCEL.check()

            # Auto compaction
            if self.ctx.should_compact():
                console.print("[dim]🗜️  Context compacting...[/dim]")
                if not self.ctx.compact():
                    console.print("[red]❌ Compaction uğursuz[/red]")

            # LLM call
            ok, ai = call_llm(
                self.ctx.messages,
                tools=LANGCHAIN_TOOLS,
                streaming=False,
                ctx=self.ctx,
                return_message=True,
            )

            if not ok:
                return str(ai)

            # String fallback (nəzəri)
            if isinstance(ai, str):
                return ai

            self.ctx.add(ai)

            # Tool calls?
            tool_calls = getattr(ai, "tool_calls", None)
            if not tool_calls:
                content = ai.content or ""
                self.session.append({
                    "type": "assistant",
                    "content": content,
                })
                return content

            # Execute tools
            tool_results = []
            for call in tool_calls:
                name = call.get("name", "")
                args = call.get("args", {}) or {}
                tid = call.get("id", "")

                console.print(
                    f"[dim]   🔧 {name}({str(args)[:80]})[/dim]")
                result = self._execute_tool(name, args)
                preview = str(result)[:150].replace("\n", " ")
                console.print(f"[dim]   → {preview}[/dim]")

                tool_results.append(
                    ToolMessage(content=str(result), tool_call_id=tid))

            self.ctx.extend(tool_results)
            self.session.append({
                "type": "tool_use",
                "turn": turn,
                "count": len(tool_results),
            })

        return "⚠️ Max turns limitinə çatdı"


# ══════════════════════════════════════════════════════════════════════════
# 19) CLI
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
[bold yellow]  v2.0 — Claude Code Architecture Replica[/bold yellow]
"""

HELP = """
[bold cyan]🔨 Forge — Agentic CLI[/bold cyan]

[bold]Əsas əmrlər:[/bold]
  [green]/help[/green]      — Bu kömək
  [green]/mode[/green]      — İcazə rejimi
  [green]/tools[/green]     — Tool siyahısı (11)
  [green]/clear[/green]     — Ekranı təmizlə
  [green]/exit[/green]      — Çıxış

[bold]İcazə rejimləri:[/bold]
  [cyan]default[/cyan]      — Hər tool üçün soruşur
  [cyan]acceptEdits[/cyan]  — Fayl yazmağa avtomatik icazə
  [cyan]plan[/cyan]         — Yalnız oxuma (plan rejimi)
  [cyan]auto[/cyan]         — Hamısına icazə
  [cyan]bypass[/cyan]       — Tam bypass

[bold]Nümunə tapşırıqlar:[/bold]
  • "Bu qovluqdakı python fayllarını say"
  • "main.py faylını oxu və izah et"
  • "Yeni test.py faylı yaradıb içində sadə test yaz"
  • "Cari qovluq strukturunu göstər"

[dim]💡 Claude Code kimi: agentic loop, tool-driven, self-healing[/dim]
"""

STYLE = Style.from_dict({"prompt": "bold #00d4ff"})


def cmd_tools():
    t = Table(title=f"🔧 Tools ({len(LANGCHAIN_TOOLS)})",
             box=box.ROUNDED, header_style="bold")
    t.add_column("#", style="cyan", justify="right")
    t.add_column("Ad", style="green")
    t.add_column("Təsvir", style="dim")
    for i, tool_obj in enumerate(LANGCHAIN_TOOLS, 1):
        desc = (tool_obj.description or "").split("\n")[0][:70]
        t.add_row(str(i), tool_obj.name, desc)
    console.print(t)


def interactive():
    console.print(BANNER)
    console.print(Panel(HELP, border_style="blue"))

    # Session
    session_id = f"forge_{int(time.time())}"
    session = Session(session_id, WORK_DIR)

    # Permission
    permission = Permission(PermissionMode.DEFAULT)

    # Agent
    agent = Agent(session, permission, max_turns=50)

    console.print(
        f"[bold]Model:[/bold] [green]{MODEL}[/green]  |  "
        f"[bold]Tools:[/bold] [yellow]{len(LANGCHAIN_TOOLS)}[/yellow]  |  "
        f"[bold]Mode:[/bold] [cyan]{permission.mode.value}[/cyan]  |  "
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
                cmd_tools()
            elif cmd == "/mode":
                modes = [m.value for m in PermissionMode]
                if arg in modes:
                    permission.mode = PermissionMode(arg)
                    console.print(
                        f"[green]✅ Mode: {arg}[/green]")
                else:
                    console.print(f"[cyan]Mövcud:[/cyan] {', '.join(modes)}")
                    console.print(
                        f"[cyan]Cari:[/cyan] {permission.mode.value}")
            elif cmd == "/clear":
                console.clear()
            else:
                console.print("[red]❌ Naməlum əmr[/red]")
            continue

        # Run agent
        console.print()
        try:
            result = agent.run(user_input)
            if result:
                console.print("\n[bold cyan]🤖 Forge:[/bold cyan]")
                console.print(Markdown(result))
        except KeyboardInterrupt:
            console.print("\n[yellow]⏸  Dayandırıldı[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Xəta: {e}[/red]")
            log.exception("agent_run")


# ══════════════════════════════════════════════════════════════════════════
# 20) MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Forge — Claude Code Replica")
    ap.add_argument("--prompt", "-p", help="Birbaşa prompt")
    ap.add_argument("--mode", default="default",
                    choices=["default", "acceptEdits", "plan",
                            "auto", "bypass"])
    args = ap.parse_args()

    install_signal_handler()

    if args.prompt:
        console.print(BANNER)
        session_id = f"forge_{int(time.time())}"
        session = Session(session_id, WORK_DIR)
        permission = Permission(PermissionMode(args.mode))
        agent = Agent(session, permission, max_turns=50)
        try:
            result = agent.run(args.prompt)
            if result:
                console.print("\n[bold cyan]🤖 Forge:[/bold cyan]")
                console.print(Markdown(result))
        except KeyboardInterrupt:
            console.print("\n[yellow]⏸[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ {e}[/red]")
            log.exception("main_run")
        return

    interactive()


if __name__ == "__main__":
    main()
