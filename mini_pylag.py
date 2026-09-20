#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MINI-CLAUDE v7.0 — Full Production + Project Resume                     ║
║                                                                          ║
║  Əsas:                                                                   ║
║    • Bütün v6 edge case fix-ləri                                         ║
║    • Dynamic base directory (spec-dən /path)                             ║
║    • Project state persistence                                           ║
║    • Resume yarımçıq layihələri                                          ║
║    • Project registry (/projects)                                        ║
║    • Smart path extraction                                               ║
║                                                                          ║
║  İstifadə:                                                               ║
║    python mini_claude_v7.py                                              ║
║    /build FastAPI app in /myproject                                      ║
║    /projects              # layihə siyahısı                              ║
║    /continue myproject    # davam et                                     ║
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
import zipfile
import tarfile
import logging
import hashlib
import tempfile
import threading
import subprocess
import logging.handlers
from collections import deque
from concurrent.futures import (
    ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeout,
)
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import tool

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.rule import Rule
from rich.live import Live
from rich import box

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

# ══════════════════════════════════════════════════════════════════════════
# KONFİQURASİYA
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

ROOT_DIR = Path.cwd()
DATA_DIR = Path.home() / ".mini_claude"
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "snapshots").mkdir(exist_ok=True)
(DATA_DIR / "logs").mkdir(exist_ok=True)
(DATA_DIR / "projects").mkdir(exist_ok=True)
HISTORY_FILE = DATA_DIR / "history.txt"
PROJECTS_DB = DATA_DIR / "projects.json"
LOCK_FILE = DATA_DIR / "session.lock"

WORK_DIR = ROOT_DIR.resolve()

# Limits
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "80000"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
MAX_TASKS = int(os.getenv("MAX_TASKS", "100"))
TASK_TIMEOUT_SEC = int(os.getenv("TASK_TIMEOUT", "600"))
SNAPSHOT_RETENTION = int(os.getenv("SNAPSHOT_RETENTION", "20"))

MAX_REQUESTS_PER_MIN = int(os.getenv("RPM", "30"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "1.5"))

FALLBACK_MODELS = [
    m.strip() for m in os.getenv("FALLBACK_MODELS", "").split(",")
    if m.strip()
]


# ══════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════
class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "ts": datetime.now().isoformat(),
            "level": record.levelname,
            "msg": record.getMessage(),
            "mod": record.module,
        }, ensure_ascii=False)


file_handler = logging.handlers.RotatingFileHandler(
    DATA_DIR / "logs" / "app.log",
    maxBytes=5_000_000, backupCount=3, encoding="utf-8")
file_handler.setFormatter(JsonFormatter())

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler],
    force=True,
)
log = logging.getLogger("mini")


# ══════════════════════════════════════════════════════════════════════════
# CANCELLATION
# ══════════════════════════════════════════════════════════════════════════
class CancelToken:
    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def check(self):
        if self._event.is_set():
            raise KeyboardInterrupt("Cancelled")

    def reset(self):
        self._event.clear()


CANCEL = CancelToken()


def _install_signal_handler():
    def handler(sig, frame):
        console.print("\n[yellow]⏸  Cancelling...[/yellow]")
        CANCEL.cancel()
        log.info("SIGINT received")
    try:
        signal.signal(signal.SIGINT, handler)
    except (ValueError, OSError):
        pass


# ══════════════════════════════════════════════════════════════════════════
# CROSS-PROCESS LOCK
# ══════════════════════════════════════════════════════════════════════════
class CrossProcessLock:
    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self.fd: Optional[int] = None

    def acquire(self, blocking: bool = False) -> bool:
        try:
            import fcntl
            self.fd = os.open(str(self.lock_path),
                            os.O_CREAT | os.O_RDWR)
            flags = fcntl.LOCK_EX
            if not blocking:
                flags |= fcntl.LOCK_NB
            fcntl.flock(self.fd, flags)
            return True
        except (ImportError, BlockingIOError, OSError):
            if self.fd is not None:
                try:
                    os.close(self.fd)
                except Exception:
                    pass
                self.fd = None
            return False

    def release(self):
        if self.fd is not None:
            try:
                import fcntl
                fcntl.flock(self.fd, fcntl.LOCK_UN)
                os.close(self.fd)
            except Exception:
                pass
            self.fd = None


# ══════════════════════════════════════════════════════════════════════════
# FILE LOCKS (thread)
# ══════════════════════════════════════════════════════════════════════════
class FileLockManager:
    def __init__(self):
        self._locks: Dict[str, threading.RLock] = {}
        self._global = threading.Lock()

    def acquire(self, path: str) -> threading.RLock:
        with self._global:
            if path not in self._locks:
                self._locks[path] = threading.RLock()
            lock = self._locks[path]
        lock.acquire()
        return lock

    def release(self, path: str):
        with self._global:
            lock = self._locks.get(path)
        if lock:
            try:
                lock.release()
            except RuntimeError:
                pass


LOCKS = FileLockManager()


# ══════════════════════════════════════════════════════════════════════════
# DYNAMIC BASE DIR — əsas fix
# ══════════════════════════════════════════════════════════════════════════
_CURRENT_BASE: Dict[str, Path] = {"dir": WORK_DIR}
_BASE_LOCK = threading.RLock()


def set_base_dir(new_base: Path):
    """Layihə qovluğunu dəyiş (thread-safe)."""
    with _BASE_LOCK:
        _CURRENT_BASE["dir"] = Path(new_base).resolve()


def get_base_dir() -> Path:
    with _BASE_LOCK:
        return _CURRENT_BASE["dir"]


def _safe(path: str) -> Optional[Path]:
    """Path-i CARİ BASE-dən resolve et."""
    if not path or not isinstance(path, str):
        return None
    if "\x00" in path:
        return None
    try:
        base = get_base_dir()
        p = Path(path)
        if not p.is_absolute():
            p = base / p
        p = p.resolve()

        # Base daxilində?
        if p == base or str(p).startswith(str(base) + os.sep):
            if p.is_symlink():
                real = p.resolve()
                if not (real == base or
                       str(real).startswith(str(base) + os.sep)):
                    return None
            return p

        # WORK_DIR daxilindədirsə OK (sessions, data)
        if p == WORK_DIR or str(p).startswith(str(WORK_DIR) + os.sep):
            return p

        return None
    except Exception:
        return None


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as f:
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


def _read_text_safe(path: Path,
                   max_size: int = MAX_FILE_SIZE) -> Tuple[bool, str]:
    try:
        st = path.stat()
        if st.st_size > max_size:
            return False, f"❌ Fayl çox böyük ({st.st_size} > {max_size})"
        if _is_binary(path):
            return False, "❌ Binary fayl"
        raw = path.read_bytes()
        for enc in ("utf-8", "utf-16", "cp1251", "latin-1"):
            try:
                return True, raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return False, "❌ Encoding tanınmadı"
    except Exception as e:
        return False, f"❌ {e}"


def _atomic_write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.stem + "_", suffix=".tmp",
        dir=str(path.parent))
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
# PROJECT REGISTRY — layihələri izləyir
# ══════════════════════════════════════════════════════════════════════════
class ProjectRegistry:
    """
    Bütün layihələri qeyd edir. Yarımçıq qalsa davam etmək üçün.
    """
    def __init__(self, path: Path = PROJECTS_DB):
        self.path = path
        self._lock = threading.Lock()
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"projects": {}}

    def save(self):
        with self._lock:
            try:
                _atomic_write(
                    self.path,
                    json.dumps(self.data, indent=2, ensure_ascii=False))
            except Exception as e:
                log.warning(f"registry_save_error {e}")

    def register(self, project_path: Path, name: str, spec: str) -> str:
        """Yeni layihə qeyd et."""
        project_id = hashlib.md5(
            str(project_path.resolve()).encode()).hexdigest()[:12]

        with self._lock:
            self.data["projects"][project_id] = {
                "id": project_id,
                "name": name,
                "path": str(project_path.resolve()),
                "spec": spec,
                "created": datetime.now().isoformat(),
                "updated": datetime.now().isoformat(),
                "status": "in_progress",  # in_progress | completed | failed
                "tasks_total": 0,
                "tasks_done": 0,
                "tasks_failed": 0,
            }
        self.save()
        return project_id

    def update(self, project_id: str, **fields):
        with self._lock:
            if project_id in self.data["projects"]:
                self.data["projects"][project_id].update(fields)
                self.data["projects"][project_id]["updated"] = \
                    datetime.now().isoformat()
        self.save()

    def get(self, project_id: str) -> Optional[dict]:
        return self.data["projects"].get(project_id)

    def find_by_path(self, path: Path) -> Optional[dict]:
        p = str(path.resolve())
        for proj in self.data["projects"].values():
            if proj["path"] == p:
                return proj
        return None

    def find_by_name(self, name: str) -> Optional[dict]:
        for proj in self.data["projects"].values():
            if proj["name"] == name:
                return proj
        return None

    def all(self) -> List[dict]:
        return list(self.data["projects"].values())

    def in_progress(self) -> List[dict]:
        return [p for p in self.all() if p["status"] == "in_progress"]


REGISTRY = ProjectRegistry()


# ══════════════════════════════════════════════════════════════════════════
# PROJECT STATE — hər layihənin öz state-i
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class SubTask:
    id: str
    title: str
    description: str
    files: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    status: str = "pending"
    result: str = ""
    error: str = ""
    duration: float = 0.0
    worker_id: int = -1
    retries: int = 0
    completed_at: Optional[str] = None


class ProjectState:
    """Layihə state-i — .mini_state.json faylında."""
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir.resolve()
        self.state_file = self.project_dir / ".mini_state.json"
        self._lock = threading.RLock()
        self.data = self._load()

    def _load(self) -> dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "created": datetime.now().isoformat(),
            "project_name": "",
            "spec": "",
            "tasks": [],  # list of dicts
            "current_idx": 0,
            "status": "pending",
        }

    def save(self):
        with self._lock:
            try:
                self.data["updated"] = datetime.now().isoformat()
                _atomic_write(
                    self.state_file,
                    json.dumps(self.data, indent=2, ensure_ascii=False))
            except Exception as e:
                log.warning(f"state_save_error {e}")

    def set_tasks(self, project_name: str, spec: str,
                 tasks: List[SubTask]):
        with self._lock:
            self.data["project_name"] = project_name
            self.data["spec"] = spec
            self.data["tasks"] = [asdict(t) for t in tasks]
            self.data["current_idx"] = 0
            self.data["status"] = "in_progress"
        self.save()

    def get_tasks(self) -> List[SubTask]:
        tasks = []
        for td in self.data.get("tasks", []):
            tasks.append(SubTask(**td))
        return tasks

    def update_task(self, task: SubTask):
        with self._lock:
            for i, td in enumerate(self.data["tasks"]):
                if td["id"] == task.id:
                    self.data["tasks"][i] = asdict(task)
                    break
        self.save()

    def is_complete(self) -> bool:
        tasks = self.get_tasks()
        return all(t.status in ("done", "skipped", "failed")
                  for t in tasks)

    def pending_tasks(self) -> List[SubTask]:
        return [t for t in self.get_tasks()
                if t.status not in ("done", "skipped", "failed")]

    def reset_failed_tasks(self):
        """Uğursuz task-ları yenidən cəhd üçün pending et."""
        with self._lock:
            for td in self.data["tasks"]:
                if td["status"] == "failed":
                    td["status"] = "pending"
                    td["error"] = ""
                    td["attempts"] = 0
        self.save()


# ══════════════════════════════════════════════════════════════════════════
# RATE LIMITER
# ══════════════════════════════════════════════════════════════════════════
class TokenBucket:
    def __init__(self, rate_per_minute: int, burst: int = 5):
        self.rate = rate_per_minute / 60.0
        self.burst = burst
        self.tokens = float(burst)
        self.last = time.time()
        self._lock = threading.Lock()
        self._errors = 0
        self._per_minute = rate_per_minute

    def wait(self, tokens: int = 1):
        with self._lock:
            now = time.time()
            self.tokens = min(
                self.burst,
                self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens < tokens:
                wait_time = (tokens - self.tokens) / self.rate
                time.sleep(wait_time)
                self.tokens = 0
                self.last = time.time()
            else:
                self.tokens -= tokens

    def report_429(self):
        with self._lock:
            self._errors += 1
            if self._errors >= 3 and self._per_minute > 5:
                self._per_minute = max(5, self._per_minute - 5)
                self.rate = self._per_minute / 60.0

    def report_ok(self):
        with self._lock:
            self._errors = max(0, self._errors - 1)

    @property
    def per_minute(self) -> int:
        return self._per_minute


RATE_LIMITER = TokenBucket(MAX_REQUESTS_PER_MIN)


# ══════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ══════════════════════════════════════════════════════════════════════════
class CircuitBreaker:
    def __init__(self, threshold: int = 20, timeout: int = 60,
                 name: str = "main"):
        self.threshold = threshold
        self.timeout = timeout
        self.name = name
        self.failures = 0
        self.opened_at: Optional[float] = None
        self.state = "closed"
        self._lock = threading.Lock()

    def check(self) -> bool:
        with self._lock:
            if self.state == "open":
                elapsed = time.time() - (self.opened_at or 0)
                if elapsed >= self.timeout:
                    self.state = "half-open"
                    return True
                return False
            return True

    def success(self):
        with self._lock:
            self.failures = 0
            self.state = "closed"

    def fail(self):
        with self._lock:
            self.failures += 1
            if self.failures >= self.threshold:
                self.state = "open"
                self.opened_at = time.time()
                console.print(
                    f"[red]⚡ [{self.name}] circuit açıldı[/red]")


CIRCUIT_MAIN = CircuitBreaker(threshold=20, timeout=60, name="main")
CIRCUIT_PER_MODEL: Dict[str, CircuitBreaker] = {}
_CIRCUIT_MODEL_LOCK = threading.Lock()


def get_model_circuit(model: str) -> CircuitBreaker:
    with _CIRCUIT_MODEL_LOCK:
        if model not in CIRCUIT_PER_MODEL:
            CIRCUIT_PER_MODEL[model] = CircuitBreaker(
                threshold=10, timeout=30, name=model[:30])
        return CIRCUIT_PER_MODEL[model]


# ══════════════════════════════════════════════════════════════════════════
# LLM POOL
# ══════════════════════════════════════════════════════════════════════════
class LLMPool:
    def __init__(self):
        self._local = threading.local()
        self._model_idx = 0
        self._model_lock = threading.Lock()
        self._model_health: Dict[str, dict] = {}

    def all_models(self) -> List[str]:
        return [MODEL] + FALLBACK_MODELS

    def current_model(self) -> str:
        with self._model_lock:
            models = self.all_models()
            return models[self._model_idx % len(models)]

    def rotate_model(self):
        with self._model_lock:
            self._model_idx += 1
            self._model_health[self.current_model()] = {
                "last_fail": time.time()}
        if hasattr(self._local, "llm"):
            del self._local.llm

    def get(self) -> ChatOpenAI:
        if not hasattr(self._local, "llm"):
            model = self.current_model()
            self._local.llm = ChatOpenAI(
                openai_api_key=API_KEY,
                openai_api_base=BASE_URL,
                model=model,
                temperature=0.2,
                max_tokens=MAX_OUTPUT_TOKENS,
                timeout=LLM_TIMEOUT,
                max_retries=0,
            )
        return self._local.llm

    def current_model_name(self) -> str:
        return self.current_model()


LLM_POOL = LLMPool()


# ══════════════════════════════════════════════════════════════════════════
# LLM CALLS
# ══════════════════════════════════════════════════════════════════════════
def safe_llm_invoke(messages: List, streaming: bool = True,
                   max_retries: int = MAX_RETRIES) -> Tuple[bool, str]:
    CANCEL.check()
    if not CIRCUIT_MAIN.check():
        time.sleep(10)

    llm = LLM_POOL.get()
    model = LLM_POOL.current_model_name()
    circuit = get_model_circuit(model)
    last_err = None

    for attempt in range(max_retries):
        CANCEL.check()
        if not circuit.check():
            time.sleep(5)
        RATE_LIMITER.wait()

        try:
            if streaming:
                buf = ""
                for chunk in llm.stream(messages):
                    CANCEL.check()
                    buf += chunk.content or ""
                RATE_LIMITER.report_ok()
                CIRCUIT_MAIN.success()
                circuit.success()
                return True, buf
            else:
                r = llm.invoke(messages)
                RATE_LIMITER.report_ok()
                CIRCUIT_MAIN.success()
                circuit.success()
                return True, (r.content or "")

        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = e
            CIRCUIT_MAIN.fail()
            circuit.fail()
            err_str = str(e).lower()

            if "429" in err_str or "too many" in err_str:
                RATE_LIMITER.report_429()
                delay = 3.0 * (attempt + 1) + random.uniform(0, 2)
            elif "500" in err_str or "internal" in err_str:
                delay = 2.0 * (attempt + 1) + random.uniform(0, 1)
                if attempt >= 2:
                    LLM_POOL.rotate_model()
            elif "timeout" in err_str:
                delay = 3.0 * (attempt + 1)
            else:
                delay = RETRY_BASE_DELAY * (attempt + 1)

            console.print(
                f"[yellow]   ⏱ Cəhd {attempt+1}/{max_retries} "
                f"({str(e)[:50]}) — {delay:.1f}s[/yellow]")
            time.sleep(delay)

    return False, f"❌ {max_retries} cəhddən sonra: {str(last_err)[:200]}"


def safe_llm_with_tools(messages: List, tools: List,
                       max_retries: int = MAX_RETRIES) -> Tuple[bool, Any]:
    CANCEL.check()
    if not CIRCUIT_MAIN.check():
        time.sleep(10)

    llm = LLM_POOL.get()
    model = LLM_POOL.current_model_name()
    circuit = get_model_circuit(model)
    llm_tools = llm.bind_tools(tools)
    last_err = None

    for attempt in range(max_retries):
        CANCEL.check()
        if not circuit.check():
            time.sleep(5)
        RATE_LIMITER.wait()

        try:
            r = llm_tools.invoke(messages)
            RATE_LIMITER.report_ok()
            CIRCUIT_MAIN.success()
            circuit.success()
            return True, r

        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = e
            CIRCUIT_MAIN.fail()
            circuit.fail()
            err_str = str(e).lower()

            if "429" in err_str:
                RATE_LIMITER.report_429()
                delay = 3.0 * (attempt + 1) + random.uniform(0, 2)
            elif "500" in err_str:
                delay = 2.0 * (attempt + 1) + random.uniform(0, 1)
                if attempt >= 2:
                    LLM_POOL.rotate_model()
            elif "timeout" in err_str:
                delay = 3.0 * (attempt + 1)
            else:
                delay = RETRY_BASE_DELAY * (attempt + 1)

            console.print(
                f"[yellow]   ⏱ Cəhd {attempt+1}/{max_retries} — "
                f"{delay:.1f}s[/yellow]")
            time.sleep(delay)

    return False, f"❌ {max_retries} cəhddən sonra: {str(last_err)[:200]}"


# ══════════════════════════════════════════════════════════════════════════
# JSON PARSE
# ══════════════════════════════════════════════════════════════════════════
def parse_json_robust(text: str) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    raw = m.group(0)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    fixed = re.sub(r",(\s*[}\]])", r"\1", raw)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    fixed = fixed.replace("'", '"')
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    try:
        last = fixed.rfind("}")
        start = fixed.find("{")
        if last > start:
            return json.loads(fixed[start:last + 1])
    except json.JSONDecodeError:
        pass

    try:
        fixed = re.sub(r"(\w+)\s*:", r'"\1":', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    return None


# ══════════════════════════════════════════════════════════════════════════
# TOOLLAR
# ══════════════════════════════════════════════════════════════════════════
@tool
def read_file(path: str, offset: int = 1, limit: int = 2000) -> str:
    """Faylı oxuyur."""
    p = _safe(path)
    if not p or not p.is_file():
        return f"❌ Fayl tapılmadı: {path}"
    lock = LOCKS.acquire(str(p))
    try:
        ok, content = _read_text_safe(p)
        if not ok:
            return content
        lines = content.splitlines()
        start = max(0, offset - 1)
        chunk = lines[start:start + limit]
        return "\n".join(
            f"{start + i + 1:6d}│ {line}"
            for i, line in enumerate(chunk)) or "(boş)"
    finally:
        LOCKS.release(str(p))


@tool
def write_file(path: str, content: str) -> str:
    """Fayl yaradır (atomic)."""
    p = _safe(path)
    if not p:
        return f"❌ Təhlükəsiz olmayan yol: {path}"
    lock = LOCKS.acquire(str(p))
    try:
        if p.exists():
            SnapshotManager.create(
                f"before_write_{p.name}",
                [str(p.relative_to(get_base_dir()))])
        _atomic_write(p, content)
        return f"✅ Yazıldı: {p} ({len(content)} bayt)"
    except Exception as e:
        return f"❌ {e}"
    finally:
        LOCKS.release(str(p))


@tool
def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Faylda mətn əvəzləməsi."""
    p = _safe(path)
    if not p or not p.is_file():
        return f"❌ Fayl tapılmadı"
    lock = LOCKS.acquire(str(p))
    try:
        ok, content = _read_text_safe(p)
        if not ok:
            return content
        count = content.count(old_string)
        if count == 0:
            return "❌ `old_string` tapılmadı"
        if count > 1:
            return f"❌ `old_string` {count} dəfə təkrar"
        SnapshotManager.create(f"before_edit_{p.name}",
                             [str(p.relative_to(get_base_dir()))])
        _atomic_write(p, content.replace(old_string, new_string, 1))
        return f"✅ Düzəliş: {p}"
    except Exception as e:
        return f"❌ {e}"
    finally:
        LOCKS.release(str(p))


@tool
def list_files(directory: str = ".", pattern: str = "*") -> str:
    """Faylları siyahıla."""
    p = _safe(directory)
    if not p or not p.is_dir():
        return f"❌ Qovluq yoxdur: {directory}"
    try:
        files = sorted(p.glob(pattern))
        if not files:
            return f"❌ `{pattern}` uyğun yoxdur"
        return "\n".join(str(f.relative_to(p)) for f in files[:200])
    except Exception as e:
        return f"❌ {e}"


@tool
def tree_view(directory: str = ".", max_depth: int = 3) -> str:
    """Ağac struktur."""
    p = _safe(directory)
    if not p or not p.is_dir():
        return "❌ Qovluq yoxdur"
    SKIP = {".git", "__pycache__", "node_modules", ".venv", "venv",
            "dist", "build", ".mini_claude", ".turbo"}
    lines = [str(p) + "/"]

    def walk(d: Path, prefix: str = "", depth: int = 0):
        if depth >= max_depth:
            return
        try:
            items = sorted(
                [x for x in d.iterdir()
                 if x.name not in SKIP and not x.name.startswith(".mini")],
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


@tool
def verify_file(path: str) -> str:
    """Sintaksis yoxla."""
    p = _safe(path)
    if not p or not p.is_file():
        return f"❌ Fayl tapılmadı"
    ext = p.suffix.lower()
    try:
        ok, content = _read_text_safe(p)
        if not ok:
            return content
        if not content.strip():
            return "❌ Boş"

        if ext == ".json":
            json.loads(content)
            return "✅ JSON"
        if ext == ".py":
            try:
                compile(content, str(p), "exec")
                return "✅ Python"
            except SyntaxError as e:
                return f"❌ Python: {e.msg} (sətir {e.lineno})"
        if ext in (".ts", ".tsx", ".js", ".jsx"):
            if content.count("{") != content.count("}"):
                return "❌ Braces"
            if content.count("(") != content.count(")"):
                return "❌ Parens"
            return f"✅ {ext[1:].upper()}"
        return f"✅ {ext or 'txt'}"
    except Exception as e:
        return f"❌ {str(e)[:150]}"


@tool
def run_bash(command: str, timeout: int = 60) -> str:
    """Təhlükəsiz shell."""
    CANCEL.check()
    if not command or not isinstance(command, str):
        return "❌ Boş"

    FORBIDDEN = {";", "&&", "||", "|", ">", "<", "`", "$(", "${",
                 "&", "\\", "\n", "\r", "\x00"}
    for sym in FORBIDDEN:
        if sym in command:
            return f"❌ Qadağan: `{sym}`"

    try:
        parts = shlex.split(command)
    except ValueError as e:
        return f"❌ Parse: {e}"

    if not parts:
        return "❌ Boş"

    base = parts[0]
    SAFE = {"ls", "pwd", "cat", "head", "tail", "wc", "grep", "find",
            "file", "echo", "date", "whoami", "uname", "df", "du",
            "ps", "env", "which", "type", "stat", "tree", "sort",
            "uniq", "tr", "cut", "awk", "sed", "jq", "git", "python3",
            "pip", "pip3", "npm", "node", "pnpm", "yarn", "make",
            "cargo", "go", "rustc", "gcc", "g++", "clang", "pytest",
            "ruff", "mypy", "black"}

    if base not in SAFE:
        return f"❌ `{base}` icazəli deyil"

    DANGEROUS = {"-rf", "-fr", "-R", "--recursive",
                "--no-preserve-root", "--force"}
    for arg in parts[1:]:
        if arg in DANGEROUS:
            return f"❌ Təhlükəli flag: {arg}"

    safe_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": "en_US.UTF-8",
    }

    try:
        r = subprocess.run(
            parts, capture_output=True, text=True,
            timeout=timeout, shell=False,
            cwd=str(get_base_dir()), env=safe_env)
        out = (r.stdout or "")[:6000]
        if r.stderr:
            out += "\n[stderr]\n" + (r.stderr or "")[:2000]
        return f"```\n{out}\n```\n_(rc={r.returncode})_"
    except subprocess.TimeoutExpired:
        return f"⏱ Timeout"
    except Exception as e:
        return f"❌ {e}"


TOOLS = [read_file, write_file, edit_file, list_files,
         tree_view, verify_file, run_bash]
TOOL_MAP = {t.name: t for t in TOOLS}


# ══════════════════════════════════════════════════════════════════════════
# SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════
class SnapshotManager:
    @staticmethod
    def create(name: str, paths: List[str]) -> str:
        snap_id = f"{int(time.time())}_{os.urandom(3).hex()}"
        snap_dir = DATA_DIR / "snapshots" / snap_id
        snap_dir.mkdir(parents=True, exist_ok=True)
        for rel in paths:
            p = _safe(rel)
            if not p or not p.is_file():
                continue
            try:
                dest = snap_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
            except Exception:
                pass

        # Rotation
        snaps = sorted(
            [d for d in (DATA_DIR / "snapshots").iterdir() if d.is_dir()],
            key=lambda d: d.name)
        for old in snaps[:-SNAPSHOT_RETENTION]:
            shutil.rmtree(old, ignore_errors=True)
        return snap_id


# ══════════════════════════════════════════════════════════════════════════
# PERMISSION
# ══════════════════════════════════════════════════════════════════════════
class Permission:
    def __init__(self, mode: str = "auto"):
        self.mode = mode
        self.readonly = {"read_file", "list_files", "tree_view",
                        "verify_file"}
        self.write = {"write_file", "edit_file"}

    def check(self, name: str, args: dict) -> bool:
        if self.mode == "auto":
            return True
        if self.mode == "readonly":
            return name in self.readonly
        if self.mode == "write":
            return name in self.readonly or name in self.write
        return Confirm.ask(f"[yellow]{name}?[/yellow]", default=True)


# ══════════════════════════════════════════════════════════════════════════
# PROMPT INJECTION SANITIZE
# ══════════════════════════════════════════════════════════════════════════
def sanitize_prompt(text: str, max_len: int = 4000) -> str:
    if not text:
        return ""
    text = text[:max_len]
    injections = [
        r"ignore\s+(all\s+)?(previous|above|prior)",
        r"disregard\s+(all\s+)?(previous|above)",
        r"forget\s+(all\s+)?(previous|above)",
        r"you\s+are\s+now",
        r"new\s+instructions?:",
        r"system\s*:",
        r"<\|.*?\|>",
    ]
    for pat in injections:
        text = re.sub(pat, "[REDACTED]", text, flags=re.IGNORECASE)
    return text


# ══════════════════════════════════════════════════════════════════════════
# CONTEXT MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════
def count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def messages_tokens(messages: List) -> int:
    return sum(count_tokens(str(getattr(m, "content", "") or ""))
               for m in messages)


def trim_context(messages: List,
                max_tokens: int = MAX_CONTEXT_TOKENS) -> List:
    total = messages_tokens(messages)
    if total <= max_tokens:
        return messages

    system = [m for m in messages if isinstance(m, SystemMessage)]
    others = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(others) <= 10:
        return messages

    old = others[:-10]
    recent = others[-10:]
    summary = "[KÖHNƏ XÜLASƏ]\n" + "\n".join(
        f"- {str(m.content or '')[:100]}" for m in old[-30:])
    recent.insert(0, HumanMessage(content=summary))
    return system + recent


# ══════════════════════════════════════════════════════════════════════════
# SMART PATH EXTRACTION
# ══════════════════════════════════════════════════════════════════════════
def extract_project_path(spec: str, default: Path) -> Path:
    """
    Spec-dən qovluq yolunu çıxar.
    Nümunələr:
      "in /myproject"
      "at /home/user/proj"
      "/xskd directory"
      "qovluğunda /test"
    """
    if not spec:
        return default

    patterns = [
        r"(?:in|at|to|into|inside|daxilində|qovluğunda)\s+([/\w][/\w\-\.]*)",
        r"(/[^\s,;]+)\s+(?:directory|folder|qovluq|dir)",
        r"(?:directory|folder|qovluq|dir)[:\s]+([/\w][/\w\-\.]*)",
        r"^\s*([/\w][/\w\-\.]+)\s+(?:project|layihə|proyekt)",
    ]

    for pat in patterns:
        m = re.search(pat, spec, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            # "in /xskd" → "/xskd"
            # "in xskd" → "xskd"
            p = Path(candidate).expanduser()
            if not p.is_absolute():
                p = WORK_DIR / p
            try:
                return p.resolve()
            except Exception:
                continue

    return default


# ══════════════════════════════════════════════════════════════════════════
# CODER AGENT
# ══════════════════════════════════════════════════════════════════════════
CODER_SYSTEM = """Sən kod yazan sub-agent-sən.
TASK: {title}
AÇIQLAMA: {description}
FAYLLAR: {files}
YALNIZ bu faylları yarat. write_file istifadə et.
Hər fayl tam və işlək olsun. Qısa xülasə ver.
TOOLLAR: {tools}"""


class CoderAgent:
    def __init__(self, task: SubTask, worker_id: int,
                 permission: Permission, dry_run: bool = False,
                 base_dir: Optional[Path] = None,
                 project_state: Optional[ProjectState] = None):
        self.task = task
        self.worker_id = worker_id
        self.permission = permission
        self.dry_run = dry_run
        self.base_dir = (base_dir or get_base_dir()).resolve()
        self.project_state = project_state

    def _tools_desc(self) -> str:
        return "\n".join(
            f"• `{t.name}`: {(t.description or '').split(chr(10))[0]}"
            for t in TOOLS)

    def _execute_tools(self, ai_msg) -> List[ToolMessage]:
        results = []
        # ⚡ VACİB: bu agent üçün base-i təyin et
        old_base = get_base_dir()
        set_base_dir(self.base_dir)

        try:
            for call in getattr(ai_msg, "tool_calls", []) or []:
                name = call["name"]
                args = call.get("args", {})
                tid = call["id"]

                if not self.permission.check(name, args):
                    results.append(ToolMessage(content="❌ Rədd",
                                              tool_call_id=tid))
                    continue

                if self.dry_run:
                    console.print(f"[dim]   [dry] {name}[/dim]")
                    out = f"[DRY-RUN] {name}"
                else:
                    try:
                        out = TOOL_MAP[name].invoke(args) \
                             if name in TOOL_MAP else "❌ Tool yoxdur"
                    except Exception as e:
                        out = f"❌ {e}"

                results.append(ToolMessage(content=str(out),
                                          tool_call_id=tid))
        finally:
            set_base_dir(old_base)
        return results

    def run(self) -> SubTask:
        t0 = time.time()
        self.task.status = "running"
        self.task.worker_id = self.worker_id

        if self.project_state:
            self.project_state.update_task(self.task)

        system = CODER_SYSTEM.format(
            title=self.task.title,
            description=self.task.description,
            files=", ".join(self.task.files),
            tools=self._tools_desc())

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=(
                f"Task: {self.task.title}\n\n{self.task.description}\n\n"
                f"Fayllar:\n" +
                "\n".join(f"  • {f}" for f in self.task.files) +
                "\n\nBaşla.")),
        ]

        iteration = 0
        max_iter = 40
        empty_retries = 0

        try:
            while iteration < max_iter:
                CANCEL.check()
                iteration += 1
                messages = trim_context(messages)

                ok, ai = safe_llm_with_tools(messages, TOOLS)
                if not ok:
                    self.task.error = str(ai)[:300]
                    if self.task.retries < 2:
                        self.task.retries += 1
                        continue
                    self.task.status = "failed"
                    break

                messages.append(ai)

                if not getattr(ai, "tool_calls", None):
                    content = ai.content or ""
                    if not content.strip():
                        empty_retries += 1
                        if empty_retries >= 3:
                            self.task.result = "(boş)"
                            self.task.status = "done"
                            break
                        messages.append(HumanMessage(
                            content="Zəhmət olmasa cavab ver."))
                        continue
                    self.task.result = content
                    self.task.status = "done"
                    break

                messages.extend(self._execute_tools(ai))
            else:
                self.task.status = "done"
                self.task.result = "(iterasiya limiti)"

        except KeyboardInterrupt:
            self.task.status = "cancelled"
            self.task.error = "İstifadəçi dayandırdı"
            raise
        except Exception as e:
            self.task.status = "failed"
            self.task.error = str(e)[:300]
            log.error(f"coder_error {self.task.id} {e}")

        self.task.duration = time.time() - t0
        self.task.completed_at = datetime.now().isoformat()

        if self.project_state:
            self.project_state.update_task(self.task)

        return self.task


# ══════════════════════════════════════════════════════════════════════════
# PLANNER
# ══════════════════════════════════════════════════════════════════════════
PLANNER_SYSTEM = """Layihəni MÜSTƏQİL sub-task-lara böl.
QAYDALAR:
1. Hər task MÜSTƏQİL
2. 1-3 fayl per task
3. Eyni fayl 2 task-da olmasın
4. ÇIXIŞ yalnız JSON:
{{
  "project_name": "ad",
  "description": "1 cümlə",
  "tasks": [{{"id":"t01","title":"...","description":"...","files":["..."],"depends_on":[]}}]
}}"""


def plan_project(spec: str) -> Tuple[str, List[SubTask]]:
    console.print(Rule("[cyan]📋 Planlama[/cyan]"))
    safe_spec = sanitize_prompt(spec, max_len=3000)

    ok, response = safe_llm_invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f"Layihə:\n\n{safe_spec}"),
    ], streaming=True)

    if not ok:
        raise ValueError(f"Plan alınmadı: {response[:200]}")

    data = parse_json_robust(response)
    if not data:
        raise ValueError("JSON parse alınmadı")

    tasks_raw = data.get("tasks", [])
    if not tasks_raw:
        raise ValueError("Boş task list")
    if len(tasks_raw) > MAX_TASKS:
        console.print(f"[yellow]⚠️  {len(tasks_raw)} → ilk {MAX_TASKS}[/yellow]")
        tasks_raw = tasks_raw[:MAX_TASKS]

    valid = []
    seen_ids = set()
    for t in tasks_raw:
        tid = t.get("id", "").strip()
        files = t.get("files", [])
        if not tid or not files:
            continue
        if tid in seen_ids:
            continue
        seen_ids.add(tid)
        valid.append(t)

    if not valid:
        raise ValueError("Valid task yoxdur")

    # File conflict
    owners: Dict[str, str] = {}
    for t in valid:
        new_files = []
        for f in t["files"]:
            if f not in owners:
                owners[f] = t["id"]
                new_files.append(f)
        t["files"] = new_files

    tasks = [SubTask(
        id=t["id"], title=t["title"],
        description=t.get("description", ""),
        files=t["files"],
        depends_on=t.get("depends_on", []),
    ) for t in valid]

    console.print(f"[green]✅ {len(tasks)} task[/green]")
    return data.get("project_name", "unnamed"), tasks


# ══════════════════════════════════════════════════════════════════════════
# SWARM BUILDER
# ══════════════════════════════════════════════════════════════════════════
class SwarmBuilder:
    def __init__(self, max_workers: int = 5,
                 permission: Optional[Permission] = None,
                 dry_run: bool = False):
        self.max_workers = max(1, min(max_workers, 20))
        self.permission = permission or Permission("auto")
        self.dry_run = dry_run

    def build(self, spec: str, project_dir: Path,
             resume: bool = False):
        project_dir = project_dir.resolve()
        project_dir.mkdir(parents=True, exist_ok=True)

        # ⚡ VACİB: base-i dəyiş
        set_base_dir(project_dir)

        console.print(f"\n[bold]📁 Qovluq:[/bold] {project_dir}\n")

        # Project state
        pstate = ProjectState(project_dir)

        # Resume?
        existing_tasks = pstate.get_tasks()
        if resume and existing_tasks:
            console.print(Panel.fit(
                f"♻️  Davam edən layihə: [bold]{pstate.data.get('project_name', '?')}[/bold]\n"
                f"   Task: {sum(1 for t in existing_tasks if t.status == 'done')}/"
                f"{len(existing_tasks)} bitdi",
                border_style="yellow"))
            project_name = pstate.data.get("project_name", "resumed")
            tasks = existing_tasks
            # Failed-ləri yenidən cəhd et
            pstate.reset_failed_tasks()
            tasks = pstate.get_tasks()
        else:
            # Yeni plan
            try:
                project_name, tasks = plan_project(spec)
            except Exception as e:
                console.print(f"[red]❌ Planlama: {e}[/red]")
                set_base_dir(WORK_DIR)
                raise

            pstate.set_tasks(project_name, spec, tasks)

            # Registry
            proj_id = REGISTRY.register(project_dir, project_name, spec)
            REGISTRY.update(proj_id,
                          tasks_total=len(tasks))

        # Table
        t = Table(title=f"📋 Plan — {project_name}",
                 header_style="bold", box=box.ROUNDED)
        t.add_column("#", style="cyan", justify="right")
        t.add_column("ID", style="dim")
        t.add_column("Başlıq", style="white")
        t.add_column("Status", style="yellow")
        t.add_column("Fayllar", style="green")
        for i, task in enumerate(tasks, 1):
            icon = {"done": "✅", "failed": "❌",
                    "skipped": "⏭️"}.get(task.status, "⏳")
            t.add_row(str(i), task.id, task.title[:45],
                     icon, ", ".join(task.files)[:40])
        console.print(t)

        initial = min(3, self.max_workers)
        console.print(
            f"\n[bold]🚀 Adaptive Swarm:[/bold] {initial} → {self.max_workers}")
        console.print(f"[dim]RPM: {RATE_LIMITER.per_minute}[/dim]")

        if self.dry_run:
            console.print("[yellow]🔍 DRY RUN[/yellow]")

        # Filter pending tasks
        pending_tasks = [t for t in tasks
                        if t.status not in ("done", "skipped")]
        if not pending_tasks:
            console.print("[green]✅ Hamısı bitdi![/green]")
            self._summary(project_name, tasks, project_dir)
            return

        results: List[SubTask] = list(tasks)
        current_workers = initial
        pending: Dict = {}

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                idx = 0

                def submit_next():
                    nonlocal idx
                    while idx < len(pending_tasks):
                        task = pending_tasks[idx]
                        idx += 1
                        # Asılılıq yoxla
                        done_ids = {r.id for r in results
                                   if r.status == "done"}
                        if not all(d in done_ids
                                  for d in task.depends_on):
                            continue
                        fut = executor.submit(
                            CoderAgent(task, (idx % current_workers) + 1,
                                      self.permission, self.dry_run,
                                      base_dir=project_dir,
                                      project_state=pstate).run)
                        pending[fut] = task
                        return True
                    return False

                for _ in range(min(current_workers, len(pending_tasks))):
                    submit_next()

                while pending:
                    CANCEL.check()
                    done = []
                    try:
                        for fut in as_completed(list(pending.keys()),
                                              timeout=TASK_TIMEOUT_SEC):
                            done.append(fut)
                            break
                    except FutureTimeout:
                        for fut in list(pending.keys()):
                            if fut.done():
                                done.append(fut)

                    for fut in done:
                        task = pending.pop(fut)
                        try:
                            result = fut.result()
                            # Update result in list
                            for i, r in enumerate(results):
                                if r.id == result.id:
                                    results[i] = result
                                    break
                            icon = "✅" if result.status == "done" else "❌"
                            console.print(
                                f"{icon} [bold]{result.id}[/bold] "
                                f"({result.duration:.1f}s) — "
                                f"{result.title[:50]}")
                            if result.status == "done":
                                current_workers = min(
                                    self.max_workers, current_workers + 1)
                                REGISTRY.update(
                                    REGISTRY.find_by_path(project_dir)["id"],
                                    tasks_done=sum(1 for r in results
                                                  if r.status == "done"))
                            else:
                                current_workers = max(1, current_workers - 1)
                                REGISTRY.update(
                                    REGISTRY.find_by_path(project_dir)["id"],
                                    tasks_failed=sum(1 for r in results
                                                   if r.status == "failed"))
                        except KeyboardInterrupt:
                            raise
                        except Exception as e:
                            task.status = "failed"
                            task.error = str(e)[:200]
                            log.error(f"task_exception {task.id} {e}")
                        submit_next()

        except KeyboardInterrupt:
            console.print("\n[yellow]⏸  Dayandırıldı — davam etmək üçün "
                         f"/continue {project_dir.name}[/yellow]")

        # Final summary
        all_done = all(r.status == "done" for r in results)
        self._summary(project_name, results, project_dir)

        # Registry update
        proj = REGISTRY.find_by_path(project_dir)
        if proj:
            REGISTRY.update(
                proj["id"],
                status="completed" if all_done else "in_progress",
                tasks_done=sum(1 for r in results if r.status == "done"),
                tasks_failed=sum(1 for r in results if r.status == "failed"))

        # Base qalsın — interaktiv davam
        # set_base_dir(WORK_DIR)  # İstəsən geri qaytar

    def _summary(self, name: str, tasks: List[SubTask],
                project_dir: Path):
        console.print()
        console.print(Rule("[bold cyan]🎉 Yekun[/bold cyan]"))

        done = sum(1 for t in tasks if t.status == "done")
        failed = sum(1 for t in tasks if t.status == "failed")
        pending = sum(1 for t in tasks
                     if t.status not in ("done", "failed", "skipped"))
        total_time = sum(t.duration for t in tasks)
        wall = max((t.duration for t in tasks), default=0)

        t = Table(box=box.SIMPLE, show_header=False)
        t.add_column("", style="cyan")
        t.add_column("", style="white")
        t.add_row("Layihə", name)
        t.add_row("Qovluq", str(project_dir))
        t.add_row("Task", f"[green]{done}[/green] ok, "
                         f"[red]{failed}[/red] fail, "
                         f"[yellow]{pending}[/yellow] pending")
        t.add_row("Ümumi vaxt", f"{total_time:.1f}s")
        t.add_row("Wall", f"{wall:.1f}s")
        t.add_row("Speedup", f"{total_time/max(1, wall):.1f}x")
        console.print(t)

        console.print()
        console.print(Markdown(
            TOOL_MAP["tree_view"].invoke(
                {"directory": str(project_dir), "max_depth": 3})))


# ══════════════════════════════════════════════════════════════════════════
# CHAT AGENT
# ══════════════════════════════════════════════════════════════════════════
CHAT_SYSTEM = """Sən Mini-Claude v7 agentsən. Azərbaycan dilində cavab ver.
Markdown istifadə et. Alətlər: {tools}"""


class ChatAgent:
    def __init__(self, permission: Permission):
        self.permission = permission
        self.messages: List = []

    def _tools_desc(self) -> str:
        return "\n".join(
            f"• `{t.name}`: {(t.description or '').split(chr(10))[0]}"
            for t in TOOLS)

    def _execute_tools(self, ai_msg) -> List[ToolMessage]:
        results = []
        for call in getattr(ai_msg, "tool_calls", []) or []:
            name = call["name"]
            args = call.get("args", {})
            tid = call["id"]

            if not self.permission.check(name, args):
                results.append(ToolMessage(content="❌",
                                          tool_call_id=tid))
                continue

            try:
                out = TOOL_MAP[name].invoke(args) \
                     if name in TOOL_MAP else "❌ Tool yoxdur"
            except Exception as e:
                out = f"❌ {e}"

            results.append(ToolMessage(content=str(out),
                                      tool_call_id=tid))
        return results

    def ask(self, user_input: str) -> str:
        CANCEL.check()
        if not self.messages:
            self.messages.append(SystemMessage(
                content=CHAT_SYSTEM.format(tools=self._tools_desc())))

        self.messages.append(HumanMessage(content=user_input))

        for _ in range(30):
            CANCEL.check()
            self.messages = trim_context(self.messages)

            ok, ai = safe_llm_with_tools(self.messages, TOOLS)
            if not ok:
                return f"❌ {ai}"

            self.messages.append(ai)
            if not getattr(ai, "tool_calls", None):
                return ai.content or "(boş)"
            self.messages.extend(self._execute_tools(ai))
        return "⚠️ Limit"

    def reset(self):
        self.messages = []


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════
BANNER = r"""
[bold cyan]
  ███╗   ███╗██╗███╗   ██╗██╗     ██████╗██╗      █████╗ ██╗   ██╗██████╗ ███████╗
  ████╗ ████║██║████╗  ██║██║    ██╔════╝██║     ██╔══██╗██║   ██║██╔══██╗██╔════╝
  ██╔████╔██║██║██╔██╗ ██║██║    ██║     ██║     ███████║██║   ██║██║  ██║█████╗  
  ██║╚██╔╝██║██║██║╚██╗██║██║    ██║     ██║     ██╔══██║██║   ██║██║  ██║██╔══╝  
  ██║ ╚═╝ ██║██║██║ ╚████║██║    ╚██████╗███████╗██║  ██║╚██████╔╝██████╔╝███████╗
  ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝     ╚═════╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝
[/bold cyan]
[bold yellow]  v7.0 — Full Production + Project Resume[/bold yellow]
"""

HELP = """
[bold cyan]⚡ Əsas:[/bold cyan]
  [green]/build <təsvir>[/green]     — Swarm build (spec-dən /path çıxarır)
  [green]/continue <ad>[/green]      — Yarımçıq layihəni davam etdir
  [green]/projects[/green]           — Bütün layihələr
  [green]/swarm <N>[/green]          — Paralel (1-20)
  [green]/rpm <N>[/green]            — Rate limit
  [green]/dry <on|off>[/green]       — Dry-run
  [green]/quick <sual>[/green]       — Sürətli cavab

[bold cyan]📋 Digər:[/bold cyan]
  [green]/help[/green] / [green]/tools[/green] / [green]/tree[/green]
  [green]/mode auto|manual|readonly|write[/green]
  [green]/reset[/green] / [green]/clear[/green] / [green]/exit[/green]

[bold cyan]💡 Nümunələr:[/bold cyan]
  • /build FastAPI app in /myproject
  • /build Visual Programming in /visul
  • /continue myproject
  • /projects
"""

STYLE = Style.from_dict({"prompt": "bold #00d4ff"})


def cmd_projects():
    """Layihələri göstər."""
    projects = REGISTRY.all()
    if not projects:
        console.print("[dim]Layihə yoxdur[/dim]")
        return
    t = Table(title=f"📁 Layihələr ({len(projects)})",
             box=box.ROUNDED, header_style="bold")
    t.add_column("Ad", style="cyan")
    t.add_column("Status", style="yellow")
    t.add_column("Task", style="green")
    t.add_column("Qovluq", style="dim")
    for p in sorted(projects, key=lambda x: x.get("updated", ""),
                   reverse=True):
        status_icon = {"completed": "✅", "in_progress": "🔄",
                      "failed": "❌"}.get(p["status"], "❓")
        t.add_row(
            p["name"][:30],
            f"{status_icon} {p['status']}",
            f"{p['tasks_done']}/{p['tasks_total']}",
            p["path"])
    console.print(t)


def interactive():
    console.print(BANNER)
    console.print(Panel(HELP, border_style="blue"))

    permission = Permission("auto")
    chat = ChatAgent(permission)
    max_workers = 5
    dry_run = False

    console.print(
        f"[bold]Model:[/bold] [green]{MODEL}[/green]  |  "
        f"[bold]RPM:[/bold] [yellow]{RATE_LIMITER.per_minute}[/yellow]  |  "
        f"[bold]Swarm:[/bold] [cyan]{max_workers}[/cyan]\n")

    session = PromptSession(
        history=FileHistory(str(HISTORY_FILE)),
        style=STYLE)

    while True:
        try:
            user_input = session.prompt("\n💬 Sən: ").strip()
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
                console.print("👋")
                break

            elif cmd == "/help":
                console.print(Panel(HELP, border_style="blue"))

            elif cmd == "/tools":
                t = Table(title=f"🔧 Tool", box=box.ROUNDED)
                t.add_column("#", style="cyan")
                t.add_column("Ad", style="green")
                t.add_column("Təsvir", style="dim")
                for i, tl in enumerate(TOOLS, 1):
                    t.add_row(str(i), tl.name,
                             (tl.description or "")[:60])
                console.print(t)

            elif cmd == "/tree":
                console.print(Markdown(
                    TOOL_MAP["tree_view"].invoke(
                        {"directory": ".", "max_depth": 3})))

            elif cmd == "/reset":
                chat.reset()
                console.print("[green]✅[/green]")

            elif cmd == "/mode":
                if arg in ("auto", "manual", "readonly", "write"):
                    permission.mode = arg
                    console.print(f"[green]✅ {arg}[/green]")

            elif cmd == "/swarm":
                if arg.isdigit():
                    max_workers = max(1, min(int(arg), 20))
                    console.print(f"[green]✅ Swarm: {max_workers}[/green]")

            elif cmd == "/rpm":
                if arg.isdigit():
                    RATE_LIMITER._per_minute = max(5, min(int(arg), 120))
                    RATE_LIMITER.rate = RATE_LIMITER._per_minute / 60.0
                    console.print(f"[green]✅ RPM: {RATE_LIMITER.per_minute}[/green]")

            elif cmd == "/dry":
                if arg in ("on", "off"):
                    dry_run = (arg == "on")
                    console.print(f"[green]✅ Dry: {dry_run}[/green]")

            elif cmd == "/projects":
                cmd_projects()

            elif cmd == "/continue":
                if not arg:
                    # İn-progress olanları göstər
                    in_prog = REGISTRY.in_progress()
                    if not in_prog:
                        console.print("[dim]Yarımçıq layihə yoxdur[/dim]")
                        continue
                    if len(in_prog) == 1:
                        project = in_prog[0]
                    else:
                        cmd_projects()
                        console.print("\n[yellow]Layihə adı daxil et:[/yellow]")
                        arg = Prompt.ask("Ad")
                        project = REGISTRY.find_by_name(arg) or \
                                 REGISTRY.find_by_path(Path(arg))
                        if not project:
                            console.print("[red]❌ Tapılmadı[/red]")
                            continue
                else:
                    project = REGISTRY.find_by_name(arg) or \
                             REGISTRY.find_by_path(Path(arg))
                    if not project:
                        console.print(f"[red]❌ Layihə tapılmadı: {arg}[/red]")
                        continue

                project_dir = Path(project["path"])
                if not project_dir.exists():
                    console.print(f"[red]❌ Qovluq yoxdur: {project_dir}[/red]")
                    continue

                console.print(f"\n[bold]♻️  Davam:[/bold] {project['name']}")
                console.print(f"[dim]{project_dir}[/dim]\n")

                try:
                    builder = SwarmBuilder(
                        max_workers=max_workers,
                        permission=permission,
                        dry_run=dry_run)
                    builder.build(
                        project["spec"],
                        project_dir,
                        resume=True)
                except KeyboardInterrupt:
                    console.print("\n[yellow]⏸[/yellow]")
                except Exception as e:
                    console.print(f"[red]❌ {e}[/red]")
                    log.exception("continue_error")

            elif cmd == "/quick":
                if not arg:
                    console.print("[yellow]/quick <sual>[/yellow]")
                    continue
                console.print()
                ok, ans = safe_llm_invoke([
                    SystemMessage(content="Qısa cavab. Azərbaycan dilində."),
                    HumanMessage(content=arg),
                ], streaming=True)
                if ok:
                    console.print(Markdown(ans))
                else:
                    console.print(f"[red]{ans}[/red]")

            elif cmd == "/build":
                if not arg:
                    console.print("[yellow]/build <təsvir>[/yellow]")
                    continue

                # ⚡ Spec-dən path çıxar
                default_name = re.sub(r"[^\w\-]", "_",
                                    arg[:30].lower()) or "project"
                default_dir = WORK_DIR / default_name
                project_dir = extract_project_path(arg, default_dir)

                console.print(f"\n[bold]📁 Layihə qovluğu:[/bold] {project_dir}")
                if project_dir != default_dir:
                    console.print(
                        f"[dim](spec-dən çıxarıldı)[/dim]")

                # Confirm if outside WORK_DIR
                try:
                    project_dir.relative_to(WORK_DIR)
                except ValueError:
                    if not Confirm.ask(
                        f"[yellow]⚠️  {project_dir} WORK_DIR xaricindədir. "
                        f"Davam?[/yellow]", default=False):
                        continue

                project_dir.mkdir(parents=True, exist_ok=True)

                try:
                    builder = SwarmBuilder(
                        max_workers=max_workers,
                        permission=permission,
                        dry_run=dry_run)
                    builder.build(arg, project_dir, resume=False)
                except KeyboardInterrupt:
                    console.print("\n[yellow]⏸  "
                                 f"Davam: /continue {project_dir.name}[/yellow]")
                except Exception as e:
                    console.print(f"[red]❌ {e}[/red]")
                    log.exception("build_error")
            else:
                console.print("[red]❌[/red]")
            continue

        # Chat
        console.print()
        try:
            result = chat.ask(user_input)
            if result:
                console.print("\n[bold cyan]🤖[/bold cyan]")
                console.print(Markdown(result))
        except KeyboardInterrupt:
            console.print("\n[yellow]⏸[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ {e}[/red]")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="auto")
    ap.add_argument("--build", "-b")
    ap.add_argument("--continue-project", "-c")
    ap.add_argument("--task", "-t")
    ap.add_argument("--swarm", "-s", type=int, default=5)
    ap.add_argument("--rpm", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", "-o")
    args = ap.parse_args()

    _install_signal_handler()
    RATE_LIMITER._per_minute = args.rpm
    RATE_LIMITER.rate = args.rpm / 60.0
    permission = Permission(args.mode)

    # Continue
    if args.continue_project:
        proj = REGISTRY.find_by_name(args.continue_project) or \
               REGISTRY.find_by_path(Path(args.continue_project))
        if not proj:
            console.print(f"[red]❌ Layihə tapılmadı[/red]")
            sys.exit(1)
        project_dir = Path(proj["path"])
        cp_lock = CrossProcessLock(LOCK_FILE)
        if not cp_lock.acquire():
            console.print("[red]❌ Başqa instans işləyir[/red]")
            sys.exit(1)
        try:
            builder = SwarmBuilder(
                max_workers=args.swarm,
                permission=permission,
                dry_run=args.dry_run)
            builder.build(proj["spec"], project_dir, resume=True)
        except KeyboardInterrupt:
            console.print("\n⏸")
        finally:
            cp_lock.release()
        return

    # Build
    if args.build:
        console.print(BANNER)
        if args.out:
            project_dir = Path(args.out).expanduser().resolve()
        else:
            default_name = re.sub(r"[^\w\-]", "_",
                                args.build[:30].lower()) or "project"
            default_dir = WORK_DIR / default_name
            project_dir = extract_project_path(args.build, default_dir)

        project_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold]📁 Qovluq:[/bold] {project_dir}\n")

        cp_lock = CrossProcessLock(LOCK_FILE)
        if not cp_lock.acquire():
            console.print("[red]❌ Başqa instans[/red]")
            sys.exit(1)
        try:
            builder = SwarmBuilder(
                max_workers=args.swarm,
                permission=permission,
                dry_run=args.dry_run)
            builder.build(args.build, project_dir, resume=False)
        except KeyboardInterrupt:
            console.print("\n⏸")
        finally:
            cp_lock.release()
        return

    if args.task:
        console.print(BANNER)
        chat = ChatAgent(permission)
        r = chat.ask(args.task)
        if r:
            console.print(Markdown(r))
        return

    interactive()


if __name__ == "__main__":
    main()
