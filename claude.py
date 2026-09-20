#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  MINI-CLAUDE v2.0 — Paralel Sub-Agent Sistemi                            ║
║                                                                          ║
║  Xüsusiyyətlər:                                                          ║
║    • 🚀 Paralel sub-agent-lər (ThreadPoolExecutor)                       ║
║    • 🔒 Fayl kilidləmə — toqquşma olmur                                  ║
║    • ♾️  Limitsiz iterasiya + retry                                       ║
║    • 🎯 Agentic loop (plan → execute → verify)                           ║
║    • 📦 Auto-export (ZIP)                                                ║
║    • 💬 Chat + Build rejimləri                                           ║
║                                                                          ║
║  İstifadə:                                                               ║
║    python mini_claude_v2.py                                              ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import re
import sys
import ast
import json
import time
import shutil
import zipfile
import logging
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import tool

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.rule import Rule
from rich.align import Align
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

# ══════════════════════════════════════════════════════════════════════════
# KONFİQURASİYA
# ══════════════════════════════════════════════════════════════════════════
load_dotenv()

console = Console()

API_KEY  = os.getenv("NVIDIA_API_KEY")
BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL    = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")

if not API_KEY:
    console.print(Panel(
        "[red]❌ NVIDIA_API_KEY tapılmadı![/red]\n\n"
        "[yellow]`.env` faylına əlavə et:[/yellow]\n"
        "   [dim]NVIDIA_API_KEY=nvapi-xxxx[/dim]",
        title="⚠️  Konfiqurasiya", border_style="red"))
    sys.exit(1)

ROOT_DIR = Path.cwd()
DATA_DIR = ROOT_DIR / ".mini_claude"
DATA_DIR.mkdir(exist_ok=True)
HISTORY_FILE = DATA_DIR / "history.txt"

WORK_DIR = ROOT_DIR.resolve()

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("mini")


# ══════════════════════════════════════════════════════════════════════════
# FAYL KİLİDLƏMƏ — paralel yazmadan qorunma
# ══════════════════════════════════════════════════════════════════════════
class FileLockManager:
    """
    Fayl səviyyəsində kilid. Eyni fayl 2 agent tərəfindən
    eyni vaxtda yazılmır.
    """
    def __init__(self):
        self._locks: Dict[str, threading.Lock] = {}
        self._global = threading.Lock()

    def get(self, path: str) -> threading.Lock:
        with self._global:
            if path not in self._locks:
                self._locks[path] = threading.Lock()
            return self._locks[path]

    def acquire(self, path: str) -> threading.Lock:
        lock = self.get(path)
        lock.acquire()
        return lock

    def release(self, path: str):
        lock = self.get(path)
        try:
            lock.release()
        except RuntimeError:
            pass


LOCKS = FileLockManager()


# ══════════════════════════════════════════════════════════════════════════
# YOL TƏHLÜKƏSİZLİYİ
# ══════════════════════════════════════════════════════════════════════════
def _safe(path: str) -> Optional[Path]:
    try:
        p = Path(path)
        if not p.is_absolute():
            p = WORK_DIR / p
        p = p.resolve()
        if str(p).startswith(str(WORK_DIR)):
            return p
        return None
    except Exception:
        return None


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


# ══════════════════════════════════════════════════════════════════════════
# ALƏTLƏR (thread-safe)
# ══════════════════════════════════════════════════════════════════════════

@tool
def read_file(path: str, offset: int = 1, limit: int = 2000) -> str:
    """Faylı oxuyur. Sətir nömrələri ilə."""
    p = _safe(path)
    if not p or not p.is_file():
        return f"❌ Fayl tapılmadı: {path}"
    lock = LOCKS.acquire(str(p))
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(0, offset - 1)
        chunk = lines[start:start + limit]
        return "\n".join(
            f"{start + i + 1:6d}│ {line}" for i, line in enumerate(chunk)
        ) or "(boş fayl)"
    except Exception as e:
        return f"❌ Oxuma xətası: {e}"
    finally:
        LOCKS.release(str(p))


@tool
def write_file(path: str, content: str) -> str:
    """Fayl yaradır və ya tamamilə əvəz edir."""
    p = _safe(path)
    if not p:
        return f"❌ Təhlükəsiz olmayan yol: {path}"
    lock = LOCKS.acquire(str(p))
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"✅ Yazıldı: {p.relative_to(WORK_DIR)} ({len(content)} bayt)"
    except Exception as e:
        return f"❌ Yazma xətası: {e}"
    finally:
        LOCKS.release(str(p))


@tool
def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Faylda dəqiq mətn əvəzləməsi."""
    p = _safe(path)
    if not p or not p.is_file():
        return f"❌ Fayl tapılmadı: {path}"
    lock = LOCKS.acquire(str(p))
    try:
        content = p.read_text(encoding="utf-8")
        count = content.count(old_string)
        if count == 0:
            return "❌ `old_string` tapılmadı."
        if count > 1:
            return f"❌ `old_string` {count} dəfə təkrarlanır."
        backup = DATA_DIR / f"{p.name}.bak"
        backup.write_text(content, encoding="utf-8")
        p.write_text(content.replace(old_string, new_string, 1),
                    encoding="utf-8")
        return f"✅ Düzəliş: {p.relative_to(WORK_DIR)}"
    except Exception as e:
        return f"❌ {e}"
    finally:
        LOCKS.release(str(p))


@tool
def list_files(directory: str = ".", pattern: str = "*") -> str:
    """Qovluqdakı faylları siyahılayır."""
    p = _safe(directory)
    if not p or not p.is_dir():
        return f"❌ Qovluq tapılmadı: {directory}"
    try:
        files = sorted(p.glob(pattern))
        if not files:
            return f"❌ `{pattern}` uyğun fayl yoxdur."
        return "\n".join(str(f.relative_to(p)) for f in files[:200])
    except Exception as e:
        return f"❌ {e}"


@tool
def tree_view(directory: str = ".", max_depth: int = 3) -> str:
    """Qovluq strukturunu ağac kimi göstərir."""
    p = _safe(directory)
    if not p or not p.is_dir():
        return f"❌ Qovluq tapılmadı: {directory}"
    SKIP = {".git", "__pycache__", "node_modules", ".venv", "venv",
            "dist", "build", ".mini_claude", ".turbo"}
    lines = [str(p.relative_to(WORK_DIR)) + "/"]

    def walk(d: Path, prefix: str = "", depth: int = 0):
        if depth >= max_depth:
            return
        try:
            items = sorted(
                [x for x in d.iterdir() if x.name not in SKIP],
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
def grep_search(pattern: str, path: str = ".", max_results: int = 50) -> str:
    """Fayl məzmununda regex axtarışı."""
    p = _safe(path)
    if not p:
        return f"❌ Təhlükəsiz olmayan yol: {path}"
    try:
        cmd = (["rg", "-n", "--no-heading", pattern, str(p)]
               if _have("rg") else
               ["grep", "-rn", "-E", pattern, str(p)])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        out = r.stdout or ""
        if not out.strip():
            return "🔍 Nəticə yoxdur."
        return "\n".join(out.splitlines()[:max_results])
    except Exception as e:
        return f"❌ {e}"


@tool
def run_bash(command: str, timeout: int = 60) -> str:
    """Təhlükəsiz shell əmri (whitelist)."""
    SAFE = {
        "ls", "pwd", "cat", "head", "tail", "wc", "grep", "find", "file",
        "echo", "date", "whoami", "uname", "df", "du", "ps", "env",
        "which", "type", "stat", "tree", "sort", "uniq", "tr", "cut",
        "awk", "sed", "jq", "git", "python3", "pip", "pip3", "npm",
        "node", "pnpm", "yarn", "make", "cargo", "go",
    }
    base = command.strip().split()[0] if command.strip() else ""
    if base not in SAFE:
        return f"❌ `{base}` icazəli deyil."
    try:
        r = subprocess.run(["bash", "-c", command], capture_output=True,
                          text=True, timeout=timeout)
        out = r.stdout or ""
        if r.stderr:
            out += ("\n" if out else "") + r.stderr
        if len(out) > 6000:
            out = out[:6000] + f"\n... [{len(out)-6000} bayt kəsildi]"
        return f"```\n{out}\n```\n_(rc={r.returncode})_"
    except subprocess.TimeoutExpired:
        return f"⏱ Timeout ({timeout}s)"
    except Exception as e:
        return f"❌ {e}"


@tool
def verify_file(path: str) -> str:
    """Faylın sintaksisini yoxla."""
    p = _safe(path)
    if not p or not p.is_file():
        return f"❌ Fayl tapılmadı: {path}"
    ext = p.suffix.lower()
    try:
        content = p.read_text(encoding="utf-8")
        if not content.strip():
            return "❌ Boş fayl"
        if ext == ".json":
            json.loads(content); return "✅ JSON OK"
        if ext in (".yaml", ".yml"):
            try:
                import yaml; yaml.safe_load(content); return "✅ YAML OK"
            except ImportError:
                return "✅ YAML (skip)"
        if ext == ".py":
            ast.parse(content); return "✅ Python OK"
        if ext in (".ts", ".tsx", ".js", ".jsx"):
            if content.count("{") != content.count("}"):
                return f"❌ Braces {content.count('{')}/{content.count('}')}"
            if content.count("(") != content.count(")"):
                return "❌ Parens balanssız"
            return f"✅ {ext[1:].upper()} OK"
        return f"✅ {ext or 'txt'} OK"
    except SyntaxError as e:
        return f"❌ Syntax: {e.msg} (sətir {e.lineno})"
    except Exception as e:
        return f"❌ {str(e)[:150]}"


@tool
def export_zip(directory: str = ".", output: str = "") -> str:
    """Qovluğu ZIP-ə ixrac et."""
    p = _safe(directory)
    if not p or not p.is_dir():
        return f"❌ Qovluq tapılmadı: {directory}"
    out_path = Path(output).expanduser() if output else \
               (WORK_DIR / f"{p.name}.zip")
    try:
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in p.rglob("*"):
                if f.is_file() and ".mini_claude" not in str(f):
                    z.write(f, f.relative_to(p))
        size_mb = out_path.stat().st_size / 1024 / 1024
        return f"✅ Export: {out_path} ({size_mb:.2f} MB)"
    except Exception as e:
        return f"❌ {e}"


TOOLS = [
    read_file, write_file, edit_file, list_files, tree_view,
    grep_search, run_bash, verify_file, export_zip,
]
TOOL_MAP = {t.name: t for t in TOOLS}


# ══════════════════════════════════════════════════════════════════════════
# İCAZƏ
# ══════════════════════════════════════════════════════════════════════════
class Permission:
    def __init__(self, mode: str = "auto"):
        self.mode = mode
        self.readonly = {"read_file", "list_files", "tree_view",
                        "grep_search", "verify_file"}
        self.write = {"write_file", "edit_file", "export_zip"}

    def check(self, name: str, args: dict) -> bool:
        if self.mode == "auto":
            return True
        if self.mode == "readonly":
            return name in self.readonly
        if self.mode == "write":
            return name in self.readonly or name in self.write
        console.print(f"\n[yellow]🔧 {name}[/yellow]")
        for k, v in args.items():
            s = str(v)[:100]
            console.print(f"   [dim]{k}: {s}[/dim]")
        return Confirm.ask("[yellow]İcazə?[/yellow]", default=True)


# ══════════════════════════════════════════════════════════════════════════
# LLM WRAPPER
# ══════════════════════════════════════════════════════════════════════════
def make_llm():
    return ChatOpenAI(
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        model=MODEL,
        temperature=0.2,
        max_tokens=8192,
        timeout=180,
        max_retries=3,
    )


# ══════════════════════════════════════════════════════════════════════════
# TASK MODELİ
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class SubTask:
    id: str
    title: str
    description: str
    files: List[str] = field(default_factory=list)
    status: str = "pending"
    result: str = ""
    error: str = ""
    duration: float = 0.0
    worker_id: int = -1


# ══════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════
CHAT_SYSTEM = """Sən Mini-Claude adlı köməkçi AI agentsən.

PRİNSİPLƏR:
• Azərbaycan dilində cavab ver.
• Markdown istifadə et (başlıq, cədvəl, kod).
• Yalnız təsdiqlənmiş məlumat ver.
• Lazım gəlsə alətlərdən istifadə et.

MÖVCUD ALƏTLƏR:
{tools}
"""

PLANNER_SYSTEM = """Sən layihə planlayıcısı-san.

İstifadəçinin təsvirini MÜSTƏQİL, PARALEL İCRA OLUNA BİLƏN sub-task-lara böl.

QAYDALAR:
1. Hər task MÜSTƏQİL olmalıdır (başqa task-dan asılı DEYİL)
2. Hər task 1-3 fayl yaradır
3. Eyni fayla iki task toxunmasın (toqquşma olmasın!)
4. 3-8 task arası optimal
5. Hamısı eyni vaxtda işləyə bilər

ÇIXIŞ (yalnız JSON):
{{
  "project_name": "ad",
  "description": "1 cümlə",
  "tasks": [
    {{
      "id": "t01",
      "title": "Qısa başlıq",
      "description": "Nə ediləcək (2-3 cümlə)",
      "files": ["path/file.ext"]
    }}
  ]
}}
"""

WORKER_SYSTEM = """Sən kod yazan sub-agent-sən.

Sənin task-ın: {title}
Açıqlama: {description}
Yaratmalı olduğun fayllar: {files}

VACİB:
• YALNIZ bu faylları yarat — başqalarına toxunma
• Hər fayl TAM işlək olsun (yarımçıq yox)
• `write_file` alətini istifadə et
• Hər fayl yazdıqdan sonra `verify_file` ilə yoxla
• Bitirdikdən sonra qısa xülasə ver

MÖVCUD ALƏTLƏR:
{tools}
"""


# ══════════════════════════════════════════════════════════════════════════
# SUB-AGENT
# ══════════════════════════════════════════════════════════════════════════
class SubAgent:
    def __init__(self, task: SubTask, worker_id: int,
                 permission: Permission):
        self.task = task
        self.worker_id = worker_id
        self.permission = permission
        self.llm = make_llm()
        self.llm_tools = self.llm.bind_tools(TOOLS)
        self.messages: List = []

    def _tools_desc(self) -> str:
        return "\n".join(
            f"• `{t.name}`: {(t.description or '').split(chr(10))[0]}"
            for t in TOOLS
        )

    def _execute_tools(self, ai_msg) -> List[ToolMessage]:
        results = []
        for call in getattr(ai_msg, "tool_calls", []) or []:
            name = call["name"]
            args = call.get("args", {})
            tid = call["id"]

            console.print(
                f"[dim]   [W{self.worker_id}] 🔧 {name}"
                f"({str(args)[:60]})[/dim]")

            if not self.permission.check(name, args):
                results.append(ToolMessage(
                    content="❌ İstifadəçi rədd etdi",
                    tool_call_id=tid))
                continue

            try:
                out = TOOL_MAP[name].invoke(args) if name in TOOL_MAP \
                     else f"❌ Alət yoxdur: {name}"
            except Exception as e:
                out = f"❌ {e}"

            preview = str(out)[:150].replace("\n", " ")
            console.print(f"[dim]   [W{self.worker_id}] → {preview}[/dim]")

            results.append(ToolMessage(content=str(out), tool_call_id=tid))
        return results

    def run(self) -> SubTask:
        """Sub-task-ı icra et. Limitsiz iterasiya."""
        t0 = time.time()
        self.task.status = "running"
        self.task.worker_id = self.worker_id

        system = WORKER_SYSTEM.format(
            title=self.task.title,
            description=self.task.description,
            files=", ".join(self.task.files),
            tools=self._tools_desc(),
        )
        self.messages.append(SystemMessage(content=system))

        user_msg = (
            f"Task: {self.task.title}\n\n"
            f"{self.task.description}\n\n"
            f"Yaratmalı olduğun fayllar:\n"
            + "\n".join(f"  • {f}" for f in self.task.files)
            + "\n\nBaşla."
        )
        self.messages.append(HumanMessage(content=user_msg))

        # Limitsiz iterasiya (praktik limit: 100)
        iteration = 0
        max_iter = 100

        try:
            while iteration < max_iter:
                iteration += 1
                try:
                    ai = self.llm_tools.invoke(self.messages)
                except Exception as e:
                    console.print(
                        f"[yellow]   [W{self.worker_id}] ⚠️ LLM xətası: "
                        f"{str(e)[:100]}[/yellow]")
                    time.sleep(2)
                    continue

                self.messages.append(ai)

                if not getattr(ai, "tool_calls", None):
                    # Son cavab
                    self.task.result = ai.content or "(boş)"
                    self.task.status = "done"
                    break

                tool_results = self._execute_tools(ai)
                self.messages.extend(tool_results)
            else:
                self.task.status = "done"
                self.task.result = "(iterasiya limiti çatdı)"

        except Exception as e:
            self.task.status = "failed"
            self.task.error = str(e)[:300]

        self.task.duration = time.time() - t0
        return self.task


# ══════════════════════════════════════════════════════════════════════════
# PARALLEL BUILDER
# ══════════════════════════════════════════════════════════════════════════
class ParallelBuilder:
    def __init__(self, max_workers: int = 4, permission: Optional[Permission] = None):
        self.max_workers = max_workers
        self.permission = permission or Permission("auto")
        self.llm = make_llm()

    # ─────── plan qur ───────
    def plan(self, spec: str) -> Tuple[str, List[SubTask]]:
        console.print(Rule("[cyan]📋 Planlama[/cyan]"))

        with console.status("[yellow]Plan qurulur...[/yellow]", spinner="dots"):
            system = PLANNER_SYSTEM
            user = f"Layihə:\n\n{spec[:4000]}"
            response = self.llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user),
            ]).content or ""

        m = re.search(r"\{.*\}", response, re.DOTALL)
        if not m:
            raise ValueError(f"JSON tapılmadı:\n{response[:500]}")

        data = json.loads(m.group(0))
        tasks = [SubTask(
            id=t["id"],
            title=t["title"],
            description=t.get("description", ""),
            files=t.get("files", []),
        ) for t in data["tasks"]]

        # Toqquşma yoxlaması — eyni fayla iki task toxunmasın
        file_owners: Dict[str, str] = {}
        for t in tasks:
            for f in t.files:
                if f in file_owners:
                    console.print(
                        f"[yellow]⚠️  Toqquşma: `{f}` "
                        f"({file_owners[f]} və {t.id})[/yellow]")
                    # İkinci task-dan faylı sil
                    t.files = [x for x in t.files if x != f]
                else:
                    file_owners[f] = t.id

        console.print(f"[green]✅ {len(tasks)} sub-task planlandı[/green]")
        return data.get("project_name", "unnamed"), tasks

    # ─────── paralel icra ───────
    def build(self, spec: str, project_dir: Path) -> Tuple[str, List[SubTask]]:
        project_name, tasks = self.plan(spec)

        # Plan cədvəli
        console.print()
        t = Table(title=f"📋 Sub-Tasks — {project_name}",
                  header_style="bold", box=box.ROUNDED)
        t.add_column("#", style="cyan", justify="right")
        t.add_column("ID", style="dim")
        t.add_column("Başlıq", style="white")
        t.add_column("Fayllar", style="green")
        for i, task in enumerate(tasks, 1):
            t.add_row(str(i), task.id, task.title[:50],
                     ", ".join(task.files)[:50])
        console.print(t)

        console.print(f"\n[bold]🚀 Paralel icra:[/bold] "
                     f"{len(tasks)} sub-agent, "
                     f"{self.max_workers} eyni vaxtda\n")

        # Paralel icra
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    SubAgent(task, i + 1, self.permission).run
                ): task
                for i, task in enumerate(tasks)
            }

            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    icon = "✅" if result.status == "done" else "❌"
                    console.print(
                        f"\n{icon} [bold]{result.id}[/bold] "
                        f"({result.duration:.1f}s) — {result.title[:50]}")
                    if result.error:
                        console.print(f"   [red]{result.error[:150]}[/red]")
                except Exception as e:
                    task.status = "failed"
                    task.error = str(e)
                    console.print(f"\n❌ [bold]{task.id}[/bold]: {e}")
                    results.append(task)

        # Yekun
        self._summary(project_name, results, project_dir)
        return project_name, results

    def _summary(self, name: str, tasks: List[SubTask], project_dir: Path):
        console.print()
        console.print(Rule("[bold cyan]🎉 Yekun[/bold cyan]"))

        done = sum(1 for t in tasks if t.status == "done")
        failed = sum(1 for t in tasks if t.status == "failed")
        total_time = sum(t.duration for t in tasks)

        t = Table(box=box.SIMPLE, show_header=False)
        t.add_column("", style="cyan")
        t.add_column("", style="white")
        t.add_row("Layihə", name)
        t.add_row("Sub-task", f"[green]{done}[/green]/"
                             f"{len(tasks)} uğurlu, "
                             f"[red]{failed}[/red] uğursuz")
        t.add_row("Ümumi iş vaxtı", f"{total_time:.1f}s")
        t.add_row("Wall time", "paralel")
        t.add_row("Qovluq", str(project_dir))
        console.print(t)

        # Strukturu göstər
        console.print()
        console.print(Markdown(TOOL_MAP["tree_view"].invoke(
            {"directory": ".", "max_depth": 3})))


# ══════════════════════════════════════════════════════════════════════════
# CHAT AGENT (Claude-Lite üslubu)
# ══════════════════════════════════════════════════════════════════════════
class ChatAgent:
    def __init__(self, permission: Permission):
        self.llm = make_llm()
        self.llm_tools = self.llm.bind_tools(TOOLS)
        self.permission = permission
        self.messages: List = []

    def _tools_desc(self) -> str:
        return "\n".join(
            f"• `{t.name}`: {(t.description or '').split(chr(10))[0]}"
            for t in TOOLS
        )

    def _execute_tools(self, ai_msg) -> List[ToolMessage]:
        results = []
        for call in getattr(ai_msg, "tool_calls", []) or []:
            name = call["name"]
            args = call.get("args", {})
            tid = call["id"]

            console.print(f"[dim]   🔧 {name}({str(args)[:80]})[/dim]")

            if not self.permission.check(name, args):
                results.append(ToolMessage(
                    content="❌ İstifadəçi rədd etdi",
                    tool_call_id=tid))
                continue

            try:
                out = TOOL_MAP[name].invoke(args) if name in TOOL_MAP \
                     else f"❌ Alət yoxdur: {name}"
            except Exception as e:
                out = f"❌ {e}"

            preview = str(out)[:200].replace("\n", " ")
            console.print(f"[dim]   → {preview}[/dim]")

            results.append(ToolMessage(content=str(out), tool_call_id=tid))
        return results

    def ask(self, user_input: str) -> str:
        if not self.messages:
            self.messages.append(SystemMessage(content=CHAT_SYSTEM.format(
                tools=self._tools_desc())))

        self.messages.append(HumanMessage(content=user_input))

        # Limitsiz iterasiya (praktik: 30)
        for _ in range(30):
            try:
                ai = self.llm_tools.invoke(self.messages)
            except Exception as e:
                return f"❌ LLM xətası: {e}"

            self.messages.append(ai)

            if not getattr(ai, "tool_calls", None):
                return ai.content or ""

            self.messages.extend(self._execute_tools(ai))

        return "⚠️ İterasiya limiti."

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
[bold yellow]  v2.0 — Paralel Sub-Agent Sistemi[/bold yellow]
"""

HELP = """
[bold cyan]🎯 Rejimlər:[/bold cyan]
  [green]/chat[/green]        — 💬 Sual-cavab (default)
  [green]/build[/green]       — 🏗️  Paralel layihə yazma

[bold cyan]🚀 Build əmrləri:[/bold cyan]
  [green]/build <təsvir>[/green]         — Layihəni paralel yaz
  [green]/parallel <N>[/green]           — Eyni vaxtda işləyən agent sayı (default: 4)

[bold cyan]📋 Əmrlər:[/bold cyan]
  [green]/help[/green]        — Kömək
  [green]/mode[/green]        — İcazə rejimi (auto/manual/readonly/write)
  [green]/reset[/green]       — Kontekst sıfırla
  [green]/tools[/green]       — Alətlər
  [green]/tree[/green]        — Qovluq strukturu
  [green]/clear[/green]       — Ekranı təmizlə
  [green]/exit[/green]        — Çıxış

[bold cyan]💡 Nümunələr:[/bold cyan]

  [dim]Chat:[/dim]
    • Python-da async/await nədir?
    • React hooks necə işləyir?

  [dim]Build (paralel):[/dim]
    • /build Python CLI todo app: add/list/done/delete
    • /build FastAPI REST API: users + posts + comments endpoints
    • /build Node.js Telegram bot: /start /help /echo commands

[dim]İpucu: Build rejimində agent 4-8 paralel sub-agent işə salır. Hər biri öz fayllarını yazır. Toqquşma yoxdur![/dim]
"""

STYLE = Style.from_dict({"prompt": "bold #00d4ff"})


def cmd_tools():
    t = Table(title=f"🔧 Alətlər ({len(TOOLS)})", header_style="bold")
    t.add_column("#", style="cyan", justify="right")
    t.add_column("Ad", style="green")
    t.add_column("Təsvir", style="dim")
    for i, tl in enumerate(TOOLS, 1):
        desc = (tl.description or "").split("\n")[0][:70]
        t.add_row(str(i), tl.name, desc)
    console.print(t)


def interactive():
    console.print(BANNER)
    console.print(Panel(HELP, title="ℹ️ Kömək", border_style="blue"))

    permission = Permission("auto")
    chat_agent = ChatAgent(permission)
    max_workers = 4

    console.print(
        f"[bold]Model:[/bold] [green]{MODEL}[/green]  |  "
        f"[bold]Alət:[/bold] [yellow]{len(TOOLS)}[/yellow]  |  "
        f"[bold]Paralel:[/bold] [cyan]{max_workers}[/cyan] agent\n"
    )

    session = PromptSession(
        history=FileHistory(str(HISTORY_FILE)),
        style=STYLE,
    )

    while True:
        try:
            user_input = session.prompt("\n💬 Sən: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n👋 Görüşənədək!")
            break

        if not user_input:
            continue

        # Əmrlər
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/exit", "/quit"):
                console.print("👋 Görüşənədək!")
                break
            elif cmd == "/help":
                console.print(Panel(HELP, title="ℹ️ Kömək", border_style="blue"))
            elif cmd == "/clear":
                console.clear()
            elif cmd == "/tools":
                cmd_tools()
            elif cmd == "/tree":
                console.print(Markdown(TOOL_MAP["tree_view"].invoke(
                    {"directory": ".", "max_depth": 3})))
            elif cmd == "/reset":
                chat_agent.reset()
                console.print("[green]✅ Kontekst sıfırlandı.[/green]")
            elif cmd == "/mode":
                if arg in ("auto", "manual", "readonly", "write"):
                    permission.mode = arg
                    console.print(f"[green]✅ İcazə: {arg}[/green]")
                else:
                    console.print("[yellow]Seçim: auto|manual|readonly|write[/yellow]")
            elif cmd == "/parallel":
                if arg.isdigit():
                    max_workers = max(1, min(int(arg), 10))
                    console.print(f"[green]✅ Paralel agent: {max_workers}[/green]")
                else:
                    console.print(f"[yellow]Cari: {max_workers}[/yellow]")
            elif cmd == "/chat":
                console.print("[green]✅ Chat rejimi[/green]")
            elif cmd == "/build":
                if not arg:
                    console.print(
                        "[yellow]İstifadə: /build <layihə təsviri>[/yellow]")
                    continue
                # Layihə qovluğu
                project_name = re.sub(r"[^\w\-]", "_", arg[:30].lower())
                project_dir = WORK_DIR / project_name
                project_dir.mkdir(exist_ok=True)
                console.print(f"\n[bold]📁 Layihə:[/bold] {project_dir}\n")

                try:
                    builder = ParallelBuilder(
                        max_workers=max_workers, permission=permission)
                    builder.build(arg, project_dir)
                except Exception as e:
                    console.print(f"[red]❌ Build xətası: {e}[/red]")
                    import traceback
                    console.print(f"[dim]{traceback.format_exc()[:1000]}[/dim]")
            else:
                console.print("[red]❌ Naməlum əmr.[/red]")
            continue

        # Chat
        console.print()
        try:
            result = chat_agent.ask(user_input)
            if result:
                console.print("\n[bold cyan]🤖 Agent:[/bold cyan]")
                console.print(Markdown(result))
        except Exception as e:
            console.print(f"[red]❌ Xəta: {e}[/red]")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Mini-Claude v2.0")
    ap.add_argument("--mode", default="auto",
                    choices=["auto", "manual", "readonly", "write"])
    ap.add_argument("--task", "-t", help="Birbaşa tapşırıq (chat)")
    ap.add_argument("--build", "-b", help="Layihəni paralel yaz")
    ap.add_argument("--parallel", "-p", type=int, default=4,
                    help="Paralel agent sayı")
    ap.add_argument("--out", "-o", default="./generated_project",
                    help="Çıxış qovluğu")
    args = ap.parse_args()

    permission = Permission(args.mode)

    if args.build:
        console.print(BANNER)
        project_dir = Path(args.out).expanduser().resolve()
        project_dir.mkdir(parents=True, exist_ok=True)
        builder = ParallelBuilder(
            max_workers=args.parallel, permission=permission)
        try:
            builder.build(args.build, project_dir)
        except KeyboardInterrupt:
            console.print("\n⏸  Dayandırıldı.")
        return

    if args.task:
        console.print(BANNER)
        chat = ChatAgent(permission)
        result = chat.ask(args.task)
        if result:
            console.print("\n[bold cyan]🤖 Agent:[/bold cyan]")
            console.print(Markdown(result))
        return

    interactive()


if __name__ == "__main__":
    main()
