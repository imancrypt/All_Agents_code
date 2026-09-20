#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  BUILDER AGENT v4.0 — Sadə Task Builder                                  ║
║                                                                          ║
║  İstifadə:                                                               ║
║    python builder_agent.py                                               ║
║                                                                          ║
║  Xüsusiyyətlər:                                                          ║
║    • İnteraktiv menyu                                                    ║
║    • Task-based generation (timeout-dan qaçır)                           ║
║    • Checkpoint + Resume                                                 ║
║    • Auto-verification                                                   ║
║    • Export (ZIP/TAR)                                                    ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import re
import sys
import ast
import json
import time
import zipfile
import tarfile
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.rule import Rule
from rich.align import Align
from rich.text import Text
from rich import box

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
        "[yellow]Necə düzəltməli:[/yellow]\n"
        "1. `.env` faylı yarat:\n"
        "   [dim]NVIDIA_API_KEY=nvapi-xxxx[/dim]\n"
        "2. Və ya mühit dəyişəni:\n"
        "   [dim]export NVIDIA_API_KEY=nvapi-xxxx[/dim]",
        title="⚠️  Konfiqurasiya", border_style="red"))
    sys.exit(1)

DATA_DIR = Path.home() / ".builder_agent"
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(DATA_DIR / "logs" / f"agent_{datetime.now():%Y%m%d}.log")],
)
log = logging.getLogger("builder")


# ══════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ══════════════════════════════════════════════════════════════════════════
class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"

    @property
    def icon(self) -> str:
        return {
            "pending": "⏳", "running": "🔄", "done": "✅",
            "failed": "❌", "skipped": "⏭️",
        }[self.value]


@dataclass
class Task:
    id: str
    title: str
    files: List[str]
    description: str = ""
    depends_on: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_sec: float = 0.0
    error: str = ""
    verification: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectPlan:
    project_name: str
    description: str
    tasks: List[Task]
    tech_stack: List[str] = field(default_factory=list)
    complexity: str = "medium"
    estimated_minutes: int = 0


# ══════════════════════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════════════════════
class LLM:
    def __init__(self):
        from langchain_openai import ChatOpenAI
        self.client = ChatOpenAI(
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL,
            model=MODEL,
            temperature=0.2,
            max_tokens=8192,
            timeout=150,
            max_retries=2,
        )

    def generate(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        buf = ""
        for chunk in self.client.stream([
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]):
            buf += chunk.content or ""
        return buf


# ══════════════════════════════════════════════════════════════════════════
# VERIFIER
# ══════════════════════════════════════════════════════════════════════════
class Verifier:
    @staticmethod
    def verify(path: Path) -> Tuple[bool, str]:
        ext = path.suffix.lower()
        try:
            content = path.read_text(encoding="utf-8")
            if not content.strip():
                return False, "Boş fayl"

            if ext == ".json":
                json.loads(content); return True, "JSON ✓"
            if ext in (".yaml", ".yml"):
                try:
                    import yaml; yaml.safe_load(content)
                    return True, "YAML ✓"
                except ImportError: return True, "YAML (skip)"
            if ext == ".py":
                ast.parse(content); return True, "Python ✓"
            if ext == ".toml":
                try:
                    import tomllib; tomllib.loads(content)
                    return True, "TOML ✓"
                except ImportError: return True, "TOML (skip)"
            if ext in (".ts", ".tsx", ".js", ".jsx", ".mjs"):
                if content.count("{") != content.count("}"):
                    return False, f"Braces {content.count('{')}/{content.count('}')}"
                if content.count("(") != content.count(")"):
                    return False, "Parens balanssız"
                if content.count("[") != content.count("]"):
                    return False, "Brackets balanssız"
                return True, f"{ext[1:].upper()} ✓"
            return True, f"{ext or 'txt'} ✓"
        except SyntaxError as e:
            return False, f"Syntax: {e.msg} (sətir {e.lineno})"
        except Exception as e:
            return False, str(e)[:150]


# ══════════════════════════════════════════════════════════════════════════
# FILE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════
FILE_MARKER_RE = re.compile(
    r"===\s*FILE:\s*([^\s=]+)\s*===\r?\n(.*?)\r?\n===\s*END\s*===",
    re.DOTALL,
)
CODE_BLOCK_RE = re.compile(
    r"```([\w+\-./]*)?\s*(?:title=|path=)?([^\n`]*)\n(.*?)```",
    re.DOTALL,
)


def extract_files(response: str) -> List[Dict[str, str]]:
    files = []
    for m in FILE_MARKER_RE.finditer(response):
        path = m.group(1).strip()
        content = m.group(2)
        if content.strip():
            files.append({"path": path, "content": content})
    if files:
        return files

    for m in CODE_BLOCK_RE.finditer(response):
        lang = (m.group(1) or "").strip()
        header = (m.group(2) or "").strip()
        content = m.group(3)
        candidate = None
        if header and re.match(r"^[\w\-./]+\.\w+$", header):
            candidate = header
        elif lang and re.match(r"^[\w\-./]+\.\w+$", lang):
            candidate = lang
        if candidate:
            files.append({"path": candidate, "content": content})
    return files


# ══════════════════════════════════════════════════════════════════════════
# PLANNER
# ══════════════════════════════════════════════════════════════════════════
PLANNER_SYSTEM = """Sən layihə planlayıcısı-san. Təsviri ATOMİK task-lara böl.

QAYDALAR:
1. Hər task 1-3 fayl (daha çox YOX)
2. Hər task 100-400 sətir (daha çox YOX)
3. Sıra məntiqli (asılılıqlar əvvəl)
4. depends_on ilə asılılıq göstər

ÇIXIŞ (yalnız JSON):
{
  "project_name": "ad",
  "description": "1 cümlə",
  "tech_stack": ["TypeScript"],
  "complexity": "low|medium|high|expert",
  "estimated_minutes": 60,
  "tasks": [
    {
      "id": "t01",
      "title": "Qısa başlıq",
      "files": ["path/file.ext"],
      "description": "Nə ediləcək",
      "depends_on": []
    }
  ]
}
"""


def plan_project(spec: str, llm: LLM) -> ProjectPlan:
    with console.status("[bold yellow]📋 Planlama...[/bold yellow]", spinner="dots"):
        response = llm.generate(PLANNER_SYSTEM, f"Layihə:\n\n{spec[:4000]}")

    m = re.search(r"\{.*\}", response, re.DOTALL)
    if not m:
        raise ValueError("Plan JSON tapılmadı")
    data = json.loads(m.group(0))

    tasks = [Task(
        id=t["id"], title=t["title"], files=t["files"],
        description=t.get("description", ""),
        depends_on=t.get("depends_on", []),
    ) for t in data["tasks"]]

    return ProjectPlan(
        project_name=data.get("project_name", "unnamed"),
        description=data.get("description", ""),
        tasks=tasks,
        tech_stack=data.get("tech_stack", []),
        complexity=data.get("complexity", "medium"),
        estimated_minutes=data.get("estimated_minutes", len(tasks) * 5),
    )


# ══════════════════════════════════════════════════════════════════════════
# GENERATOR
# ══════════════════════════════════════════════════════════════════════════
GENERATOR_SYSTEM = """Sən kod yazan agentsən. YALNIZ göstərilən faylları yarat.

ÇIXIŞ (MÜTLƏQ):
=== FILE: path/to/file.ext ===
<fayl məzmunu>
=== END ===

QAYDALAR:
1. Yalnız göstərilən faylları yarat
2. Hər fayl TAM işlək olsun
3. Import-lar tam
4. Versiyalar kontekstə uyğun
"""


def generate_task(task: Task, spec: str, context: str, llm: LLM) -> Dict[str, str]:
    prompt = f"""SPESİFİKASİYA:
{spec[:1200]}

═════ KONTEKST ═════
{context[:2500]}

═════ TASK #{task.id}: {task.title} ═════
Fayllar: {', '.join(task.files)}
Təsvir: {task.description}

İndi bu faylları yarat:
{chr(10).join('  • ' + f for f in task.files)}
"""
    response = llm.generate(GENERATOR_SYSTEM, prompt)
    files = extract_files(response)
    if not files:
        raise ValueError("Heç bir fayl çıxarılmadı")
    return {f["path"]: f["content"] for f in files}


# ══════════════════════════════════════════════════════════════════════════
# PROJECT STATE
# ══════════════════════════════════════════════════════════════════════════
class ProjectState:
    def __init__(self, project_dir: Path):
        self.dir = project_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.dir / ".builder_state.json"
        self.state = self._load()

    def _load(self) -> dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "created": datetime.now().isoformat(),
            "spec": "", "plan": None,
            "completed": [], "failed": [],
        }

    def save(self):
        self.state["updated"] = datetime.now().isoformat()
        self.state_file.write_text(
            json.dumps(self.state, indent=2, ensure_ascii=False),
            encoding="utf-8")

    def set_plan(self, plan: ProjectPlan):
        self.state["plan"] = {
            "project_name": plan.project_name,
            "description": plan.description,
            "tech_stack": plan.tech_stack,
            "complexity": plan.complexity,
            "estimated_minutes": plan.estimated_minutes,
            "tasks": [asdict(t) for t in plan.tasks],
        }
        self.save()

    def get_plan(self) -> Optional[ProjectPlan]:
        p = self.state.get("plan")
        if not p:
            return None
        tasks = []
        for td in p["tasks"]:
            td["status"] = TaskStatus(td.get("status", "pending"))
            tasks.append(Task(**td))
        return ProjectPlan(
            project_name=p["project_name"], description=p["description"],
            tasks=tasks, tech_stack=p.get("tech_stack", []),
            complexity=p.get("complexity", "medium"),
            estimated_minutes=p.get("estimated_minutes", 0),
        )

    def next_pending(self) -> Optional[Task]:
        plan = self.get_plan()
        if not plan:
            return None
        done = {t.id for t in plan.tasks if t.status == TaskStatus.DONE}
        for t in plan.tasks:
            if t.status in (TaskStatus.DONE, TaskStatus.SKIPPED):
                continue
            if all(d in done for d in t.depends_on):
                return t
        return None

    def mark(self, task: Task):
        plan = self.get_plan()
        if not plan:
            return
        for i, t in enumerate(plan.tasks):
            if t.id == task.id:
                plan.tasks[i] = task
                break
        self.set_plan(plan)
        if task.status == TaskStatus.DONE and task.id not in self.state["completed"]:
            self.state["completed"].append(task.id)
        if task.status == TaskStatus.FAILED and task.id not in self.state["failed"]:
            self.state["failed"].append(task.id)
        self.save()

    def stats(self) -> dict:
        plan = self.get_plan()
        if not plan:
            return {"total": 0, "done": 0, "failed": 0, "remaining": 0, "skipped": 0}
        total = len(plan.tasks)
        done = sum(1 for t in plan.tasks if t.status == TaskStatus.DONE)
        failed = sum(1 for t in plan.tasks if t.status == TaskStatus.FAILED)
        skipped = sum(1 for t in plan.tasks if t.status == TaskStatus.SKIPPED)
        return {"total": total, "done": done, "failed": failed,
                "skipped": skipped, "remaining": total - done - skipped}


# ══════════════════════════════════════════════════════════════════════════
# EXPORTER
# ══════════════════════════════════════════════════════════════════════════
class Exporter:
    @staticmethod
    def zip(project_dir: Path) -> Path:
        out = project_dir.parent / f"{project_dir.name}.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in project_dir.rglob("*"):
                if f.is_file() and ".builder_state.json" not in str(f):
                    z.write(f, f.relative_to(project_dir))
        return out

    @staticmethod
    def tar(project_dir: Path) -> Path:
        out = project_dir.parent / f"{project_dir.name}.tar.gz"
        with tarfile.open(out, "w:gz") as t:
            t.add(project_dir, arcname=project_dir.name,
                  filter=lambda ti: None
                  if ".builder_state.json" in ti.name else ti)
        return out


# ══════════════════════════════════════════════════════════════════════════
# BUILDER AGENT
# ══════════════════════════════════════════════════════════════════════════
class BuilderAgent:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir.resolve()
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.llm = LLM()
        self.state = ProjectState(self.project_dir)
        self.start_time = time.time()

    def _write_files(self, files: Dict[str, str]) -> List[str]:
        written = []
        for rel, content in files.items():
            rel = rel.lstrip("/").replace("..", "_")
            target = self.project_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            written.append(rel)
        return written

    def _context(self, max_tasks: int = 10) -> str:
        plan = self.state.get_plan()
        if not plan:
            return ""
        parts = []
        for t in plan.tasks:
            if t.status == TaskStatus.DONE:
                parts.append(f"✅ {t.id}: {t.title}\n   → {', '.join(t.files)}")
            elif t.status == TaskStatus.FAILED:
                parts.append(f"❌ {t.id}: {t.title}")
        return "\n".join(parts[-max_tasks:])

    def execute_task(self, task: Task, spec: str) -> bool:
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now().isoformat()
        task.attempts += 1
        self.state.mark(task)

        console.print()
        console.print(Panel(
            f"[bold]{task.id}[/bold] — {task.title}\n"
            f"[dim]📁 {len(task.files)} fayl: {', '.join(task.files)}[/dim]\n"
            f"[dim]🔁 Cəhd {task.attempts}/{task.max_attempts}[/dim]",
            border_style="cyan", box=box.ROUNDED))

        context = self._context()
        t0 = time.time()

        try:
            with console.status("[cyan]Kod generasiya olunur...[/cyan]",
                              spinner="dots12"):
                files = generate_task(task, spec, context, self.llm)
        except Exception as e:
            task.error = str(e)[:300]
            task.status = TaskStatus.FAILED
            task.duration_sec = time.time() - t0
            self.state.mark(task)
            console.print(f"[red]  ✗ Generasiya uğursuz: {e}[/red]")
            return False

        written = self._write_files(files)

        verify = {}
        all_ok = True
        for rel in written:
            p = self.project_dir / rel
            ok, msg = Verifier.verify(p)
            verify[rel] = {"ok": ok, "msg": msg}
            if ok:
                console.print(f"  [green]✓[/green] {rel} [dim]{msg}[/dim]")
            else:
                console.print(f"  [red]✗[/red] {rel} [red]{msg}[/red]")
                all_ok = False

        task.verification = verify
        task.duration_sec = time.time() - t0

        if all_ok:
            task.status = TaskStatus.DONE
            task.completed_at = datetime.now().isoformat()
            self.state.mark(task)
            console.print(
                f"[green]✅ {task.id} tamamlandı[/green] "
                f"[dim]({task.duration_sec:.1f}s)[/dim]")
            return True
        else:
            task.error = "Verification failed"
            task.status = TaskStatus.FAILED
            self.state.mark(task)
            console.print(f"[red]❌ {task.id} uğursuz (verification)[/red]")
            return False

    def build(self, spec: str, resume: bool = True) -> ProjectPlan:
        # Başlıq
        console.print()
        console.print(Panel.fit(
            f"[bold cyan]🚀 BUILDER AGENT v4.0[/bold cyan]\n\n"
            f"[bold]Qovluq:[/bold] {self.project_dir}\n"
            f"[bold]Model:[/bold] {MODEL}",
            border_style="cyan", box=box.DOUBLE))

        # Resume?
        existing = self.state.get_plan()
        if existing and resume:
            stats = self.state.stats()
            console.print()
            console.print(Panel(
                f"📂 Davam edən layihə: [bold]{existing.project_name}[/bold]\n"
                f"   Bitdi: [green]{stats['done']}/{stats['total']}[/green]   "
                f"Uğursuz: [red]{stats['failed']}[/red]   "
                f"Qaldı: [yellow]{stats['remaining']}[/yellow]",
                title="♻️  Resume", border_style="yellow"))
            if not Confirm.ask("Davam edilsin?", default=True):
                existing = None
                self.state.state["plan"] = None
                self.state.save()

        # Plan
        if not existing:
            try:
                plan = plan_project(spec, self.llm)
            except Exception as e:
                console.print(f"[red]❌ Planlama uğursuz: {e}[/red]")
                sys.exit(1)
            self.state.state["spec"] = spec
            self.state.set_plan(plan)
        else:
            plan = existing

        self._render_plan(plan)

        if not Confirm.ask("\n▶ İcraya başlansın?", default=True):
            console.print("[yellow]Ləğv edildi.[/yellow]")
            return plan

        # İcra
        while True:
            task = self.state.next_pending()
            if not task:
                break

            if task.attempts >= task.max_attempts:
                console.print(f"[yellow]⚠️  {task.id} — max cəhd, skip[/yellow]")
                task.status = TaskStatus.SKIPPED
                self.state.mark(task)
                continue

            ok = self.execute_task(task, spec)

            if not ok and task.attempts < task.max_attempts:
                console.print(f"[yellow]🔄 Yenidən cəhd...[/yellow]")

        self._final_summary(plan)
        return plan

    def _render_plan(self, plan: ProjectPlan):
        console.print()
        t = Table(title=f"📋 Plan — {plan.project_name}",
                  header_style="bold", box=box.ROUNDED)
        t.add_column("#", style="cyan", justify="right")
        t.add_column("ID", style="dim")
        t.add_column("Başlıq", style="white")
        t.add_column("📁", justify="right")
        t.add_column("→", style="yellow", justify="center")
        t.add_column("Status", justify="center")
        for i, task in enumerate(plan.tasks, 1):
            t.add_row(
                str(i), task.id, task.title[:45],
                str(len(task.files)),
                ", ".join(task.depends_on) or "—",
                f"{task.status.icon}",
            )
        console.print(t)
        console.print(
            f"[dim]Stack: {', '.join(plan.tech_stack)} │ "
            f"Complexity: [bold]{plan.complexity}[/bold] │ "
            f"~{plan.estimated_minutes} dəq │ "
            f"{len(plan.tasks)} task[/dim]")

    def _final_summary(self, plan: ProjectPlan):
        stats = self.state.stats()
        elapsed = time.time() - self.start_time

        console.print()
        console.print(Rule("[bold cyan]🎉 Yekun[/bold cyan]", style="cyan"))

        t = Table(box=box.SIMPLE, show_header=False)
        t.add_column("", style="cyan")
        t.add_column("", style="white")
        t.add_row("Layihə", plan.project_name)
        t.add_row("Uğurlu", f"[green]{stats['done']}/{stats['total']}[/green]")
        t.add_row("Uğursuz", f"[red]{stats['failed']}[/red]")
        t.add_row("Skip", f"[yellow]{stats['skipped']}[/yellow]")
        t.add_row("Müddət", f"{elapsed:.0f}s")
        t.add_row("Qovluq", str(self.project_dir))
        console.print(t)


# ══════════════════════════════════════════════════════════════════════════
# CLI — İNTERAKTİV MENYU
# ══════════════════════════════════════════════════════════════════════════
BANNER = r"""
[bold cyan]
  ██████╗ ██╗   ██╗██╗██╗     ██████╗ ███████╗██████╗ 
  ██╔══██╗██║   ██║██║██║     ██╔══██╗██╔════╝██╔══██╗
  ██████╔╝██║   ██║██║██║     ██║  ██║█████╗  ██████╔╝
  ██╔══██╗██║   ██║██║██║     ██║  ██║██╔══╝  ██╔══██╗
  ██████╔╝╚██████╔╝██║███████╗██████╔╝███████╗██║  ██║
  ╚═════╝  ╚═════╝ ╚═╝╚══════╝╚═════╝ ╚══════╝╚═╝  ╚═╝
[/bold cyan]
[bold yellow]         v4.0 — Task-Based Project Builder[/bold yellow]
"""


def show_banner():
    console.clear()
    console.print(BANNER)
    console.print(Align.center(f"[dim]Model: {MODEL}[/dim]\n"))


def menu_main() -> str:
    console.print("[bold cyan]Nə etmək istəyirsən?[/bold cyan]\n")
    console.print("  [cyan]1[/cyan]  🆕 Yeni layihə yarat")
    console.print("  [cyan]2[/cyan]  ♻️  Davam edən layihəni davam etdir")
    console.print("  [cyan]3[/cyan]  📊 Layihə statusunu göstər")
    console.print("  [cyan]4[/cyan]  📦 Layihəni ixrac et (ZIP)")
    console.print("  [cyan]5[/cyan]  📦 Layihəni ixrac et (TAR.GZ)")
    console.print("  [cyan]0[/cyan]  🚪 Çıxış\n")
    return Prompt.ask("[yellow]Seçim[/yellow]",
                     choices=["0", "1", "2", "3", "4", "5"], default="1")


def ask_project_dir() -> Path:
    default = Path.cwd() / "generated_project"
    console.print(f"\n[dim]Default: {default}[/dim]")
    path = Prompt.ask("[yellow]Layihə qovluğu[/yellow]",
                     default=str(default))
    return Path(path).expanduser().resolve()


def ask_spec_text() -> str:
    console.print("\n[bold cyan]Layihə spesifikasiyası[/bold cyan]")
    console.print("[dim]Çoxsətirli daxil et. Bitirmək üçün boş sətir + Enter.[/dim]\n")

    lines = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        if not line.strip() and lines:
            break
        lines.append(line)

    return "\n".join(lines)


def ask_spec_file() -> Optional[str]:
    path = Prompt.ask("\n[cyan]Və ya fayl yolu (boş buraxsan əl ilə yaz)[/cyan]",
                     default="")
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.exists():
        console.print(f"[red]❌ Fayl tapılmadı: {p}[/red]")
        return None
    return p.read_text(encoding="utf-8")


def show_status(project_dir: Path):
    state = ProjectState(project_dir)
    plan = state.get_plan()
    if not plan:
        console.print(Panel(
            "[yellow]Bu qovluqda layihə state yoxdur.[/yellow]\n"
            f"[dim]{project_dir}[/dim]",
            border_style="yellow"))
        return

    stats = state.stats()
    console.print()
    console.print(Panel.fit(
        f"[bold]Layihə:[/bold] {plan.project_name}\n"
        f"[bold]Açıqlama:[/bold] {plan.description}\n"
        f"[bold]Complexity:[/bold] {plan.complexity}\n"
        f"[bold]Stack:[/bold] {', '.join(plan.tech_stack)}\n\n"
        f"[bold]Progress:[/bold] [green]{stats['done']}[/green]/"
        f"{stats['total']}   "
        f"[red]Uğursuz: {stats['failed']}[/red]   "
        f"[yellow]Skip: {stats['skipped']}[/yellow]   "
        f"[cyan]Qaldı: {stats['remaining']}[/cyan]",
        title="📊 Status", border_style="cyan"))

    t = Table(title="Tasks", box=box.ROUNDED, header_style="bold")
    t.add_column("#", style="cyan", justify="right")
    t.add_column("ID", style="dim")
    t.add_column("Başlıq", style="white")
    t.add_column("Fayllar", style="green", justify="right")
    t.add_column("Status", justify="center")
    t.add_column("Müddət", justify="right", style="dim")
    for i, task in enumerate(plan.tasks, 1):
        t.add_row(
            str(i), task.id, task.title[:45],
            str(len(task.files)),
            f"{task.status.icon} {task.status.value}",
            f"{task.duration_sec:.1f}s" if task.duration_sec else "—")
    console.print(t)


def export_project(project_dir: Path, fmt: str):
    if not project_dir.exists():
        console.print(f"[red]❌ Qovluq yoxdur: {project_dir}[/red]")
        return
    try:
        if fmt == "zip":
            out = Exporter.zip(project_dir)
        else:
            out = Exporter.tar(project_dir)
        size_mb = out.stat().st_size / 1024 / 1024
        console.print(Panel(
            f"[green]✅ İxrac edildi[/green]\n\n"
            f"[bold]Fayl:[/bold] {out}\n"
            f"[bold]Ölçü:[/bold] {size_mb:.2f} MB",
            border_style="green"))
    except Exception as e:
        console.print(f"[red]❌ İxrac xətası: {e}[/red]")


def run_new_project():
    console.print()
    console.print(Rule("[cyan]🆕 Yeni Layihə[/cyan]"))

    project_dir = ask_project_dir()

    # Köhnə state?
    if (project_dir / ".builder_state.json").exists():
        console.print(f"[yellow]⚠️  {project_dir} içində köhnə state var.[/yellow]")
        if not Confirm.ask("Silinsin və sıfırdan başlansın?", default=True):
            console.print("[yellow]Ləğv edildi.[/yellow]")
            return

    # Spec
    spec = ask_spec_file()
    if not spec:
        spec = ask_spec_text()

    if not spec.strip():
        console.print("[red]❌ Spesifikasiya boşdur.[/red]")
        return

    console.print(f"\n[dim]Spesifikasiya: {len(spec)} simvol[/dim]")

    # Build
    agent = BuilderAgent(project_dir)
    try:
        agent.build(spec, resume=False)
    except KeyboardInterrupt:
        console.print("\n[yellow]⏸  Dayandırıldı.[/yellow]")
        agent.state.save()


def run_resume():
    console.print()
    console.print(Rule("[cyan]♻️  Davam etdir[/cyan]"))

    default = Path.cwd() / "generated_project"
    path = Prompt.ask("[yellow]Layihə qovluğu[/yellow]", default=str(default))
    project_dir = Path(path).expanduser().resolve()

    state = ProjectState(project_dir)
    plan = state.get_plan()
    if not plan:
        console.print(f"[red]❌ {project_dir} — layihə tapılmadı.[/red]")
        return

    spec = state.state.get("spec", "")
    if not spec:
        console.print("[yellow]⚠️  Spesifikasiya boşdur, yenidən daxil et.[/yellow]")
        spec = ask_spec_text()

    agent = BuilderAgent(project_dir)
    try:
        agent.build(spec, resume=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]⏸  Dayandırıldı.[/yellow]")
        agent.state.save()


def run_status():
    console.print()
    console.print(Rule("[cyan]📊 Status[/cyan]"))
    default = Path.cwd() / "generated_project"
    path = Prompt.ask("[yellow]Layihə qovluğu[/yellow]", default=str(default))
    show_status(Path(path).expanduser().resolve())
    Prompt.ask("\n[dim]Davam etmək üçün Enter[/dim]", default="")


def run_export(fmt: str):
    console.print()
    console.print(Rule(f"[cyan]📦 İxrac ({fmt.upper()})[/cyan]"))
    default = Path.cwd() / "generated_project"
    path = Prompt.ask("[yellow]Layihə qovluğu[/yellow]", default=str(default))
    export_project(Path(path).expanduser().resolve(), fmt)
    Prompt.ask("\n[dim]Davam etmək üçün Enter[/dim]", default="")


def main_menu_loop():
    while True:
        show_banner()
        choice = menu_main()

        if choice == "0":
            console.print("\n[cyan]👋 Görüşənədək![/cyan]\n")
            break
        elif choice == "1":
            run_new_project()
        elif choice == "2":
            run_resume()
        elif choice == "3":
            run_status()
        elif choice == "4":
            run_export("zip")
        elif choice == "5":
            run_export("tar")


# ══════════════════════════════════════════════════════════════════════════
# CLI ARGS
# ══════════════════════════════════════════════════════════════════════════
def main():
    # Arqument varsa, birbaşa işlə
    if len(sys.argv) > 1:
        import argparse
        ap = argparse.ArgumentParser(description="Builder Agent v4.0")
        ap.add_argument("--spec", "-s", help="Spesifikasiya faylı")
        ap.add_argument("--out", "-o", default="./generated_project",
                       help="Çıxış qovluğu")
        ap.add_argument("--status", action="store_true", help="Status göstər")
        ap.add_argument("--export", choices=["zip", "tar"], help="İxrac et")
        ap.add_argument("--resume", action="store_true",
                       help="Davam etdir")
        args = ap.parse_args()

        project_dir = Path(args.out).expanduser().resolve()

        if args.status:
            show_status(project_dir)
            return

        if args.export:
            export_project(project_dir, args.export)
            return

        if args.resume:
            state = ProjectState(project_dir)
            plan = state.get_plan()
            if not plan:
                console.print(f"[red]❌ Layihə tapılmadı: {project_dir}[/red]")
                sys.exit(1)
            spec = state.state.get("spec", "")
            if not spec:
                console.print("[red]❌ Spesifikasiya tapılmadı.[/red]")
                sys.exit(1)
            agent = BuilderAgent(project_dir)
            try:
                agent.build(spec, resume=True)
            except KeyboardInterrupt:
                console.print("\n[yellow]⏸  Dayandırıldı.[/yellow]")
                agent.state.save()
            return

        # Yeni build
        if args.spec:
            spec = Path(args.spec).read_text(encoding="utf-8")
        else:
            console.print("[yellow]Spesifikasiya daxil et (Ctrl+D bitir):[/yellow]")
            spec = sys.stdin.read()

        if not spec.strip():
            console.print("[red]❌ Spesifikasiya boşdur.[/red]")
            sys.exit(1)

        agent = BuilderAgent(project_dir)
        try:
            agent.build(spec, resume=args.resume)
        except KeyboardInterrupt:
            console.print("\n[yellow]⏸  Dayandırıldı.[/yellow]")
            agent.state.save()
        return

    # Arqument yoxsa → interaktiv menyu
    try:
        main_menu_loop()
    except KeyboardInterrupt:
        console.print("\n\n[cyan]👋 Görüşənədək![/cyan]\n")


if __name__ == "__main__":
    main()
