#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  CLAUDE-LITE v1.0 — A Tribute to Claude Code's Architecture              ║
║                                                                          ║
║  Bu agent, Claude Code-un agentic loop, alət sistemi və icazə modelini   ║
║  təqlid edir. Tək faylda, asılılıqsız işləyir.                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import sys
import json
import time
import shutil
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import tool
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.live import Live

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

# ══════════════════════════════════════════════════════════════════════════
# 1) KONFİQURASİYA
# ══════════════════════════════════════════════════════════════════════════
load_dotenv()

API_KEY  = os.getenv("NVIDIA_API_KEY")
BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL    = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")

if not API_KEY:
    print("❌ NVIDIA_API_KEY tapılmadı. .env faylına əlavə edin.")
    sys.exit(1)

ROOT_DIR = Path.cwd()
DATA_DIR = ROOT_DIR / "claude_lite_data"
DATA_DIR.mkdir(exist_ok=True)
HISTORY_FILE = DATA_DIR / "history.txt"

# Təhlükəsizlik: yalnız bu qovluqda işləyirik
SAFE_ROOT = ROOT_DIR.resolve()

console = Console()
logging.basicConfig(level=logging.WARNING)

# ══════════════════════════════════════════════════════════════════════════
# 2) KÖMƏKÇİ FUNKSİYALAR
# ══════════════════════════════════════════════════════════════════════════
def _safe_path(path: str) -> Optional[Path]:
    """Yolu təhlükəsizlik kökünə bağlayır."""
    try:
        p = Path(path)
        if not p.is_absolute():
            p = SAFE_ROOT / p
        p = p.resolve()
        if str(p).startswith(str(SAFE_ROOT)):
            return p
        return None
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════
# 3) ALƏTLƏR (TOOLS) — Claude Code-un daxili alətlərinin təqlidi
# ══════════════════════════════════════════════════════════════════════════
@tool
def read_file(path: str, offset: int = 1, limit: int = 2000) -> str:
    """Faylı oxuyur. 1-dən başlayan sətir nömrələrini qaytarır."""
    p = _safe_path(path)
    if not p or not p.is_file():
        return f"❌ Fayl tapılmadı: {path}"
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(0, offset - 1)
        chunk = lines[start:start + limit]
        numbered = "\n".join(f"{start + i + 1:6d}│ {line}" for i, line in enumerate(chunk))
        return numbered or "(boş fayl)"
    except Exception as e:
        return f"❌ Oxuma xətası: {e}"

@tool
def write_file(path: str, content: str) -> str:
    """Yeni fayl yaradır və ya mövcud faylı tamamilə əvəz edir."""
    p = _safe_path(path)
    if not p:
        return f"❌ Təhlükəsiz olmayan yol: {path}"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"✅ Fayl yazıldı: {p} ({len(content)} bayt)"
    except Exception as e:
        return f"❌ Yazma xətası: {e}"

@tool
def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Faylda dəqiq mətn əvəzləməsi edir. Köhnə mətn unikal olmalıdır."""
    p = _safe_path(path)
    if not p or not p.is_file():
        return f"❌ Fayl tapılmadı: {path}"
    try:
        content = p.read_text(encoding="utf-8")
        count = content.count(old_string)
        if count == 0:
            return "❌ `old_string` faylda tapılmadı."
        if count > 1:
            return f"❌ `old_string` faylda {count} dəfə təkrarlanır. Unikal olmalıdır."
        # Ehtiyat nüsxəsi
        backup = DATA_DIR / f"{p.name}.bak"
        backup.write_text(content, encoding="utf-8")
        new_content = content.replace(old_string, new_string, 1)
        p.write_text(new_content, encoding="utf-8")
        return f"✅ Düzəliş edildi: {p}. Ehtiyat nüsxə: {backup}"
    except Exception as e:
        return f"❌ Düzəliş xətası: {e}"

@tool
def list_files(directory: str = ".", pattern: str = "*") -> str:
    """Qovluqdakı faylları siyahılayır. `pattern` glob formatındadır (məs: '*.py')."""
    p = _safe_path(directory)
    if not p or not p.is_dir():
        return f"❌ Qovluq tapılmadı: {directory}"
    try:
        files = sorted(p.glob(pattern))
        if not files:
            return f"❌ `{pattern}` uyğun fayl tapılmadı."
        return "\n".join(str(f.relative_to(p)) for f in files[:200])
    except Exception as e:
        return f"❌ Siyahılama xətası: {e}"

@tool
def grep_search(pattern: str, path: str = ".", max_results: int = 100) -> str:
    """Fayl məzmununda regex ilə axtarış edir (ripgrep varsa onunla)."""
    p = _safe_path(path)
    if not p:
        return f"❌ Təhlükəsiz olmayan yol: {path}"
    try:
        if shutil.which("rg"):
            result = subprocess.run(
                ["rg", "-n", "--no-heading", pattern, str(p)],
                capture_output=True, text=True, timeout=30
            )
            output = result.stdout
        else:
            result = subprocess.run(
                ["grep", "-rn", "-E", pattern, str(p)],
                capture_output=True, text=True, timeout=30
            )
            output = result.stdout
        if not output.strip():
            return "🔍 Nəticə tapılmadı."
        lines = output.splitlines()[:max_results]
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Axtarış xətası: {e}"

@tool
def run_bash(command: str, timeout: int = 30) -> str:
    """Shell əmri icra edir. Təhlükəsizlik üçün whitelist tətbiq olunur."""
    SAFE_COMMANDS = {
        "ls", "pwd", "cat", "head", "tail", "wc", "grep", "find", "file",
        "echo", "date", "whoami", "uname", "df", "du", "ps", "env",
        "which", "type", "stat", "tree", "sort", "uniq", "tr", "cut",
        "awk", "sed", "jq", "git", "python3", "pip", "pip3",
    }
    base = command.strip().split()[0] if command.strip() else ""
    if base not in SAFE_COMMANDS:
        return f"❌ `{base}` əmri icazəli deyil. İcazəli əmrlər: {', '.join(sorted(SAFE_COMMANDS))}"
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout or ""
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        if len(output) > 8000:
            output = output[:8000] + f"\n... [{len(output)-8000} bayt kəsildi]"
        return f"```\n{output}\n```\n_(rc={result.returncode})_"
    except subprocess.TimeoutExpired:
        return f"⏱ Əmr {timeout} saniyədən sonra dayandırıldı."
    except Exception as e:
        return f"❌ İcra xətası: {e}"

@tool
def spawn_subagent(task: str) -> str:
    """Mürəkkəb tapşırıqları yerinə yetirmək üçün alt-agent yaradır. (Simulyasiya)"""
    return (
        f"🤖 Alt-agent yaradıldı (simulyasiya).\n"
        f"Tapşırıq: {task}\n"
        f"Qeyd: Real Claude Code-da bu, ayrı bir kontekst pəncərəsində "
        f"müstəqil işləyən bir agent yaradır[reference:9]."
    )

# Alət siyahısı
TOOLS = [
    read_file, write_file, edit_file, list_files, grep_search,
    run_bash, spawn_subagent,
]
TOOL_MAP = {t.name: t for t in TOOLS}

# ══════════════════════════════════════════════════════════════════════════
# 4) İCAZƏ QATI (PERMISSION LAYER)
# ══════════════════════════════════════════════════════════════════════════
class PermissionManager:
    """
    Claude Code-un icazə modelini təqlid edir.
    Rejimlər: 'manual' (hər dəfə soruş), 'auto' (avtomatik təsdiq),
              'readonly' (yalnız oxuma əməliyyatları).
    """
    def __init__(self, mode: str = "manual"):
        self.mode = mode
        self.readonly_tools = {"read_file", "list_files", "grep_search"}

    def check(self, tool_name: str, args: dict) -> bool:
        if self.mode == "auto":
            return True
        if self.mode == "readonly":
            return tool_name in self.readonly_tools
        # Manual mode — istifadəçidən soruş
        console.print(f"\n[yellow]🔧 Alət çağırışı:[/yellow] [bold]{tool_name}[/bold]")
        # Arqumentləri göstər (qısa)
        for k, v in args.items():
            display = str(v)
            if len(display) > 100:
                display = display[:100] + "..."
            console.print(f"   [dim]{k}:[/dim] {display}")
        return Confirm.ask("[yellow]Bu alət çağırışına icazə verilsin?[/yellow]", default=True)

# ══════════════════════════════════════════════════════════════════════════
# 5) AGENT (QUERY LOOP)
# ══════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """Sən Claude-Lite adlı bir kodlaşdırma agentsən. Claude Code-un arxitekturasını təqlid edirsən.

PRİNSİPLƏR:
• İstifadəçinin tapşırığını yerinə yetirmək üçün alətlərdən istifadə et.
• Hər alət çağırışı məqsədli olmalıdır.
• Nəticələri qiymətləndir, lazım gələrsə yenidən cəhd et.
• Azərbaycan dilində cavab ver.
• Kod bloklarını ```ilə``` işarələ.
• Uzun cavablarda struktur (başlıq, cədvəl) istifadə et.
• Yalnız təsdiqlənmiş məlumat ver, spekulyasiya etmə.

MÖVCUD ALƏTLƏR:
{tool_descriptions}
"""

class ClaudeLite:
    def __init__(self, permission_mode: str = "manual"):
        self.llm = ChatOpenAI(
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL,
            model=MODEL,
            temperature=0.2,
            max_tokens=16384,
            timeout=180,
        )
        self.llm_with_tools = self.llm.bind_tools(TOOLS)
        self.permission = PermissionManager(permission_mode)
        self.messages: List = []
        self.stats = {"tool_calls": 0, "errors": 0, "start": time.time()}

    def _get_tool_descriptions(self) -> str:
        lines = []
        for t in TOOLS:
            desc = (t.description or "").split("\n")[0]
            lines.append(f"• `{t.name}`: {desc}")
        return "\n".join(lines)

    def _execute_tools(self, ai_msg) -> List[ToolMessage]:
        """Alət çağırışlarını icra edir."""
        results = []
        for call in getattr(ai_msg, "tool_calls", []) or []:
            name = call["name"]
            args = call.get("args", {})
            tool_id = call["id"]

            # İcazə yoxlaması
            if not self.permission.check(name, args):
                result = "❌ İstifadəçi tərəfindən rədd edildi."
                results.append(ToolMessage(content=result, tool_call_id=tool_id))
                continue

            # Aləti icra et
            self.stats["tool_calls"] += 1
            try:
                if name in TOOL_MAP:
                    output = TOOL_MAP[name].invoke(args)
                else:
                    output = f"❌ Alət tapılmadı: {name}"
            except Exception as e:
                output = f"❌ Alət xətası: {e}"
                self.stats["errors"] += 1

            results.append(ToolMessage(content=str(output), tool_call_id=tool_id))
        return results

    def run(self, user_input: str) -> str:
        """Əsas agentic loop."""
        # Sistem promptunu hazırla (əgər ilk dəfədirsə)
        if not self.messages:
            system = SYSTEM_PROMPT.format(
                tool_descriptions=self._get_tool_descriptions()
            )
            self.messages.append(SystemMessage(content=system))

        self.messages.append(HumanMessage(content=user_input))

        max_iterations = 15
        for iteration in range(max_iterations):
            # Modeli çağır
            ai_msg = self.llm_with_tools.invoke(self.messages)
            self.messages.append(ai_msg)

            # Əgər alət çağırışı yoxdursa — son cavabdır
            if not getattr(ai_msg, "tool_calls", None):
                return ai_msg.content or ""

            # Alətləri icra et və nəticələri geri qaytar
            tool_results = self._execute_tools(ai_msg)
            self.messages.extend(tool_results)

        return "⚠️ Maksimum iterasiya sayına çatdı."

# ══════════════════════════════════════════════════════════════════════════
# 6) CLI
# ══════════════════════════════════════════════════════════════════════════
BANNER = r"""
[bold cyan]
  ██████╗██╗      █████╗ ██╗   ██╗██████╗ ███████╗    ██╗     ██╗████████╗███████╗
 ██╔════╝██║     ██╔══██╗██║   ██║██╔══██╗██╔════╝    ██║     ██║╚══██╔══╝██╔════╝
 ██║     ██║     ███████║██║   ██║██║  ██║█████╗      ██║     ██║   ██║   █████╗  
 ██║     ██║     ██╔══██║██║   ██║██║  ██║██╔══╝      ██║     ██║   ██║   ██╔══╝  
 ╚██████╗███████╗██║  ██║╚██████╔╝██████╔╝███████╗    ███████╗██║   ██║   ███████╗
  ╚═════╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝    ╚══════╝╚═╝   ╚═╝   ╚══════╝
[/bold cyan]
[bold yellow]  v1.0 — A Coding Agent Inspired by Claude Code[/bold yellow]
"""

HELP = """
[bold cyan]🎯 Əmrlər:[/bold cyan]
  [green]/help[/green]       — Bu kömək
  [green]/mode[/green]       — İcazə rejimini dəyiş (manual/auto/readonly)
  [green]/clear[/green]      — Ekranı təmizlə
  [green]/tools[/green]      — Mövcud alətləri göstər
  [green]/stats[/green]      — Statistika
  [green]/exit[/green]       — Çıxış

[bold]Natural dil nümunələri:[/bold]
  • "main.py faylını oxu və nə etdiyini izah et"
  • "yeni test.py faylı yarat və içində sadə bir test yaz"
  • "bütün .py fayllarını tap"
  • "requirements.txt faylında 'requests' paketinin olub-olmadığını yoxla"
  • "git status əmrini işə sal"
"""

STYLE = Style.from_dict({"prompt": "bold #00d4ff"})

def cmd_tools():
    t = Table(title=f"🔧 Alətlər ({len(TOOLS)})", header_style="bold")
    t.add_column("#", style="cyan")
    t.add_column("Ad", style="green")
    t.add_column("Təsvir", style="dim")
    for i, tool in enumerate(TOOLS, 1):
        desc = (tool.description or "").split("\n")[0][:70]
        t.add_row(str(i), tool.name, desc)
    console.print(t)

def cmd_stats(agent: ClaudeLite):
    elapsed = time.time() - agent.stats["start"]
    t = Table(title="📊 Statistika", header_style="bold")
    t.add_column("Metrik", style="cyan")
    t.add_column("Dəyər", style="green")
    t.add_row("Alət çağırışı", str(agent.stats["tool_calls"]))
    t.add_row("Xəta", str(agent.stats["errors"]))
    t.add_row("Sessiya müddəti", f"{elapsed:.0f}s")
    console.print(t)

def interactive(agent: ClaudeLite):
    console.print(BANNER)
    console.print(Panel(HELP, title="ℹ️ Kömək", border_style="blue"))
    console.print(
        f"[bold]Model:[/bold] [green]{MODEL}[/green]  |  "
        f"[bold]Alət:[/bold] [yellow]{len(TOOLS)}[/yellow]  |  "
        f"[bold]İcazə rejimi:[/bold] [cyan]{agent.permission.mode}[/cyan]\n"
    )

    session = PromptSession(history=FileHistory(str(HISTORY_FILE)), style=STYLE)

    while True:
        try:
            user_input = session.prompt("\n👤 Sən: ").strip()
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
            elif cmd == "/mode":
                if arg in ("manual", "auto", "readonly"):
                    agent.permission.mode = arg
                    console.print(f"[green]✅ İcazə rejimi: {arg}[/green]")
                else:
                    console.print("[yellow]Seçim: manual | auto | readonly[/yellow]")
            elif cmd == "/clear":
                console.clear()
            elif cmd == "/tools":
                cmd_tools()
            elif cmd == "/stats":
                cmd_stats(agent)
            else:
                console.print("[red]❌ Naməlum əmr.[/red]")
            continue

        # Agentə göndər
        console.print()
        try:
            result = agent.run(user_input)
            if result:
                console.print("\n[bold cyan]🤖 Agent:[/bold cyan]")
                console.print(Markdown(result))
        except Exception as e:
            console.print(f"[red]❌ Xəta: {e}[/red]")

# ══════════════════════════════════════════════════════════════════════════
# 7) MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Claude-Lite Agent")
    parser.add_argument("--mode", "-m", default="manual",
                        choices=["manual", "auto", "readonly"],
                        help="İcazə rejimi")
    parser.add_argument("--task", "-t", help="Birbaşa tapşırıq icra et")
    args = parser.parse_args()

    agent = ClaudeLite(permission_mode=args.mode)

    if args.task:
        console.print(BANNER)
        result = agent.run(args.task)
        if result:
            console.print("\n[bold cyan]🤖 Agent:[/bold cyan]")
            console.print(Markdown(result))
        return

    interactive(agent)

if __name__ == "__main__":
    main()
