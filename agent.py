"""
╔══════════════════════════════════════════════════════════════════════╗
║  NVIDIA NEMOTRON AGENT v6.0 PRO — Bug Hunter Edition                ║
║  Navigation • Taint • Git • Static • Kernel • Runtime • Fuzzing     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import sys
import ast
import json
import time
import shutil
import logging
import subprocess
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import tool

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.live import Live

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

# ============================================================
# 1) KONFİQURASİYA
# ============================================================
load_dotenv()

API_KEY  = os.getenv("NVIDIA_API_KEY")
BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL    = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")

if not API_KEY:
    raise SystemExit("❌ NVIDIA_API_KEY tapılmadı (.env).")

DATA_DIR = Path("agent_data"); DATA_DIR.mkdir(exist_ok=True)
(LOG_DIR := DATA_DIR / "logs").mkdir(exist_ok=True)
(REPORT_DIR := DATA_DIR / "reports").mkdir(exist_ok=True)
(CVE_CACHE := DATA_DIR / "cve_cache").mkdir(exist_ok=True)
HISTORY_FILE = DATA_DIR / "history.txt"
MEMORY_FILE  = DATA_DIR / "memory.json"

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / f"agent_{datetime.now():%Y%m%d}.log", encoding="utf-8")],
)
log = logging.getLogger("agent")

SAFE_ROOT = Path.cwd().resolve()
IGNORE_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv",
               "dist", "build", "agent_data", ".cache", "target", ".ccache"}

KERNEL_EXTS = (".c", ".h", ".S", ".s", ".cpp", ".hpp", ".cc")
CODE_EXTS = (".py", ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx",
             ".asm", ".s", ".S", ".rs", ".go", ".js", ".ts", ".java")

# ============================================================
# 2) PERSONALAR
# ============================================================
PERSONAS: Dict[str, Dict[str, str]] = {
    "default":  {"emoji": "🤖", "system": "Sən köməkçi AI agentsən."},
    "coder":    {"emoji": "👨‍💻", "system": (
        "Senior software engineer. Python, C, C++, Assembly. "
        "Best practices, edge cases, test nümunələri.")},
    "reviewer": {"emoji": "🔍", "system": (
        "Təcrübəli Code Reviewer. Struktur:\n"
        "## 🎯 Xülasə\n## 🐛 Problemlər (cədvəl)\n## ✅ Yaxşı\n## 🔧 Patch\n## 📊 Bal")},
    "systems":  {"emoji": "⚙️", "system": (
        "Systems Programmer (kernel/driver). Fokus: memory safety, UB, "
        "concurrency (race/deadlock/atomicity), ABI (SysV AMD64), cache, asm. "
        "Hər tapıntı üçün konkret sətir + fix.")},
    "security": {"emoji": "🛡", "system": (
        "Security Auditor (OWASP/CWE/exploit). CWE+CVSS. Diqqət: "
        "UAF, buffer overflow, integer overflow, format string, race, "
        "path traversal, crypto misuse, info leak.")},
    "hunter":   {"emoji": "🎯", "system": (
        "Sən 0day Vulnerability Hunter-sən. Sistematik yanaş:\n"
        "1. Attack surface mapping (user_inputs)\n"
        "2. Taint flow analysis\n"
        "3. Lock/race pattern detection\n"
        "4. Recent changes (git) + CVE bənzərliyi\n"
        "5. Sanitizer/fuzzing ilə təsdiq\n"
        "Hər tapıntı üçün: reliability, impact, exploitability, PoC sketch.")},
    "asm_expert": {"emoji": "🔬", "system": (
        "Assembly expert (x86-64, ARM64). Register rolları, stack frame, "
        "instruction mənası, side-effect, performance. Yüksək səviyyə ilə əlaqələndir.")},
    "teacher":  {"emoji": "👨‍🏫", "system": "Səbirli müəllim. Addım-addım izah."},
    "translator": {"emoji": "🌍", "system": "Tərcüməçi. AZ ⇄ EN ⇄ TR ⇄ RU."},
}

# ============================================================
# 3) KÖMƏKÇİLƏR
# ============================================================
def _is_safe_path(p: str) -> bool:
    try:
        t = (SAFE_ROOT / p).resolve() if not Path(p).is_absolute() else Path(p).resolve()
        return SAFE_ROOT in t.parents or t == SAFE_ROOT
    except Exception:
        return False


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _run(cmd, timeout=60, cwd=None, input_text=None):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, cwd=cwd, input=input_text)
        out = (r.stdout or "")
        if r.stderr:
            out += ("\n" if out else "") + r.stderr
        return r.returncode, out
    except FileNotFoundError:
        return -1, f"❌ tapılmadı: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return -2, f"⏱ Timeout ({timeout}s)"
    except Exception as e:
        return -3, f"❌ {e}"


def _iter_files(root: Path, exts: tuple = CODE_EXTS):
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        if any(part in IGNORE_DIRS for part in p.parts):
            continue
        yield p


def _read(path: str, max_chars: int = 20000) -> str:
    try:
        c = Path(path).read_text(encoding="utf-8", errors="replace")
        return c[:max_chars] + (f"\n\n... [{len(c)-max_chars} kəsildi]" if len(c) > max_chars else "")
    except Exception as e:
        return f"❌ {e}"


def _detect_lang(path: str) -> str:
    p = Path(path)
    n = p.name.lower()
    if n in ("makefile", "gnumakefile"): return "makefile"
    if n == "cmakelists.txt": return "cmake"
    ext = p.suffix.lower()
    return {".py": "python", ".pyi": "python",
            ".c": "c", ".h": "c-header",
            ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
            ".hpp": "cpp-header", ".hh": "cpp-header",
            ".asm": "assembly", ".s": "assembly", ".S": "assembly",
            ".rs": "rust", ".go": "go", ".js": "javascript",
            ".ts": "typescript", ".java": "java",
            ".o": "binary", ".so": "binary", ".a": "binary",
            ".elf": "binary", ".out": "binary",
            }.get(ext, "unknown")

# ============================================================
# 4) TOOLLAR
# ============================================================

# ---------- 4.1 ÜMUMİ ----------
@tool
def get_current_time() -> str:
    """Cari tarix və saat."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """Riyazi ifadə hesabla. Məsələn: 'sqrt(16) + 2**8'."""
    import math
    allowed = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
               "tan": math.tan, "log": math.log, "pi": math.pi, "e": math.e,
               "abs": abs, "pow": pow, "round": round}
    try:
        return f"{expression} = {eval(expression.replace('^','**'), {'__builtins__': {}}, allowed)}"
    except Exception as e:
        return f"❌ {e}"


@tool
def check_tools_available() -> str:
    """Hansı analiz alətləri quraşdırılıb."""
    tools = ["bandit", "semgrep", "cppcheck", "clang", "clang-tidy",
             "gcc", "g++", "objdump", "readelf", "nm", "strings",
             "valgrind", "nasm", "make", "cmake", "file", "rg",
             "ctags", "sparse", "smatch", "git", "curl",
             "radare2", "r2", "ROPgadget", "afl-fuzz"]
    lines = ["| Alət | Mövcud |", "|---|---|"]
    for t in tools:
        lines.append(f"| `{t}` | {'✅' if _have(t) else '❌'} |")
    return "\n".join(lines)


# ---------- 4.2 FAYL SİSTEMİ ----------
@tool
def read_file(path: str, max_chars: int = 20000) -> str:
    """Faylı oxu."""
    return _read(path, max_chars) if _is_safe_path(path) else "❌ Yol təhlükəsiz deyil."


@tool
def write_file(path: str, content: str) -> str:
    """Fayla yaz (tam əvəzləmə)."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    try:
        Path(path).write_text(content, encoding="utf-8")
        return f"✅ Yazıldı: {path} ({len(content)} simvol)"
    except Exception as e:
        return f"❌ {e}"


@tool
def apply_patch(path: str, old_text: str, new_text: str) -> str:
    """Faylda mətn əvəzlə (backup ilə)."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    try:
        p = Path(path)
        c = p.read_text(encoding="utf-8")
        if old_text not in c:
            return "❌ `old_text` tapılmadı."
        (DATA_DIR / f"{p.name}.bak").write_text(c, encoding="utf-8")
        p.write_text(c.replace(old_text, new_text, 1), encoding="utf-8")
        return f"✅ Patch tətbiq olundu. Backup: agent_data/{p.name}.bak"
    except Exception as e:
        return f"❌ {e}"


@tool
def tree_view(directory: str = ".", max_depth: int = 3) -> str:
    """Qovluq strukturunu ağac kimi göstər."""
    if not _is_safe_path(directory):
        return "❌ Yol təhlükəsiz deyil."
    root = Path(directory).resolve()
    lines = [str(root.name) + "/"]

    def walk(d: Path, prefix="", depth=0):
        if depth >= max_depth: return
        try:
            items = sorted([x for x in d.iterdir() if x.name not in IGNORE_DIRS],
                           key=lambda x: (x.is_file(), x.name))
        except PermissionError: return
        for i, item in enumerate(items):
            last = i == len(items) - 1
            lines.append(f"{prefix}{'└── ' if last else '├── '}{item.name}{'/' if item.is_dir() else ''}")
            if item.is_dir():
                walk(item, prefix + ("    " if last else "│   "), depth + 1)

    walk(root)
    return "\n".join(lines)


@tool
def find_files(directory: str = ".", pattern: str = "*.c", max_results: int = 200) -> str:
    """Fayl pattern-i ilə axtarış (məs: '*.cpp', '*.h')."""
    if not _is_safe_path(directory):
        return "❌ Yol təhlükəsiz deyil."
    try:
        files = [f for f in Path(directory).rglob(pattern)
                 if not any(p in IGNORE_DIRS for p in f.parts)]
        return "\n".join(str(f) for f in sorted(files)[:max_results]) or "❌ tapılmadı"
    except Exception as e:
        return f"❌ {e}"


@tool
def detect_language(path: str) -> str:
    """Faylın dilini müəyyən et."""
    return f"**{path}** → `{_detect_lang(path)}`"


# ---------- 4.3 KOD NAVİQASİYASI (Bug Hunter A) ----------
@tool
def find_callers(function_name: str, path: str = ".", context: int = 2) -> str:
    """Funksiyanın bütün çağırış yerlərini tap (ripgrep ilə)."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    if _have("rg"):
        cmd = ["rg", "-n", "-C", str(context), "--type-add", "code:*.{c,h,cpp,cc,hpp,py,rs,go}",
               "-tcode", rf"\b{re.escape(function_name)}\s*\(", path]
    else:
        cmd = ["grep", "-rn", "-C", str(context), rf"\b{function_name}\s*(", path]
    rc, out = _run(cmd, timeout=30)
    if not out.strip():
        return f"❌ `{function_name}` çağırışı tapılmadı."
    return f"**Caller-lər `{function_name}`:**\n```\n{out[:8000]}\n```"


@tool
def find_definition(symbol: str, path: str = ".") -> str:
    """Simvolun tərifini tap (funksiya, struct, dəyişən)."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    if _have("rg"):
        cmd = ["rg", "-n", "-t", "c", "-t", "cpp", "-t", "py",
               rf"^\s*(static\s+|inline\s+|extern\s+)?[\w\*\s]+\b{re.escape(symbol)}\s*[(;=]",
               path]
        rc, out = _run(cmd, timeout=30)
        if out.strip():
            return f"**Tərif `{symbol}`:**\n```\n{out[:5000]}\n```"
    return f"❌ `{symbol}` tərifi tapılmadı."


@tool
def call_graph(function_name: str, path: str = ".", depth: int = 2) -> str:
    """Sadə call graph (regex əsaslı, dərinlik limitli)."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    seen = set()
    lines = [f"📊 Call Graph: `{function_name}` (depth={depth})"]

    def visit(fn: str, level: int):
        if level > depth or fn in seen:
            return
        seen.add(fn)
        indent = "  " * level
        lines.append(f"{indent}└─ {fn}")
        # Bu funksiyanın gövdəsini tap, içindəki çağırışları çıxar
        for f in _iter_files(Path(path), KERNEL_EXTS + (".py",)):
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                # Funksiya gövdəsini təxmini götür
                m = re.search(rf"\b{re.escape(fn)}\s*\([^)]*\)\s*\{{", content)
                if not m: continue
                start = m.end()
                # Sadə brace matching
                depth_b = 1
                i = start
                while i < len(content) and depth_b > 0:
                    if content[i] == "{": depth_b += 1
                    elif content[i] == "}": depth_b -= 1
                    i += 1
                body = content[start:i]
                calls = set(re.findall(r"\b([a-z_][a-z0-9_]{2,})\s*\(", body))
                for c in sorted(calls)[:10]:
                    if c not in ("if", "for", "while", "switch", "return", "sizeof", "typeof"):
                        visit(c, level + 1)
            except Exception:
                continue

    visit(function_name, 0)
    return "```\n" + "\n".join(lines[:200]) + "\n```"


@tool
def generate_ctags(path: str = ".") -> str:
    """ctags ilə simvol indeksi qur (tags faylı)."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    if not _have("ctags"):
        return "❌ `ctags` quraşdırılmayıb."
    rc, out = _run(["ctags", "-R", "--languages=C,C++,Python",
                    "-f", str(DATA_DIR / "tags"), path], timeout=60)
    tag_file = DATA_DIR / "tags"
    if not tag_file.exists():
        return f"❌ ctags xətası: {out[:500]}"
    count = sum(1 for _ in tag_file.open())
    return f"✅ tags faylı yaradıldı: {tag_file} ({count} simvol)"


# ---------- 4.4 TAINT / USER INPUT (Bug Hunter A) ----------
USER_INPUT_PATTERNS = {
    "copy_from_user": r"\bcopy_from_user\s*\(",
    "get_user": r"\bget_user\s*\(",
    "strncpy_from_user": r"\bstrncpy_from_user\s*\(",
    "ioctl": r"\.unlocked_ioctl\s*=|\.compat_ioctl\s*=",
    "sysfs_show_store": r"DEVICE_ATTR|sysfs_ops|kobj_attribute",
    "procfs": r"proc_create|seq_file|single_open",
    "netlink": r"netlink_kernel_create|genl_register_family",
    "socket": r"sock_recvmsg|kernel_recvmsg",
    "mount": r"mount_|fs_context_operations|super_operations",
    "fuse": r"fuse_conn|fuse_req",
    "bpf": r"bpf_verifier_ops|bpf_func_proto",
    "mmap": r"\.mmap\s*=|remap_pfn_range",
}


@tool
def find_user_inputs(path: str = ".") -> str:
    """Kernel/user giriş nöqtələrini tap (attack surface)."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    hits = {name: [] for name in USER_INPUT_PATTERNS}
    for f in _iter_files(Path(path)):
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
            for name, pat in USER_INPUT_PATTERNS.items():
                for i, line in enumerate(content.splitlines(), 1):
                    if re.search(pat, line):
                        hits[name].append(f"{f}:{i}  {line.strip()[:100]}")
        except Exception:
            continue

    total = sum(len(v) for v in hits.values())
    if total == 0:
        return "🔍 User input nöqtəsi tapılmadı."

    out = [f"# 🎯 Attack Surface — `{path}`\n",
           f"**Ümumi entry point:** {total}\n"]
    for name, lst in hits.items():
        if not lst: continue
        out.append(f"## {name} ({len(lst)})")
        out.append("```")
        out.extend(lst[:15])
        if len(lst) > 15:
            out.append(f"... +{len(lst)-15}")
        out.append("```\n")
    return "\n".join(out)


DANGEROUS_SINKS = {
    "memcpy": r"\bmemcpy\s*\(",
    "memmove": r"\bmemmove\s*\(",
    "strcpy": r"\bstrcpy\s*\(",
    "strcat": r"\bstrcat\s*\(",
    "sprintf": r"\bsprintf\s*\(",
    "kfree": r"\bkfree\s*\(",
    "vfree": r"\bvfree\s*\(",
    "kmalloc": r"\bk(malloc|zalloc|calloc)\s*\(",
    "vmalloc": r"\bvmalloc\s*\(",
    "alloc_pages": r"\balloc_pages\s*\(",
    "copy_to_user": r"\bcopy_to_user\s*\(",
    "put_user": r"\bput_user\s*\(",
    "kstrto": r"\bkstrto(int|long|ulong|u|s)\s*\(",
    "sscanf": r"\bsscanf\s*\(",
    "container_of": r"\bcontainer_of\s*\(",
    "cast_user_ptr": r"\([^)]*__user[^)]*\)",
}


@tool
def dangerous_sinks(path: str = ".", sink: str = "") -> str:
    """Təhlükəli əməliyyatları tap (memcpy, kfree, copy_to_user...)."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    patterns = {sink: DANGEROUS_SINKS[sink]} if sink and sink in DANGEROUS_SINKS else DANGEROUS_SINKS
    hits = {name: [] for name in patterns}
    for f in _iter_files(Path(path)):
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
            for name, pat in patterns.items():
                for i, line in enumerate(content.splitlines(), 1):
                    if re.search(pat, line):
                        hits[name].append(f"{f}:{i}  {line.strip()[:110]}")
        except Exception:
            continue

    total = sum(len(v) for v in hits.values())
    out = [f"# ⚠️ Sinks — `{path}` (cəmi {total})\n"]
    for name, lst in hits.items():
        if not lst: continue
        out.append(f"**{name}** ({len(lst)}): `{lst[0].split()[0]}` və s.")
    return "\n".join(out) if total else "✅ Tapılmadı."


@tool
def taint_analysis(path: str = ".", source: str = "copy_from_user") -> str:
    """
    Sadə taint yayılması: source funksiyasının nəticəsi 
    hansı funksiyalara ötürülür.
    """
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    results = []
    for f in _iter_files(Path(path)):
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(content.splitlines(), 1):
                if source in line:
                    # Növbəti 5 sətir
                    lines = content.splitlines()
                    ctx = "\n".join(lines[i-1:i+5])
                    # Çağırılan funksiyaları çıxar
                    calls = set(re.findall(r"\b([a-z_][a-z0-9_]{2,})\s*\(", ctx))
                    calls -= {source, "if", "return", "sizeof", "for", "while"}
                    if calls:
                        results.append(f"### `{f}:{i}`\n```c\n{ctx[:400]}\n```\n"
                                     f"**→ ötürülür:** {', '.join(sorted(calls)[:8])}")
        except Exception:
            continue

    return "\n\n".join(results[:20]) if results else f"❌ `{source}` üçün flow tapılmadı."


# ---------- 4.5 GIT HISTORY (Bug Hunter C) ----------
@tool
def recent_changes(path: str = ".", days: int = 180, author: str = "") -> str:
    """Son N gündə dəyişmiş faylları göstər."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    cmd = ["git", "-C", path, "log", f"--since={days} days ago",
           "--pretty=format:%h|%an|%ad|%s", "--date=short", "--name-only"]
    if author:
        cmd += ["--author", author]
    rc, out = _run(cmd, timeout=30)
    if rc != 0:
        return f"❌ git: {out[:500]}"

    # Commit sayı və fayl tezliyi
    commits = []
    files = Counter()
    cur = None
    for line in out.splitlines():
        if "|" in line and len(line.split("|")) >= 4:
            cur = line
            commits.append(line)
        elif line.strip() and cur:
            files[line.strip()] += 1

    lines = [f"# 📅 Recent Changes ({days} gün) — `{path}`\n",
             f"**Commit:** {len(commits)} | **Fayl:** {len(files)}\n",
             "## 🔥 Ən çox dəyişən fayllar",
             "| Fayl | Commit |", "|---|---|"]
    for f_, c in files.most_common(20):
        lines.append(f"| `{f_}` | {c} |")

    lines += ["\n## 📝 Son commit-lər"]
    for c in commits[:20]:
        lines.append(f"- `{c}`")
    return "\n".join(lines)


@tool
def blame_around(path: str, line: int, context: int = 10) -> str:
    """Sətir ətrafındaki commit-ləri göstər (git blame)."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    cmd = ["git", "blame", "-L", f"{max(1,line-context)},{line+context}", path]
    rc, out = _run(cmd, timeout=20)
    return f"**Blame `{path}:{line}`:**\n```\n{out[:5000]}\n```"


@tool
def find_cve_info(cve_id: str) -> str:
    """CVE məlumatını NVD-dən çək (cache ilə)."""
    cve_id = cve_id.upper().strip()
    cache = CVE_CACHE / f"{cve_id}.json"
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "agent/6.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            return f"❌ {cve_id} NVD-də tapılmadı."
        v = vulns[0]["cve"]
        desc = next((d["value"] for d in v.get("descriptions", []) if d["lang"] == "en"), "?")
        metrics = v.get("metrics", {})
        cvss = "?"
        for k in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            if k in metrics:
                cvss = metrics[k][0]["cvssData"].get("baseScore", "?")
                break
        refs = [r["url"] for r in v.get("references", [])][:5]

        out = [f"# {cve_id}\n",
               f"**CVSS:** {cvss}",
               f"**Təsvir:** {desc[:800]}\n",
               f"**Referanslar:**"]
        out += [f"- {r}" for r in refs]
        text = "\n".join(out)
        cache.write_text(text, encoding="utf-8")
        return text
    except Exception as e:
        return f"❌ {e}"


@tool
def search_cve(keyword: str, results: int = 10) -> str:
    """NVD-də keyword ilə CVE axtar."""
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch={urllib.parse.quote(keyword)}&resultsPerPage={results}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "agent/6.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            return f"❌ '{keyword}' üçün nəticə yoxdur."
        out = [f"# 🔍 CVE axtarış: `{keyword}` ({len(vulns)})\n"]
        for v in vulns[:results]:
            c = v["cve"]
            cid = c["id"]
            desc = next((d["value"] for d in c.get("descriptions", []) if d["lang"] == "en"), "?")
            out.append(f"### {cid}\n{desc[:300]}\n")
        return "\n".join(out)
    except Exception as e:
        return f"❌ {e}"


@tool
def fetch_cve_patch(cve_id: str) -> str:
    """CVE-nin patch-ini tap (kernel.org git-dən)."""
    cve_id = cve_id.upper()
    # Əvvəlcə NVD-dən referansları al
    info = find_cve_info.invoke({"cve_id": cve_id})
    # git.kernel.org linklərini axtar
    urls = re.findall(r"https://git\.kernel\.org/[^\s\)]+", info)
    if not urls:
        return f"❌ {cve_id} üçün kernel patch linki tapılmadı.\n\n{info[:1000]}"
    return f"**{cve_id} patch linkləri:**\n" + "\n".join(f"- {u}" for u in urls[:5])


@tool
def diff_patch(url: str) -> str:
    """Patch URL-dən diff-i yüklə."""
    try:
        # git.kernel.org linklərini raw-a çevir
        raw = url.replace("git.kernel.org/pub/scm/", "git.kernel.org/pub/scm/") \
                 .replace("/commit/", "/patch/")
        req = urllib.request.Request(raw, headers={"User-Agent": "agent/6.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            text = r.read().decode("utf-8", errors="replace")
        return f"**Patch:** {url}\n```diff\n{text[:6000]}\n```"
    except Exception as e:
        return f"❌ {e}"


@tool
def git_status() -> str:
    """git status."""
    rc, out = _run(["git", "status", "--short", "--branch"], timeout=10)
    return out or "✅ Təmiz."


@tool
def git_diff(path: str = "", staged: bool = False) -> str:
    """git diff."""
    cmd = ["git", "diff", "--no-color", "--unified=3"]
    if staged: cmd.append("--staged")
    if path: cmd += ["--", path]
    rc, out = _run(cmd, timeout=15)
    return f"```diff\n{out[:6000] or '(fərq yoxdur)'}\n```"


@tool
def git_log(n: int = 10) -> str:
    """Son n commit."""
    rc, out = _run(["git", "log", f"-{n}", "--oneline", "--decorate"], timeout=10)
    return out or "❌ Git repo deyil."


# ---------- 4.6 STATIC ANALYSIS ----------
@tool
def run_bandit_json(path: str = ".") -> str:
    """Bandit (Python) JSON + cədvəl."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    rc, out = _run(["bandit", "-r", path, "-f", "json", "-q"], timeout=180)
    if rc == -1:
        return "❌ `bandit` yoxdur."
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return f"❌ Parse xətası:\n```\n{out[:1500]}\n```"
    results = data.get("results", [])
    if not results:
        return "✅ Bandit: təmizdir."
    by_sev = Counter(r.get("issue_severity", "LOW") for r in results)
    lines = [f"# 🛡 Bandit — {path}\n",
             f"**Cəmi:** {len(results)} | 🔴 {by_sev['HIGH']} | 🟠 {by_sev['MEDIUM']} | 🟡 {by_sev['LOW']}\n",
             "| # | Sev | Test | Fayl:Sətir | CWE | Təsvir |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(results[:60], 1):
        sev = r.get("issue_severity", "?")
        e = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡"}.get(sev, "⚪")
        cwe = (r.get("issue_cwe") or {}).get("id", "-")
        fn = Path(r.get("filename", "?")).name
        msg = (r.get("issue_text") or "").replace("\n", " ")[:80]
        lines.append(f"| {i} | {e} {sev} | `{r.get('test_name','')}` | {fn}:{r.get('line_number','?')} | CWE-{cwe} | {msg} |")
    return "\n".join(lines)


@tool
def run_cppcheck(path: str = ".", enable: str = "all") -> str:
    """cppcheck (C/C++ statik analiz)."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    if not _have("cppcheck"):
        return "❌ `cppcheck` yoxdur."
    rc, out = _run(["cppcheck", "--enable=" + enable, "--inline-suppr",
                    "--template={file}:{line}: [{severity}] {id}: {message}",
                    "--quiet", path], timeout=180)
    return f"**cppcheck:**\n```\n{out[:8000] or '✅ Təmizdir.'}\n```"


@tool
def run_clang_tidy(path: str, checks: str = "*") -> str:
    """clang-tidy (C/C++)."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    if not _have("clang-tidy"):
        return "❌ `clang-tidy` yoxdur."
    rc, out = _run(["clang-tidy", path, f"--checks=-*,{checks}", "--", "-std=c++17"], timeout=120)
    return f"**clang-tidy:**\n```\n{out[:8000] or '✅ Təmizdir.'}\n```"


@tool
def run_sparse(path: str) -> str:
    """sparse — kernel-specific checker (__user, lock annotation)."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    if not _have("sparse"):
        return "❌ `sparse` yoxdur. `sudo apt install sparse`"
    rc, out = _run(["sparse", path], timeout=60)
    if not out.strip():
        return "✅ sparse: təmizdir."
    return f"**sparse:**\n```\n{out[:6000]}\n```"


@tool
def run_semgrep(path: str = ".", rules: str = "auto") -> str:
    """Semgrep ilə taint / pattern axtarışı."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    if not _have("semgrep"):
        return "❌ `semgrep` yoxdur. `pip install semgrep`"
    cmd = ["semgrep", "--config", rules, "--quiet", "--json", path]
    rc, out = _run(cmd, timeout=300)
    try:
        data = json.loads(out)
        results = data.get("results", [])
        if not results:
            return "✅ Semgrep: təmizdir."
        lines = [f"# 🔍 Semgrep ({len(results)} tapıntı)\n",
                 "| # | Qayda | Fayl:Sətir | Mesaj |", "|---|---|---|---|"]
        for i, r in enumerate(results[:60], 1):
            rule = r.get("check_id", "?").split(".")[-1]
            fn = Path(r.get("path", "?")).name
            ln = r.get("start", {}).get("line", "?")
            msg = (r.get("extra", {}).get("message", "") or "")[:70]
            lines.append(f"| {i} | `{rule}` | {fn}:{ln} | {msg} |")
        return "\n".join(lines)
    except Exception:
        return f"**Semgrep:**\n```\n{out[:6000]}\n```"


# ---------- 4.7 KERNEL-SPESİFİK (Bug Hunter B) ----------
@tool
def find_lock_issues(path: str = ".") -> str:
    """
    Lock problemlərini tap:
    - mutex_init var, mutex_lock yox
    - spin_lock/lock alınan, unlock itirilən
    - error path-da unlock olmayan
    """
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    issues = []
    for f in _iter_files(Path(path), KERNEL_EXTS):
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()

            # 1) mutex_init var, mutex_lock yox
            inits = len(re.findall(r"\bmutex_init\s*\(", content)) + \
                    len(re.findall(r"\bspin_lock_init\s*\(", content))
            locks = len(re.findall(r"\bmutex_lock\s*\(", content)) + \
                    len(re.findall(r"\bspin_lock\s*\(", content))
            if inits > 0 and locks == 0:
                issues.append(f"🔴 `{f}`: {inits} init, 0 lock — mutex istifadə olunmur")

            # 2) Hər funksiyada lock/unlock balansı
            func_re = re.compile(r"\b(\w+)\s*\([^)]*\)\s*\{")
            for m in func_re.finditer(content):
                name = m.group(1)
                start = m.end()
                depth = 1
                i = start
                while i < len(content) and depth > 0:
                    if content[i] == "{": depth += 1
                    elif content[i] == "}": depth -= 1
                    i += 1
                body = content[start:i]
                for lock, unlock in [("mutex_lock", "mutex_unlock"),
                                     ("spin_lock", "spin_unlock"),
                                     ("read_lock", "read_unlock"),
                                     ("write_lock", "write_unlock"),
                                     ("down", "up")]:
                    lc = len(re.findall(rf"\b{lock}\s*\(", body))
                    uc = len(re.findall(rf"\b{unlock}\s*\(", body))
                    if lc != uc:
                        issues.append(f"⚠️ `{f}` → `{name}()`: {lock}={lc}, {unlock}={uc} (balans pozulub)")
        except Exception:
            continue

    return "\n".join(issues[:80]) if issues else "✅ Lock problemi tapılmadı."


@tool
def find_rcu_issues(path: str = ".") -> str:
    """RCU qaydaları pozuntusu."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    issues = []
    for f in _iter_files(Path(path), KERNEL_EXTS):
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
            # rcu_dereference var, rcu_read_lock yox
            has_deref = "rcu_dereference" in content
            has_read_lock = "rcu_read_lock" in content
            if has_deref and not has_read_lock:
                issues.append(f"🔴 `{f}`: `rcu_dereference` var, `rcu_read_lock` yox")
            # synchronize_rcu + rcu_read_lock — eyni funksiyada olmamalıdır
            if "synchronize_rcu" in content and "rcu_read_lock" in content:
                # funksiya səviyyəsində yoxla
                pass
            # __rcu annotation yoxlanışı
            for i, line in enumerate(content.splitlines(), 1):
                if re.search(r"\bstruct\s+\w+\s*__rcu\s*\*\s*(\w+)", line):
                    var = re.search(r"\*\s*(\w+)", line)
                    if var and var.group(1) in content:
                        if f"rcu_dereference({var.group(1)})" not in content and \
                           f"rcu_dereference_protected({var.group(1)}" not in content:
                            issues.append(f"🟡 `{f}:{i}`: `{var.group(1)}` __rcu, dereference yox")
        except Exception:
            continue
    return "\n".join(issues[:80]) if issues else "✅ RCU problemi tapılmadı."


@tool
def find_refcount_issues(path: str = ".") -> str:
    """refcount_t / kref istifadə problemləri."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    issues = []
    for f in _iter_files(Path(path), KERNEL_EXTS):
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
            # atomic_t ilə refcount əvəzinə refcount_t istifadə etməli
            if re.search(r"atomic_(inc|dec)\s*\(\s*&\w*refcount", content):
                issues.append(f"🟡 `{f}`: `atomic_inc/dec` refcount üçün — `refcount_t` tövsiyə olunur")
            # kref_put callback yox
            for i, line in enumerate(content.splitlines(), 1):
                if "kref_put(" in line and "NULL" not in line:
                    # ikinci arqument release funksiyası olmalıdır
                    args = re.search(r"kref_put\s*\(([^)]+)\)", line)
                    if args and args.group(1).count(",") < 2:
                        issues.append(f"🟡 `{f}:{i}`: `kref_put` release funksiyası olmadan")
            # refcount_inc/dec — 0 yoxlaması
            if "refcount_dec(" in content and "refcount_dec_and_test" not in content:
                if "WARN" not in content:
                    issues.append(f"🟠 `{f}`: `refcount_dec` amma test yox — UAF riski")
        except Exception:
            continue
    return "\n".join(issues[:80]) if issues else "✅ Refcount problemi tapılmadı."


@tool
def find_user_pointer_issues(path: str = ".") -> str:
    """__user pointer cast problemləri."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    issues = []
    for f in _iter_files(Path(path), KERNEL_EXTS):
        try:
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                # __user cast
                if re.search(r"\(\s*(void|char|int|u\d+)\s*\*?\s*__user\s*\*?\s*\)", line):
                    issues.append(f"🟡 `{f}:{i}`: user pointer cast — diqqətli ol\n   `{line.strip()[:110]}`")
                # __user olmadan copy_to/from_user?
                if re.search(r"\bcopy_(to|from)_user\s*\(", line):
                    if "(" in line:
                        # pointer tipini yoxla
                        m = re.search(r"copy_(?:to|from)_user\s*\(\s*([^,]+)", line)
                        if m and "__user" not in m.group(1) and "uptr" not in m.group(1):
                            pass  # context lazımdır, burada atlayaq
        except Exception:
            continue
    return "\n".join(issues[:60]) if issues else "✅ __user problemi tapılmadı."


@tool
def find_integer_overflow(path: str = ".") -> str:
    """Potensial integer overflow pattern-ləri."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    issues = []
    pats = [
        (r"\b(size_t|unsigned\s+int|u32|u64)\s+(\w+)\s*=\s*[^;]*\*[^;]*;", "🟡 multiply — overflow yoxla"),
        (r"(\w+)\s*\+\s*(\w+)\s*>\s*\w+", "🟡 addition — overflow yoxla"),
        (r"<\s*(\w+)\s*-\s*(\w+)", "🟠 subtraction unsigned — underflow"),
        (r"kmalloc\s*\(\s*[^,]*\*", "🟡 kmalloc multiply — `array_size` istifadə et"),
        (r"memcpy\s*\([^,]+,[^,]+,\s*[^)]*\*[^)]*\)", "🟡 memcpy multiply — length yoxla"),
    ]
    for f in _iter_files(Path(path), KERNEL_EXTS):
        try:
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                for pat, msg in pats:
                    if re.search(pat, line) and "sizeof" not in line[:line.find("(") + 20]:
                        issues.append(f"{msg}\n   `{f}:{i}` — `{line.strip()[:100]}`")
                        break
        except Exception:
            continue
    return "\n".join(issues[:60]) if issues else "✅ Integer overflow pattern tapılmadı."


@tool
def find_toctou(path: str = ".") -> str:
    """TOCTOU (check-then-use) pattern-ləri."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    issues = []
    for f in _iter_files(Path(path), KERNEL_EXTS):
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
            # access_ok sonra copy_from_user — arasında validation olmalıdır
            for i, line in enumerate(lines, 1):
                if "access_ok(" in line:
                    # növbəti 10 sətirə bax
                    ctx = "\n".join(lines[i:i+10])
                    if "copy_from_user" in ctx and "if" not in ctx[:200]:
                        issues.append(f"🟡 `{f}:{i}`: access_ok sonra copy, aralıqda yoxlama yox")
                # Kilit xaricində len oxunur, sonra kilid altında istifadə
                if re.search(r"\bn\s*=\s*\w+->len\s*;", line):
                    # Növbəti 15 sətirə bax
                    ctx = "\n".join(lines[i:i+15])
                    if "mutex_lock" in ctx and ctx.index("mutex_lock") > 0:
                        issues.append(f"🟠 `{f}:{i}`: `n = ->len` kiliddən kənar (TOCTOU)")
        except Exception:
            continue
    return "\n".join(issues[:60]) if issues else "✅ TOCTOU tapılmadı."


@tool
def check_kernel_config(config_path: str = "") -> str:
    """Kernel hardening config-i yoxla."""
    paths = [config_path] if config_path else [
        f"/boot/config-{os.uname().release}",
        "/proc/config.gz", "/boot/config",
    ]
    content = ""
    src = ""
    for p in paths:
        if not p: continue
        try:
            if p.endswith(".gz"):
                import gzip
                content = gzip.open(p, "rt").read()
            else:
                content = Path(p).read_text(errors="ignore")
            src = p
            break
        except Exception:
            continue
    if not content:
        return "❌ Kernel config tapılmadı (bəlkə /proc/config.gz mövcud deyil)."

    checks = {
        "KASAN (kernel address sanitizer)": "CONFIG_KASAN=y",
        "KCSAN (race detector)": "CONFIG_KCSAN=y",
        "UBSan (UB detector)": "CONFIG_UBSAN=y",
        "HARDENED_USERCOPY": "CONFIG_HARDENED_USERCOPY=y",
        "SLAB_FREELIST_HARDENED": "CONFIG_SLAB_FREELIST_HARDENED=y",
        "SLAB_FREELIST_RANDOM": "CONFIG_SLAB_FREELIST_RANDOM=y",
        "STACKPROTECTOR_STRONG": "CONFIG_STACKPROTECTOR_STRONG=y",
        "STRICT_KERNEL_RWX": "CONFIG_STRICT_KERNEL_RWX=y",
        "FORTIFY_SOURCE": "CONFIG_FORTIFY_SOURCE=y",
        "RANDOMIZE_BASE (KASLR)": "CONFIG_RANDOMIZE_BASE=y",
        "INIT_ON_ALLOC": "CONFIG_INIT_ON_ALLOC_DEFAULT_ON=y",
        "INIT_ON_FREE": "CONFIG_INIT_ON_FREE_DEFAULT_ON=y",
        "DEBUG_LIST": "CONFIG_DEBUG_LIST=y",
        "REFCOUNT_FULL": "CONFIG_REFCOUNT_FULL=y",
        "PAGE_TABLE_ISOLATION": "CONFIG_PAGE_TABLE_ISOLATION=y",
        "VMAP_STACK": "CONFIG_VMAP_STACK=y",
    }
    lines = [f"# 🛡 Kernel Hardening — `{src}`\n",
             "| Feature | Status |", "|---|---|"]
    present = 0
    for name, key in checks.items():
        ok = key in content
        if ok: present += 1
        lines.append(f"| {name} | {'✅' if ok else '❌'} |")
    lines.append(f"\n**{present}/{len(checks)}** hardening var.")
    return "\n".join(lines)


# ---------- 4.8 RUNTIME / SANITIZER ----------
@tool
def compile_with_sanitizers(path: str, sanitizer: str = "kasan",
                            compiler: str = "gcc") -> str:
    """
    Sanitizer ilə compile (yürütmə yox).
    sanitizer: 'kasan' | 'ubsan' | 'asan' | 'msan' | 'thread'
    """
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    if not _have(compiler):
        return f"❌ `{compiler}` yoxdur."
    flag = {"kasan": "-fsanitize=kernel-address",
            "ubsan": "-fsanitize=undefined",
            "asan":  "-fsanitize=address -fno-omit-frame-pointer",
            "msan":  "-fsanitize=memory",
            "thread": "-fsanitize=thread"}.get(sanitizer)
    if not flag:
        return f"❌ Naməlum sanitizer: {sanitizer}"
    out_path = DATA_DIR / f"_san_{sanitizer}.out"
    cmd = [compiler, "-g", "-O0", *flag.split(), "-Wall", "-Wextra", path, "-o", str(out_path)]
    rc, out = _run(cmd, timeout=60)
    status = "✅ OK" if rc == 0 else f"❌ rc={rc}"
    return f"**{compiler} {sanitizer}** ({status}):\n```\n{out[:6000] or '(xəbərdarlıq yoxdur)'}\n```"


# ---------- 4.9 FUZZING ----------
@tool
def run_syzkaller(duration_sec: int = 60) -> str:
    """
    syzkaller ilə kernel fuzzing (əgər qurulubsa).
    QEYD: syzkaller manual setup tələb edir (VM/image).
    """
    if not _have("syz-manager") and not _have("syz-fuzzer"):
        return ("❌ syzkaller qurulmayıb. Manual setup tələb olunur:\n"
                "```bash\n"
                "git clone https://github.com/google/syzkaller\n"
                "cd syzkaller && make\n"
                "# + kernel image (buildroot) hazırla\n"
                "```")
    return "⚠️ syzkaller konfiqurasiyası manual lazımdır (config faylı + VM image)."


@tool
def run_afl(target: str, input_dir: str = "", timeout_sec: int = 60) -> str:
    """AFL++ ilə userspace fuzzing (qısa sessiya)."""
    if not _is_safe_path(target):
        return "❌ Yol təhlükəsiz deyil."
    if not _have("afl-fuzz"):
        return "❌ `afl-fuzz` yoxdur."
    inp = input_dir or str(DATA_DIR / "afl_in")
    out = DATA_DIR / "afl_out"
    Path(inp).mkdir(exist_ok=True)
    if not any(Path(inp).iterdir()):
        (Path(inp) / "seed").write_bytes(b"AAAA")
    cmd = ["timeout", str(timeout_sec), "afl-fuzz", "-i", inp, "-o", str(out), "--", target, "@@"]
    rc, out_text = _run(cmd, timeout=timeout_sec + 10)
    crashes = list((out / "default" / "crashes").glob("*")) if (out / "default" / "crashes").exists() else []
    return (f"**AFL++ nəticəsi:**\n```\n{out_text[:4000]}\n```\n"
            f"**Crashes:** {len([c for c in crashes if c.name != 'README.txt'])}")


@tool
def parse_crash_report(path: str) -> str:
    """KASAN/KCSAN crash trace-ini parse et."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    content = _read(path, 20000)
    # KASAN pattern-ləri
    patterns = {
        "BUG: KASAN": r"BUG: KASAN",
        "Read of size": r"Read of size (\d+)",
        "Write of size": r"Write of size (\d+)",
        "Call Trace": r"Call Trace:",
        "Allocated by": r"Allocated by task",
        "Freed by": r"Freed by task",
        "slab": r"kmalloc-\d+|kmem_cache",
    }
    lines = ["# 🔬 Crash Report Analizi\n"]
    for name, pat in patterns.items():
        m = re.search(pat, content)
        if m:
            lines.append(f"- ✅ **{name}**: `{m.group(0)[:80]}`")
    # Stack trace çıxar
    trace = re.search(r"Call Trace:(.*?)(?=\n\n|\Z)", content, re.DOTALL)
    if trace:
        lines.append(f"\n## 📚 Call Trace\n```\n{trace.group(1)[:2000]}\n```")
    return "\n".join(lines) if len(lines) > 1 else "❌ KASAN/KCSAN pattern tapılmadı."


# ---------- 4.10 BINARY ANALYSIS ----------
@tool
def file_info(path: str) -> str:
    """`file` əmri ilə fayl tipini müəyyən et."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    if not _have("file"):
        return "❌ `file` yoxdur."
    rc, out = _run(["file", "-b", path], timeout=10)
    return f"**`{path}`:** {out.strip()}"


@tool
def disassemble_binary(path: str, function: str = "") -> str:
    """objdump ilə disassemble (Intel syntax)."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    if not _have("objdump"):
        return "❌ `objdump` yoxdur."
    cmd = ["objdump", "-d", "-M", "intel", path]
    if function:
        cmd.append(f"--disassemble={function}")
    rc, out = _run(cmd, timeout=60)
    return f"**Disassembly `{path}`:**\n```asm\n{out[:8000]}\n```"


@tool
def read_elf_info(path: str) -> str:
    """readelf ilə ELF metadata."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    if not _have("readelf"):
        return "❌ `readelf` yoxdur."
    parts = []
    for flag, title in [("-h", "Header"), ("-S", "Sections"),
                        ("-s", "Symbols"), ("-l", "Program headers")]:
        rc, out = _run(["readelf", flag, path], timeout=20)
        parts.append(f"### {title}\n```\n{out[:2500]}\n```")
    return "\n\n".join(parts)


@tool
def nm_symbols(path: str) -> str:
    """nm ilə simvollar."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    if not _have("nm"):
        return "❌ `nm` yoxdur."
    rc, out = _run(["nm", "-C", "--defined-only", path], timeout=30)
    return f"**Simvollar:**\n```\n{out[:6000] or '(yoxdur)'}\n```"


@tool
def binary_strings(path: str, min_len: int = 5) -> str:
    """Binary-dən string çıxar."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    if not _have("strings"):
        return "❌ `strings` yoxdur."
    rc, out = _run(["strings", "-n", str(min_len), path], timeout=30)
    lines = out.splitlines()
    return f"**Strings ({len(lines)}):**\n```\n" + "\n".join(lines[:200]) + "\n```"


@tool
def analyze_assembly(path: str) -> str:
    """Assembly faylını analiz et (instruction/register/label/syscall)."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"❌ {e}"

    instrs, labels, regs, syscalls = [], [], set(), []
    reg_re = re.compile(r"\b(rax|rbx|rcx|rdx|rsi|rdi|rbp|rsp|r8-r15|eax|ebx|ecx|edx|esi|edi|ebp|esp|xmm\d+|ymm\d+)\b".replace("r8-r15", "r[89]|r1[0-5]"))
    label_re = re.compile(r"^\s*([A-Za-z_.][\w.$]*):")
    for line in content.splitlines():
        s = line.split(";")[0].split("#")[0].strip()
        if not s: continue
        if (m := label_re.match(s)):
            labels.append(m.group(1))
        parts = s.split(None, 1)
        if parts and parts[0][0].isalpha():
            op = parts[0].lower()
            instrs.append(op)
            if op in ("syscall", "int"): syscalls.append(s)
            for r in reg_re.findall(s): regs.add(r)

    top = Counter(instrs).most_common(15)
    out = [f"# 🔬 Assembly: `{path}`\n",
           f"**Sətir:** {len(content.splitlines())} | **Instr:** {len(instrs)} | "
           f"**Label:** {len(labels)} | **Reg:** {len(regs)} | **Syscall/int:** {len(syscalls)}\n",
           "## 📊 Top instruction-lar", "| Instr | Say |", "|---|---|"]
    for op, c in top:
        out.append(f"| `{op}` | {c} |")
    if regs:
        out.append("\n## 🧮 Register-lər\n```\n" + ", ".join(sorted(regs)) + "\n```")
    if syscalls:
        out.append("\n## ⚡ Syscall/int\n```\n" + "\n".join(syscalls[:20]) + "\n```")
    return "\n".join(out)


@tool
def run_radare2(path: str, command: str = "aa; afl") -> str:
    """radare2 ilə binary analiz."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    r2 = "r2" if _have("r2") else ("radare2" if _have("radare2") else None)
    if not r2:
        return "❌ `radare2`/`r2` yoxdur."
    cmd = [r2, "-q", "-c", command, path]
    rc, out = _run(cmd, timeout=60)
    return f"**radare2 ({command}):**\n```\n{out[:8000]}\n```"


@tool
def find_rop_gadgets(binary: str, pattern: str = "pop rdi") -> str:
    """ROPgadget ilə gadget axtarışı."""
    if not _is_safe_path(binary):
        return "❌ Yol təhlükəsiz deyil."
    if not _have("ROPgadget"):
        return "❌ `ROPgadget` yoxdur. `pip install ROPgadget`"
    rc, out = _run(["ROPgadget", "--binary", binary, "--only", "pop|ret|syscall|jmp"], timeout=60)
    if pattern:
        filtered = [l for l in out.splitlines() if pattern in l]
        return f"**ROP ({pattern}):**\n```\n" + "\n".join(filtered[:40]) + "\n```"
    return f"**ROP gadgets:**\n```\n{out[:6000]}\n```"


# ---------- 4.11 METRİKA ----------
@tool
def count_lines(path: str = ".") -> str:
    """Kod statistikası (SLOC/comment/blank, dil üzrə)."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    total = code = comments = blank = 0
    by_lang = Counter()
    file_count = 0
    for f in _iter_files(Path(path)):
        file_count += 1
        lang = _detect_lang(str(f))
        try:
            for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                total += 1; by_lang[lang] += 1
                s = line.strip()
                if not s: blank += 1
                elif s.startswith(("#", "//", "/*", "*", ";")): comments += 1
                else: code += 1
        except Exception:
            continue
    lines = [f"📊 **Statistika** (`{path}`)\n",
             f"- Fayl: **{file_count}**", f"- Sətir: **{total}**",
             f"- Kod: **{code}**", f"- Şərh: **{comments}** ({100*comments//max(1,total)}%)",
             f"- Boş: **{blank}**\n", "| Dil | Sətir |", "|---|---|"]
    for lang, n in by_lang.most_common():
        lines.append(f"| {lang} | {n} |")
    return "\n".join(lines)


@tool
def find_todos(path: str = ".", keywords: str = "TODO,FIXME,HACK,XXX,BUG,NOTE,WARN") -> str:
    """TODO/FIXME şərhlərini tap."""
    if not _is_safe_path(path):
        return "❌ Yol təhlükəsiz deyil."
    pat = re.compile(rf"({keywords})\b[:\s]*(.+)", re.IGNORECASE)
    hits = []
    for f in _iter_files(Path(path)):
        try:
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if (m := pat.search(line)):
                    hits.append(f"{f}:{i}  [{m.group(1)}]  {m.group(2).strip()[:100]}")
        except Exception:
            continue
    return "\n".join(hits[:150]) if hits else "✅ Tapılmadı."


# ---------- 4.12 ŞƏBƏKƏ ----------
@tool
def web_search(query: str, max_results: int = 5) -> str:
    """DuckDuckGo axtarışı."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "🔍 Nəticə yoxdur."
        return "\n\n".join(f"{i}. **{r['title']}**\n   {r['href']}\n   {r['body'][:200]}..."
                           for i, r in enumerate(results, 1))
    except Exception as e:
        return f"❌ {e}"


# ---------- TOOLS CƏMİ ----------
TOOLS = [
    # Ümumi
    get_current_time, calculate, check_tools_available,
    # FS
    read_file, write_file, apply_patch, tree_view, find_files, detect_language,
    # Navigation
    find_callers, find_definition, call_graph, generate_ctags,
    # Taint
    find_user_inputs, dangerous_sinks, taint_analysis,
    # Git
    recent_changes, blame_around, git_status, git_diff, git_log,
    # CVE
    find_cve_info, search_cve, fetch_cve_patch, diff_patch,
    # Static
    run_bandit_json, run_cppcheck, run_clang_tidy, run_sparse, run_semgrep,
    # Kernel-spesifik
    find_lock_issues, find_rcu_issues, find_refcount_issues,
    find_user_pointer_issues, find_integer_overflow, find_toctou,
    check_kernel_config,
    # Runtime
    compile_with_sanitizers,
    # Fuzzing
    run_syzkaller, run_afl, parse_crash_report,
    # Binary
    file_info, disassemble_binary, read_elf_info, nm_symbols,
    binary_strings, analyze_assembly, run_radare2, find_rop_gadgets,
    # Metrika
    count_lines, find_todos,
    # Şəbəkə
    web_search,
]

# ============================================================
# 5) MEMORY + STATS
# ============================================================
class PersistentMemory:
    def __init__(self, path: Path, max_turns: int = 20):
        self.path = path; self.max_turns = max_turns
        self.messages: List[Dict[str, str]] = []
        if path.exists():
            try: self.messages = json.loads(path.read_text(encoding="utf-8"))
            except Exception: self.messages = []
    def _save(self):
        self.path.write_text(json.dumps(self.messages, ensure_ascii=False, indent=2), encoding="utf-8")
    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content,
                              "ts": datetime.now().isoformat(timespec="seconds")})
        self.messages = self.messages[-self.max_turns * 2:]; self._save()
    def as_langchain(self):
        out = []
        for m in self.messages:
            if m["role"] == "user": out.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant": out.append(AIMessage(content=m["content"]))
        return out
    def clear(self): self.messages.clear(); self._save()
    def export_markdown(self) -> str:
        lines = [f"# Söhbət — {datetime.now():%Y-%m-%d %H:%M}\n"]
        for m in self.messages:
            e = "👤" if m["role"] == "user" else "🤖"
            lines.append(f"### {e} {m['role'].capitalize()}  \n_{m.get('ts','')}_\n\n{m['content']}\n\n---\n")
        return "\n".join(lines)
    def __len__(self): return len(self.messages)


class Stats:
    def __init__(self):
        self.requests = 0; self.total_time = 0.0
        self.tool_calls = 0; self.errors = 0
        self.tokens_estimate = 0; self.start_time = time.time()
    def record(self, elapsed: float, content: str = ""):
        self.requests += 1; self.total_time += elapsed
        self.tokens_estimate += len(content) // 4
    def report(self) -> Table:
        t = Table(title="📊 Sessiya", header_style="bold magenta")
        t.add_column("Metrik", style="cyan"); t.add_column("Dəyər", style="green")
        t.add_row("Sorğu", str(self.requests))
        t.add_row("Tool çağırışı", str(self.tool_calls))
        t.add_row("Xəta", str(self.errors))
        t.add_row("Ümumi müddət", f"{self.total_time:.1f}s")
        t.add_row("Orta müddət", f"{self.total_time/max(1,self.requests):.2f}s")
        t.add_row("Təxmini token", f"~{self.tokens_estimate:,}")
        t.add_row("Sessiya", f"{time.time()-self.start_time:.0f}s")
        return t

# ============================================================
# 6) AGENT
# ============================================================
class ProAgent:
    def __init__(self, persona: str = "hunter", streaming: bool = True):
        self.persona_name = persona if persona in PERSONAS else "hunter"
        self.persona = PERSONAS[self.persona_name]
        self.streaming = streaming
        self.memory = PersistentMemory(MEMORY_FILE)
        self.stats = Stats()

        self.llm = ChatOpenAI(
            openai_api_key=API_KEY, openai_api_base=BASE_URL, model=MODEL,
            temperature=0.2, max_tokens=16384, top_p=1,
            streaming=streaming, timeout=240, max_retries=3,
        )
        self.llm_tools = self.llm.bind_tools(TOOLS)
        self.tool_map = {t.name: t for t in TOOLS}
        log.info("✅ v6.0 PRO hazır | persona=%s | tools=%d", self.persona_name, len(TOOLS))

    def set_persona(self, name: str):
        if name not in PERSONAS:
            raise ValueError(f"Naməlum persona: {name}")
        self.persona_name = name
        self.persona = PERSONAS[name]

    def _messages(self, user_input: str):
        return [SystemMessage(content=self.persona["system"]),
                *self.memory.as_langchain(),
                HumanMessage(content=user_input)]

    def _exec_tools(self, ai_msg) -> List[ToolMessage]:
        results = []
        for call in getattr(ai_msg, "tool_calls", []) or []:
            name, args, tid = call["name"], call.get("args", {}), call["id"]
            console.print(f"[dim]   🔧 {name}({str(args)[:110]})[/dim]")
            log.info("Tool: %s | %s", name, args)
            self.stats.tool_calls += 1
            try:
                out = self.tool_map[name].invoke(args) if name in self.tool_map else f"❌ Tool yoxdur: {name}"
            except Exception as e:
                log.exception("Tool xətası"); self.stats.errors += 1
                out = f"❌ {e}"
            results.append(ToolMessage(content=str(out), tool_call_id=tid))
        return results

    def _stream(self, messages) -> str:
        emoji = self.persona["emoji"]
        console.print(f"\n[bold cyan]{emoji} Agent:[/bold cyan]")
        buf = ""
        try:
            with Live(console=console, refresh_per_second=10, transient=False) as live:
                for chunk in self.llm.stream(messages):
                    p = chunk.content or ""
                    buf += p; live.update(Markdown(buf))
        except Exception as e:
            self.stats.errors += 1
            console.print(f"[red]❌ Stream xətası: {e}[/red]")
        return buf

    def ask(self, user_input: str, max_iterations: int = 12) -> str:
        t0 = time.time()
        self.memory.add("user", user_input)
        messages = self._messages(user_input)
        final = ""
        try:
            for _ in range(max_iterations):
                ai = self.llm_tools.invoke(messages)
                if not getattr(ai, "tool_calls", None):
                    content = ai.content or ""
                    if self.streaming and content:
                        emoji = self.persona["emoji"]
                        console.print(f"\n[bold cyan]{emoji} Agent:[/bold cyan]")
                        console.print(Markdown(content))
                    final = content
                    break
                messages.append(ai)
                messages.extend(self._exec_tools(ai))
            else:
                console.print("[yellow]⚠️ Iterasiya limiti.[/yellow]")
                final = self._stream(messages)
            self.memory.add("assistant", final)
            self.stats.record(time.time() - t0, final)
            return final
        except Exception as e:
            self.stats.errors += 1; log.exception("Ask xətası")
            console.print(f"[red]❌ Xəta: {e}[/red]")
            return ""

    # --------- Xüsusi metodlar ---------
    def audit_file(self, path: str) -> str:
        lang = _detect_lang(path)
        hints = {
            "python": "`read_file`, `ast_analyze`, `run_bandit_json`",
            "c":      "`read_file`, `find_user_inputs`, `find_lock_issues`, `run_cppcheck`, `run_sparse`",
            "c-header": "`read_file`, `find_definition`",
            "cpp":    "`read_file`, `run_cppcheck`, `run_clang_tidy`, `find_lock_issues`",
            "assembly": "`read_file`, `analyze_assembly`",
            "binary": "`file_info`, `read_elf_info`, `nm_symbols`, `disassemble_binary`",
        }.get(lang, "`read_file`")
        return self.ask(
            f"`{path}` ({lang}) — **0day audit**. "
            f"Tool-lar: {hints}. "
            f"Hər tapıntı üçün: CWE, severity, reliability, exploitability, fix."
        )

    def hunt_surface(self, directory: str = ".") -> str:
        return self.ask(
            f"`{directory}` layihəsində **attack surface** xəritəsini çıxar:\n"
            f"1. `find_user_inputs` → bütün entry point-lər\n"
            f"2. `recent_changes(days=180)` → son dəyişikliklər\n"
            f"3. `find_lock_issues`, `find_toctou`, `find_integer_overflow`\n"
            f"4. Prioritetləşdir: ən riskli 5 fayl + səbəb."
        )

# ============================================================
# 7) CLI
# ============================================================
BANNER = r"""
[bold cyan]
 ██████╗ ██████╗  ██████╗     ██╗   ██╗ ██████╗        ██████╗ 
 ██╔══██╗██╔══██╗██╔═══██╗    ██║   ██║██╔════╝       ██╔════╝ 
 ██████╔╝██████╔╝██║   ██║    ██║   ██║███████╗ █████ ╚█████╗ 
 ██╔═══╝ ██╔══██╗██║   ██║    ╚██╗ ██╔╝██╔═══██╗      ╚═══██╗
 ██║     ██║  ██║╚██████╔╝     ╚████╔╝ ╚██████╔╝     ██████╔╝
 ╚═╝     ╚═╝  ╚═╝ ╚═════╝       ╚═══╝   ╚═════╝      ╚═════╝ 
[/bold cyan]
[bold yellow]  PRO v6.0 — Bug Hunter Edition  |  Navigation • Taint • Git • Kernel • Fuzz • Binary[/bold yellow]
"""

HELP = """
[bold cyan]🎯 Hunter Workflow:[/bold cyan]
  [green]/audit <fayl>[/green]      — Tam 0day audit
  [green]/hunt [qovluq][/green]    — Attack surface xəritəsi
  [green]/persona hunter[/green]   — Hunter mode

[bold cyan]🗺 Navigation:[/bold cyan]
  [green]/callers <fn>[/green]     — Funksiyanın çağırışları
  [green]/def <symbol>[/green]     — Tərifi tap
  [green]/callgraph <fn>[/green]   — Call graph
  [green]/ctags[/green]            — ctags indeksi

[bold cyan]💧 Taint:[/bold cyan]
  [green]/inputs [path][/green]    — User input nöqtələri
  [green]/sinks [path][/green]     — Təhlükəli əməliyyatlar
  [green]/taint <func>[/green]     — Flow analizi

[bold cyan]📅 Git / CVE:[/bold cyan]
  [green]/recent [days][/green]    — Son dəyişikliklər
  [green]/blame <fayl> <sətir>[/green] — Blame
  [green]/cve <CVE-ID>[/green]     — CVE məlumatı
  [green]/cve-search <kw>[/green]  — CVE axtarışı
  [green]/patch <CVE>[/green]      — Patch tap
  [green]/diff-url <url>[/green]   — Patch diff

[bold cyan]🔬 Kernel Analyzer:[/bold cyan]
  [green]/locks [path][/green]     — Lock problemləri
  [green]/rcu [path][/green]       — RCU audit
  [green]/refcount [path][/green]  — refcount_t
  [green]/toctou [path][/green]    — TOCTOU
  [green]/int-over [path][/green]  — Integer overflow
  [green]/uptr [path][/green]      — __user pointer

[bold cyan]🛠 Static:[/bold cyan]
  [green]/bandit [path][/green]    — Bandit
  [green]/cppcheck [path][/green]  — cppcheck
  [green]/tidy <fayl>[/green]      — clang-tidy
  [green]/sparse <fayl>[/green]    — sparse
  [green]/semgrep [path][/green]   — Semgrep

[bold cyan]⚡ Runtime / Fuzz:[/bold cyan]
  [green]/kconfig [fayl][/green]   — Kernel hardening
  [green]/san <fayl> <type>[/green]— Sanitizer compile
  [green]/afl <target>[/green]     — AFL fuzz
  [green]/crash <fayl>[/green]     — KASAN trace parse

[bold cyan]🔧 Binary / ASM:[/bold cyan]
  [green]/file <fayl>[/green] / [green]/elf[/green] / [green]/nm[/green] / [green]/disasm[/green]
  [green]/asm <fayl>[/green] / [green]/r2 <bin>[/green] / [green]/rop <bin>[/green]

[bold cyan]📊 Digər:[/bold cyan]
  [green]/lines [path][/green] / [green]/todos[/green] / [green]/check[/green]
  [green]/memory[/green] / [green]/stats[/green] / [green]/export[/green] / [green]/exit[/green]
"""

STYLE = Style.from_dict({"prompt": "bold #ffaa00"})


def _t(name: str):
    for t in TOOLS:
        if t.name == name: return t
    raise KeyError(name)


def cmd_tools():
    t = Table(title=f"🧰 Tool-lar ({len(TOOLS)})", header_style="bold")
    t.add_column("#", style="cyan"); t.add_column("Ad", style="green"); t.add_column("Təsvir", style="dim")
    for i, tl in enumerate(TOOLS, 1):
        t.add_row(str(i), tl.name, (tl.description or "").split("\n")[0].strip()[:80])
    console.print(t)


def cmd_personas():
    t = Table(title="🎭 Persona-lar", header_style="bold")
    t.add_column("Ad ", style="cyan"); t.add_column("Emoji"); t.add_column("Təsvir", style="dim")
    for k, v in PERSONAS.items():
        t.add_row(k, v["emoji"], v["system"][:70].replace("\n", " ") + "...")
    console.print(t)


def interactive(agent: ProAgent):
    console.print(BANNER)
    console.print(Panel(HELP, title="ℹ️ Kömək", border_style="blue"))
    console.print(
        f"[bold]Persona:[/bold] {agent.persona['emoji']} [cyan]{agent.persona_name}[/cyan]  |  "
        f"[bold]Model:[/bold] [green]{MODEL}[/green]  |  "
        f"[bold]Tool:[/bold] [yellow]{len(TOOLS)}[/yellow]\n"
    )
    session = PromptSession(history=FileHistory(str(HISTORY_FILE)), style=STYLE)

    while True:
        try:
            user_input = session.prompt("\n👤 Sən: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n👋 Görüşənədək!"); break
        if not user_input: continue

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=2)
            cmd = parts[0].lower()
            arg1 = parts[1].strip() if len(parts) > 1 else ""
            arg2 = parts[2].strip() if len(parts) > 2 else ""

            if cmd in ("/exit", "/quit"):
                console.print("\n[bold cyan]📊 Final:[/bold cyan]")
                console.print(agent.stats.report()); break
            elif cmd == "/help":
                console.print(Panel(HELP, title="ℹ️ Kömək", border_style="blue"))
            elif cmd == "/reset":
                agent.memory.clear(); console.print("[green]🧹 Təmiz.[/green]")
            elif cmd == "/clear":
                console.clear()
            elif cmd == "/tools":
                cmd_tools()
            elif cmd == "/personas":
                cmd_personas()
            elif cmd == "/check":
                console.print(Markdown(_t("check_tools_available").invoke({})))
            elif cmd == "/persona":
                if arg1:
                    try: agent.set_persona(arg1); console.print(f"[green]✅ {arg1}[/green]")
                    except ValueError as e: console.print(f"[red]❌ {e}[/red]")
            elif cmd == "/memory":
                msgs = agent.memory.messages
                if not msgs: console.print("[dim]Boş.[/dim]")
                else:
                    console.print(Panel(f"Ümumi: {len(msgs)} mesaj", title="💾"))
                    for m in msgs[-6:]:
                        e = "👤" if m["role"] == "user" else "🤖"
                        console.print(f"{e} [dim]{m.get('ts','')}[/dim]\n  {m['content'][:180]}\n")
            elif cmd == "/stats":
                console.print(agent.stats.report())
            elif cmd == "/export":
                f = REPORT_DIR / f"export_{datetime.now():%Y%m%d_%H%M%S}.md"
                f.write_text(agent.memory.export_markdown(), encoding="utf-8")
                console.print(f"[green]✅ {f}[/green]")

            # ---- Hunter ----
            elif cmd == "/audit":
                if not arg1: console.print("[yellow]/audit <fayl>[/yellow]")
                else: agent.audit_file(arg1)
            elif cmd == "/hunt":
                agent.set_persona("hunter"); agent.hunt_surface(arg1 or ".")

            # ---- Navigation ----
            elif cmd == "/callers":
                console.print(Markdown(_t("find_callers").invoke({"function_name": arg1})))
            elif cmd == "/def":
                console.print(Markdown(_t("find_definition").invoke({"symbol": arg1})))
            elif cmd == "/callgraph":
                console.print(Markdown(_t("call_graph").invoke({"function_name": arg1})))
            elif cmd == "/ctags":
                console.print(Markdown(_t("generate_ctags").invoke({"path": arg1 or "."})))

            # ---- Taint ----
            elif cmd == "/inputs":
                console.print(Markdown(_t("find_user_inputs").invoke({"path": arg1 or "."})))
            elif cmd == "/sinks":
                console.print(Markdown(_t("dangerous_sinks").invoke({"path": arg1 or "."})))
            elif cmd == "/taint":
                console.print(Markdown(_t("taint_analysis").invoke({"source": arg1 or "copy_from_user"})))

            # ---- Git / CVE ----
            elif cmd == "/recent":
                d = int(arg1) if arg1.isdigit() else 180
                console.print(Markdown(_t("recent_changes").invoke({"path": ".", "days": d})))
            elif cmd == "/blame":
                if arg1 and arg2:
                    console.print(Markdown(_t("blame_around").invoke({"path": arg1, "line": int(arg2)})))
            elif cmd == "/cve":
                console.print(Markdown(_t("find_cve_info").invoke({"cve_id": arg1})))
            elif cmd == "/cve-search":
                console.print(Markdown(_t("search_cve").invoke({"keyword": arg1})))
            elif cmd == "/patch":
                console.print(Markdown(_t("fetch_cve_patch").invoke({"cve_id": arg1})))
            elif cmd == "/diff-url":
                console.print(Markdown(_t("diff_patch").invoke({"url": arg1})))

            # ---- Kernel ----
            elif cmd == "/locks":
                console.print(Markdown(_t("find_lock_issues").invoke({"path": arg1 or "."})))
            elif cmd == "/rcu":
                console.print(Markdown(_t("find_rcu_issues").invoke({"path": arg1 or "."})))
            elif cmd == "/refcount":
                console.print(Markdown(_t("find_refcount_issues").invoke({"path": arg1 or "."})))
            elif cmd == "/toctou":
                console.print(Markdown(_t("find_toctou").invoke({"path": arg1 or "."})))
            elif cmd == "/int-over":
                console.print(Markdown(_t("find_integer_overflow").invoke({"path": arg1 or "."})))
            elif cmd == "/uptr":
                console.print(Markdown(_t("find_user_pointer_issues").invoke({"path": arg1 or "."})))

            # ---- Static ----
            elif cmd == "/bandit":
                console.print(Markdown(_t("run_bandit_json").invoke({"path": arg1 or "."})))
            elif cmd == "/cppcheck":
                console.print(Markdown(_t("run_cppcheck").invoke({"path": arg1 or "."})))
            elif cmd == "/tidy":
                if arg1: console.print(Markdown(_t("run_clang_tidy").invoke({"path": arg1})))
            elif cmd == "/sparse":
                if arg1: console.print(Markdown(_t("run_sparse").invoke({"path": arg1})))
            elif cmd == "/semgrep":
                console.print(Markdown(_t("run_semgrep").invoke({"path": arg1 or "."})))

            # ---- Runtime / Fuzz ----
            elif cmd == "/kconfig":
                console.print(Markdown(_t("check_kernel_config").invoke({"config_path": arg1})))
            elif cmd == "/san":
                if arg1: console.print(Markdown(_t("compile_with_sanitizers").invoke({"path": arg1, "sanitizer": arg2 or "asan"})))
            elif cmd == "/afl":
                if arg1: console.print(Markdown(_t("run_afl").invoke({"target": arg1})))
            elif cmd == "/crash":
                if arg1: console.print(Markdown(_t("parse_crash_report").invoke({"path": arg1})))

            # ---- Binary / ASM ----
            elif cmd == "/file":
                console.print(Markdown(_t("file_info").invoke({"path": arg1})))
            elif cmd == "/elf":
                console.print(Markdown(_t("read_elf_info").invoke({"path": arg1})))
            elif cmd == "/nm":
                console.print(Markdown(_t("nm_symbols").invoke({"path": arg1})))
            elif cmd == "/disasm":
                console.print(Markdown(_t("disassemble_binary").invoke({"path": arg1})))
            elif cmd == "/asm":
                console.print(Markdown(_t("analyze_assembly").invoke({"path": arg1})))
            elif cmd == "/r2":
                console.print(Markdown(_t("run_radare2").invoke({"path": arg1})))
            elif cmd == "/rop":
                console.print(Markdown(_t("find_rop_gadgets").invoke({"binary": arg1})))

            # ---- Digər ----
            elif cmd == "/lines":
                console.print(Markdown(_t("count_lines").invoke({"path": arg1 or "."})))
            elif cmd == "/todos":
                console.print(Markdown(_t("find_todos").invoke({"path": arg1 or "."})))
            else:
                console.print("[red]❌ Naməlum əmr. /help[/red]")
            continue

        agent.ask(user_input)


# ============================================================
# 8) MAIN
# ============================================================
if __name__ == "__main__":
    import urllib.parse  # for search_cve
    persona = "hunter"
    args = sys.argv[1:]
    if "--persona" in args:
        i = args.index("--persona")
        if i + 1 < len(args): persona = args[i + 1]
    elif args and args[0] in PERSONAS:
        persona = args[0]

    if args and args[0] == "audit" and len(args) > 1:
        ProAgent(persona="hunter").audit_file(args[1]); sys.exit(0)
    if args and args[0] == "hunt":
        ProAgent(persona="hunter").hunt_surface(args[1] if len(args) > 1 else "."); sys.exit(0)
    if args and args[0] == "check":
        console.print(Markdown(_t("check_tools_available").invoke({}))); sys.exit(0)

    agent = ProAgent(persona=persona, streaming=True)
    interactive(agent)
