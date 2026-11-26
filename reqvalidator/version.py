"""Определение версии агента для паспорта прогона: семантическая версия + коммит git."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from reqvalidator import __version__

_REPO_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AgentVersion:
    version: str
    commit: str | None = None
    tag: str | None = None
    dirty: bool = False

    @property
    def label(self) -> str:
        """Строка для паспорта: «1.0.0, коммит 1a2b3c4d5e6f (тег v1.0.0)»."""
        parts = [self.version]
        if self.commit:
            parts.append(f"коммит {self.commit}")
        else:
            parts.append("коммит не определён")
        label = ", ".join(parts)
        if self.tag:
            label += f" (тег {self.tag})"
        if self.dirty:
            label += " + незакоммиченные изменения кода"
        return label


def _git(*args: str, strip: bool = True) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=_REPO_DIR,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # Для `git status --porcelain` ведущий пробел значим (« M путь»), поэтому его не срезаем.
    return out.stdout.strip() if strip else out.stdout.rstrip("\n")


def agent_version(ignore_paths: list[Path] | tuple[Path, ...] = ()) -> AgentVersion:
    """Вернуть версию агента.

    ``ignore_paths`` — файлы, изменение которых не считается изменением кода агента
    (например, перезаписываемый отчёт). Если агент установлен не из git-клона,
    коммит можно передать через переменную окружения ``REQVALIDATOR_COMMIT``.
    """
    env_commit = os.getenv("REQVALIDATOR_COMMIT")
    commit = _git("rev-parse", "--short=12", "HEAD")
    if commit is None:
        return AgentVersion(__version__, commit=env_commit or None)

    tag = _git("describe", "--tags", "--exact-match", "HEAD") or None
    status = _git("status", "--porcelain", "--untracked-files=no", strip=False) or ""
    return AgentVersion(__version__, commit=commit, tag=tag, dirty=_is_dirty(status, ignore_paths))


def _is_dirty(status: str, ignore_paths: list[Path] | tuple[Path, ...] = ()) -> bool:
    """Есть ли изменения кода агента: строки `git status --porcelain`, кроме отчётов в reports/."""
    ignored = set()
    for p in ignore_paths:
        try:
            ignored.add(Path(p).resolve().relative_to(_REPO_DIR).as_posix())
        except ValueError:
            continue
    for line in status.splitlines():
        path = line[3:].strip().strip('"')  # формат porcelain: «XY путь»
        if " -> " in path:  # переименование
            path = path.split(" -> ", 1)[1]
        if path and path not in ignored and not path.startswith("reports/"):
            return True
    return False
