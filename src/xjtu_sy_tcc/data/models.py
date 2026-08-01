"""Immutable models shared by dataset discovery and validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

from xjtu_sy_tcc.config.data import ConditionConfig


class Severity(StrEnum):
    """Severity assigned to a dataset validation issue."""

    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One structured problem found while auditing the dataset."""

    code: str
    message: str
    severity: Severity
    path: Path | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Detach and expose issue details through a read-only mapping."""

        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@dataclass(frozen=True, slots=True)
class DiscoveredAcquisition:
    """One acquisition located in a validated numeric bearing sequence."""

    condition: ConditionConfig
    bearing_id: str
    file_path: Path
    relative_path: Path
    acquisition_number: int
    sequence_index: int
    file_size_bytes: int
    modified_time_ns: int | None = None
    device_id: int | None = None
    inode: int | None = None


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Deterministic acquisitions and structural issues from discovery."""

    acquisitions: tuple[DiscoveredAcquisition, ...]
    issues: tuple[ValidationIssue, ...]
    unexpected_entries: tuple[str, ...]

    @property
    def has_errors(self) -> bool:
        """Return whether discovery found at least one critical issue."""

        return any(issue.severity is Severity.ERROR for issue in self.issues)
