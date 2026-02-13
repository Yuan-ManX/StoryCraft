from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable
import json
import time
import uuid

from storycraft.memory.file import FileCompressor
from storycraft.utils.logging import get_logger


logger = get_logger(__name__)


# ============================================================
# 🎞 Artifact Metadata Schema
# ============================================================

@dataclass
class ArtifactMeta:
    """
    Agent runtime artifact metadata.

    Represents one atomic output snapshot of a node execution.
    """
    session_id: str
    artifact_id: str
    node_id: str
    path: str
    summary: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    size_bytes: Optional[int] = None
    tags: List[str] = field(default_factory=list)


# ============================================================
# 🎛 Artifact Kernel
# ============================================================

class ArtifactStore:
    """
    StoryCraft Artifact Kernel.

    Responsibilities:
    - Node execution snapshot persistence
    - Multi-modal artifact storage
    - Artifact indexing & querying
    - Trace replay foundation
    """

    META_FILENAME = "meta.json"

    def __init__(self, artifacts_dir: str | Path, session_id: str):
        self.root = Path(artifacts_dir).expanduser().resolve()
        self.session_id = session_id
        self.session_dir = self.root / session_id
        self.meta_path = self.session_dir / self.META_FILENAME

        self.session_dir.mkdir(parents=True, exist_ok=True)

        if not self.meta_path.exists() or self.meta_path.stat().st_size == 0:
            self._save_meta_index([])

    # ============================================================
    # Meta Index
    # ============================================================

    def _load_meta_index(self) -> List[ArtifactMeta]:
        if not self.meta_path.exists():
            return []
        with self.meta_path.open("r", encoding="utf-8") as f:
            return [ArtifactMeta(**m) for m in json.load(f)]

    def _save_meta_index(self, metas: List[ArtifactMeta]):
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in metas], f, ensure_ascii=False, indent=2)

    def _append_meta(self, meta: ArtifactMeta):
        metas = self._load_meta_index()
        metas.append(meta)
        self._save_meta_index(metas)

    # ============================================================
    # Artifact ID
    # ============================================================

    def generate_artifact_id(self, node_id: str) -> str:
        return f"{node_id}_{uuid.uuid4().hex[:12]}"

    # ============================================================
    # Media Handling
    # ============================================================

    def _is_media_list(self, items: Any) -> bool:
        return isinstance(items, list) and all(isinstance(i, dict) for i in items)

    def _save_single_media(self, item: Dict[str, Any], store_dir: Path, artifact_id: str):
        base64_data = item.pop("base64", None)
        if not base64_data:
            return

        rel_path = Path(item.get("path", ""))
        file_path = store_dir / rel_path

        file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"[Artifact:{artifact_id}] saving media -> {file_path}")

        FileCompressor.decompress_from_string(base64_data, file_path)

        item["path"] = str(file_path)

    def _traverse_and_save_media(self, payload: Any, store_dir: Path, artifact_id: str):
        if isinstance(payload, dict):
            for v in payload.values():
                self._traverse_and_save_media(v, store_dir, artifact_id)
        elif self._is_media_list(payload):
            for item in payload:
                self._save_single_media(item, store_dir, artifact_id)

    # ============================================================
    # Artifact Persistence
    # ============================================================

    def save_result(
        self,
        *,
        node_id: str,
        data: Dict[str, Any],
        search_media_dir: Optional[Path] = None,
        tags: Optional[List[str]] = None,
    ) -> ArtifactMeta:
        """
        Persist one node execution snapshot.
        """
        create_time = time.time()

        artifact_id = data.get("artifact_id") or self.generate_artifact_id(node_id)
        summary = data.get("summary")
        payload = data.get("tool_excute_result")

        store_dir = self.session_dir / node_id
        store_dir.mkdir(parents=True, exist_ok=True)

        media_dir = search_media_dir or store_dir

        self._traverse_and_save_media(payload, media_dir, artifact_id)

        save_blob = {
            "payload": payload,
            "session_id": self.session_id,
            "artifact_id": artifact_id,
            "node_id": node_id,
            "created_at": create_time,
        }

        file_path = store_dir / f"{artifact_id}.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(save_blob, f, ensure_ascii=False, indent=2)

        size_bytes = file_path.stat().st_size

        logger.info(f"[Node:{node_id}] artifact persisted -> {file_path}")

        meta = ArtifactMeta(
            session_id=self.session_id,
            artifact_id=artifact_id,
            node_id=node_id,
            path=str(file_path),
            summary=summary,
            created_at=create_time,
            size_bytes=size_bytes,
            tags=tags or [],
        )

        self._append_meta(meta)
        return meta

    # ============================================================
    # Artifact Query
    # ============================================================

    def load_result(self, artifact_id: str) -> Tuple[Optional[ArtifactMeta], Any]:
        metas = self._load_meta_index()
        meta = next((m for m in metas if m.artifact_id == artifact_id), None)

        if not meta:
            return None, f"artifact `{artifact_id}` not found"

        with open(meta.path, "r", encoding="utf-8") as f:
            return meta, json.load(f)

    def list_artifacts(
        self,
        *,
        node_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ArtifactMeta]:
        metas = self._load_meta_index()

        if node_id:
            metas = [m for m in metas if m.node_id == node_id]

        if tags:
            metas = [m for m in metas if set(tags).issubset(set(m.tags))]

        return metas

    def get_latest(
        self,
        *,
        node_id: str,
    ) -> Optional[ArtifactMeta]:
        candidates = [
            m for m in self._load_meta_index()
            if m.node_id == node_id
        ]
        return max(candidates, key=lambda m: m.created_at) if candidates else None
