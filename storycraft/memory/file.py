#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import gzip
import base64
import zlib
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Union, Optional, Literal

from storycraft.utils.logging import get_logger


logger = get_logger(__name__)


# ============================================================
# 🎛 Artifact Transport Packet
# ============================================================

CompressionMethod = Literal["gzip", "zlib"]


@dataclass
class CompressedFile:
    """
    Artifact transport packet.

    Represents a fully encoded transferable binary payload with integrity
    and compression metadata.
    """
    filename: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    method: CompressionMethod
    md5: str
    base64: str
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.filename} | "
            f"{self.original_size} → {self.compressed_size} bytes | "
            f"{self.compression_ratio:.2f}% | {self.method}"
        )


# ============================================================
# 🎛 Artifact Transport Kernel
# ============================================================

class FileCompressor:
    """
    StoryCraft Artifact Transport Kernel.

    Responsibilities:
    - Binary payload compression
    - Base64 encoding
    - Integrity verification
    - Artifact transport packaging
    - Persistent reconstruction
    """

    DEFAULT_METHOD: CompressionMethod = "gzip"

    # ============================================================
    # Hash Utilities
    # ============================================================

    @staticmethod
    def md5(data: bytes) -> str:
        return hashlib.md5(data).hexdigest()

    # ============================================================
    # Encode Path → Transport Packet
    # ============================================================

    @classmethod
    def compress_and_encode(
        cls,
        file_path: Union[str, Path],
        method: CompressionMethod = DEFAULT_METHOD,
    ) -> CompressedFile:
        """
        Compress and encode a file into transferable artifact packet.
        """
        file_path = Path(file_path).expanduser().resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        original_data = file_path.read_bytes()

        original_size = len(original_data)
        original_md5 = cls.md5(original_data)

        compressed = cls._compress(original_data, method)

        encoded = base64.b64encode(compressed).decode("utf-8")

        ratio = (1 - len(compressed) / original_size) * 100 if original_size else 0

        packet = CompressedFile(
            filename=file_path.name,
            original_size=original_size,
            compressed_size=len(compressed),
            compression_ratio=ratio,
            method=method,
            md5=original_md5,
            base64=encoded,
        )

        logger.debug(f"[Artifact Transport] {packet.summary()}")
        return packet

    # ============================================================
    # Decode Packet → Raw Bytes
    # ============================================================

    @classmethod
    def decode_and_decompress(
        cls,
        packet: CompressedFile,
        output_path: Optional[Union[str, Path]] = None,
    ) -> bytes:
        """
        Decode and reconstruct original binary data.
        """
        compressed_data = base64.b64decode(packet.base64)
        raw_data = cls._decompress(compressed_data, packet.method)

        decoded_md5 = cls.md5(raw_data)
        if decoded_md5 != packet.md5:
            raise ValueError(
                f"MD5 checksum mismatch: expected {packet.md5}, got {decoded_md5}"
            )

        if output_path:
            cls._write_file(raw_data, output_path)

        return raw_data

    # ============================================================
    # JSON Persistence (Trace Replay Support)
    # ============================================================

    @staticmethod
    def save_to_json(packet: CompressedFile, json_path: Union[str, Path]):
        json_path = Path(json_path).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(packet), f, ensure_ascii=False, indent=2)

        logger.info(f"[Artifact Transport] packet saved → {json_path}")

    @staticmethod
    def load_from_json(json_path: Union[str, Path]) -> CompressedFile:
        json_path = Path(json_path).expanduser().resolve()

        if not json_path.exists():
            raise FileNotFoundError(f"Packet file not found: {json_path}")

        with json_path.open("r", encoding="utf-8") as f:
            return CompressedFile(**json.load(f))

    # ============================================================
    # Direct Decode + Write (Fast Path)
    # ============================================================

    @classmethod
    def decompress_from_string(
        cls,
        encoded_string: str,
        output_path: Union[str, Path],
        method: CompressionMethod = DEFAULT_METHOD,
    ) -> bytes:
        """
        Directly decode base64 + decompress + persist.
        """
        compressed = base64.b64decode(encoded_string)
        raw_data = cls._decompress(compressed, method)

        cls._write_file(raw_data, output_path)
        return raw_data

    # ============================================================
    # Internal Compression Core
    # ============================================================

    @staticmethod
    def _compress(data: bytes, method: CompressionMethod) -> bytes:
        if method == "gzip":
            return gzip.compress(data)
        if method == "zlib":
            return zlib.compress(data)
        raise ValueError(f"Unsupported compression method: {method}")

    @staticmethod
    def _decompress(data: bytes, method: CompressionMethod) -> bytes:
        if method == "gzip":
            return gzip.decompress(data)
        if method == "zlib":
            return zlib.decompress(data)
        raise ValueError(f"Unsupported compression method: {method}")

    @staticmethod
    def _write_file(data: bytes, path: Union[str, Path]):
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        logger.debug(f"[Artifact Transport] write → {path}")

  
