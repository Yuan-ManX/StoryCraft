from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS


logger = logging.getLogger(__name__)


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class StorycraftRecall:
    """
    Storycraft Recall Module

    Provide:
        - Vectorstore building from structured storycraft data
        - Semantic recall based on FAISS + HF Embeddings
    """

    @staticmethod
    def _resolve_model_path(model_name: str | Path) -> str:
        """
        Resolve embedding model path.

        If local model does not exist, fallback to HuggingFace Hub model.
        """
        model_path = Path(model_name)
        if model_path.exists():
            return str(model_path)

        logger.warning(
            f"[Recall] Local model not found: {model_name}, fallback to {DEFAULT_EMBEDDING_MODEL}"
        )
        return DEFAULT_EMBEDDING_MODEL

    @classmethod
    def build_vectorstore(
        cls,
        data: List[Dict[str, Any]],
        field: str = "description",
        model_name: str | Path = "./.storycraft/models/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> Optional[FAISS]:
        """
        Build FAISS vectorstore from structured storycraft data.

        Args:
            data: List of structured dict entries.
            field: Text field to embed.
            model_name: Local or HuggingFace embedding model.
            device: "cpu" or "cuda".

        Returns:
            FAISS vectorstore or None.
        """
        if not data:
            logger.warning("[Recall] Empty dataset provided, skip building vectorstore.")
            return None

        model_name = cls._resolve_model_path(model_name)

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
        )

        docs: List[Document] = []
        skipped = 0

        for item in data:
            text = str(item.get(field, "")).strip()
            if not text:
                skipped += 1
                continue

            docs.append(
                Document(
                    page_content=text,
                    metadata=item,
                )
            )

        if not docs:
            logger.warning(
                f"[Recall] No valid text found for field='{field}', skipped={skipped}"
            )
            return None

        logger.info(
            f"[Recall] Building vectorstore: docs={len(docs)}, skipped={skipped}, model={model_name}"
        )

        return FAISS.from_documents(docs, embeddings)

    @staticmethod
    def query_top_n(
        vectorstore: FAISS,
        query: str,
        n: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Semantic recall top-N entries.

        Args:
            vectorstore: FAISS vectorstore.
            query: Query string.
            n: Top-N results.

        Returns:
            List of metadata dicts.
        """
        if not vectorstore:
            logger.warning("[Recall] vectorstore is None, returning empty result.")
            return []

        if not query.strip():
            logger.warning("[Recall] Empty query received.")
            return []

        results = vectorstore.similarity_search(query, k=n)
        return [doc.metadata for doc in results]

  
