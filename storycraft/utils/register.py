from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Type, TypeVar


logger = logging.getLogger(__name__)

T = TypeVar("T")


class Registry:
    """
    Universal Class Registry

    Used for:
        - Node registration
        - Tool registration
        - Agent registration
        - Pipeline registration

    Design goals:
        - Explicit lifecycle control
        - Deterministic scanning
        - Strong typing
        - Debug-friendly logging
    """

    def __init__(self, name: str = "registry"):
        self._name = name
        self._items: Dict[str, Type[Any]] = {}

    # ----------------------------------------------------------------------
    # Core APIs
    # ----------------------------------------------------------------------

    def register(self, name: Optional[str] = None, override: bool = False):
        """
        Decorator for registering classes.

        Example:
            @NODE_REGISTRY.register()
            class MyNode:
                ...

            @NODE_REGISTRY.register("clip_filter")
            class FilterNode:
                ...

        Args:
            name: Custom registration key.
            override: Whether to override existing registration.

        Returns:
            Decorated class.
        """

        def decorator(cls: Type[T]) -> Type[T]:
            reg_name = name or cls.__name__

            if reg_name in self._items and not override:
                raise KeyError(
                    f"[{self._name}] '{reg_name}' already registered. "
                    f"Use override=True to replace."
                )

            if reg_name in self._items and override:
                logger.warning(
                    f"[{self._name}] Overriding existing registration: {reg_name}"
                )

            self._items[reg_name] = cls

            logger.info(f"[{self._name}] Registered: {reg_name} -> {cls}")

            return cls

        return decorator

    def get(self, name: str, default: Optional[Type[T]] = None) -> Optional[Type[T]]:
        """
        Retrieve registered class.

        Args:
            name: Registered key.
            default: Default value if not found.

        Returns:
            Class or default.
        """
        return self._items.get(name, default)

    def require(self, name: str) -> Type[Any]:
        """
        Strict version of get(). Raises if not found.

        Useful for:
            - pipeline resolution
            - config driven construction
        """
        if name not in self._items:
            raise KeyError(
                f"[{self._name}] '{name}' not registered. "
                f"Available: {list(self._items.keys())}"
            )
        return self._items[name]

    def list(self) -> list[str]:
        """Return all registered names."""
        return list(self._items.keys())

    def values(self) -> Iterable[Type[Any]]:
        """Return all registered classes."""
        return self._items.values()

    def items(self) -> Iterable[tuple[str, Type[Any]]]:
        """Return registry items."""
        return self._items.items()

    def clear(self):
        """Clear registry."""
        logger.warning(f"[{self._name}] Registry cleared.")
        self._items.clear()

    def __len__(self):
        return len(self._items)

    def __contains__(self, name: str):
        return name in self._items

    def __repr__(self):
        return f"<Registry {self._name}: {len(self)} items>"

    # ----------------------------------------------------------------------
    # Package Scanning
    # ----------------------------------------------------------------------

    def scan_package(self, package: str | Path):
        """
        Recursively scan a package to trigger decorators.

        Args:
            package: package name string or filesystem path.

        Example:
            NODE_REGISTRY.scan_package("storycraft.nodes")
        """
        if isinstance(package, Path):
            package = self._path_to_package(package)

        logger.info(f"[{self._name}] Scanning package: {package}")

        try:
            module = importlib.import_module(package)
        except Exception as e:
            raise ImportError(f"[{self._name}] Failed to import package: {package}") from e

        if not hasattr(module, "__path__"):
            logger.warning(f"[{self._name}] {package} is not a package, skip scanning.")
            return

        for _, modname, _ in pkgutil.walk_packages(
            module.__path__, prefix=module.__name__ + "."
        ):
            try:
                importlib.import_module(modname)
                logger.debug(f"[{self._name}] Imported: {modname}")
            except Exception as e:
                logger.exception(f"[{self._name}] Failed importing: {modname}", exc_info=e)

    @staticmethod
    def _path_to_package(path: Path) -> str:
        """
        Convert filesystem path to python package path.

        Example:
            storycraft/nodes -> storycraft.nodes
        """
        return ".".join(path.parts)

  
