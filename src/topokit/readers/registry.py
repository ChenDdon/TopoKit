"""Format selection and extension; readers never choose a topology."""
from pathlib import Path
from typing import Protocol
from ..data import PointCloud


class Reader(Protocol):
    def __call__(self, path, **options) -> PointCloud: ...


class ReaderRegistry:
    """Independent reader registry, suitable for application-specific formats."""

    def __init__(self):
        self._readers = {}
        self._extensions = {}

    def register(self, name, reader, *, extensions=(), replace=False):
        if not isinstance(name, str) or not name.strip() or not callable(reader):
            raise ValueError("reader needs a nonempty name and a callable")
        if not isinstance(replace, bool):
            raise ValueError("replace must be a boolean")
        name = name.strip().lower()
        extensions = (extensions,) if isinstance(extensions, str) else tuple(extensions)
        if any(not isinstance(ext, str) for ext in extensions):
            raise ValueError("extensions must be filename suffixes")
        suffixes = tuple(ext.strip().lower().lstrip(".") for ext in extensions)
        if any(not ext or "/" in ext or "\\" in ext or any(char.isspace() for char in ext)
               or any(not part for part in ext.split(".")) for ext in suffixes):
            raise ValueError("extensions must be filename suffixes")
        if not replace and (name in self._readers or any(ext in self._extensions for ext in suffixes)):
            raise ValueError("reader name or extension already registered; use replace=True explicitly")
        if replace:
            self._extensions = {ext: owner for ext, owner in self._extensions.items() if owner != name}
        self._readers[name] = reader
        self._extensions.update({ext: name for ext in suffixes})

    def read(self, path, *, format=None, **options):
        if format is None:
            filename = Path(path).name.lower()
            # A plugin can register compound suffixes such as xyz.gz. Prefer
            # the most specific registered suffix without inspecting contents.
            suffix = next((ext for ext in sorted(self._extensions, key=len, reverse=True)
                           if filename.endswith("." + ext)), None)
            name = self._extensions.get(suffix)
        else:
            name = str(format).strip().lower()
        if name not in self._readers:
            raise ValueError(f"Unknown input format {format or Path(path).suffix!r}; register a reader first")
        result = self._readers[name](path, **options)
        if not isinstance(result, PointCloud):
            raise TypeError("readers must return PointCloud with scientific attributes in metadata")
        return result

    def available(self):
        return tuple(sorted(self._readers))
