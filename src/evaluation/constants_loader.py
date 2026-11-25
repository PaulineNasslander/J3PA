"""Read back the constants that a training run wrote.

Newer runs copy `src/constants.py` verbatim into the run directory as
``constants.py``; it is executed and its public, non-module, non-callable names are
returned. Older runs wrote a lossy ``constants_values.txt`` (``NAME = repr`` lines
plus a free-form trailing line), which is parsed best-effort.
Missing or unparsable entries fall back to the current `src/constants.py`, which
keeps old runs loadable after a constant is added.
"""

import ast
import importlib.util
import os
import re
import types

import constants as live_constants


def _load_constants_py(path: str) -> dict:
    # Not registered in sys.modules so the live `constants` module is left untouched.
    spec = importlib.util.spec_from_file_location("_run_constants", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        name: value
        for name, value in vars(module).items()
        if not name.startswith("_") and not callable(value) and not isinstance(value, types.ModuleType)
    }


def load_constants_file(path: str) -> dict:
    """Load the constants of a training run into a name -> value dict.

    Args:
        path: Path to a ``constants.py`` / ``constants_values.txt``, or to the run
            directory containing one (``constants.py`` is preferred).

    Returns:
        Dict of the constants that could be loaded. For the legacy txt format,
        values that are not Python literals (module reprs, the trailing
        debug-mode line) are skipped.
    """
    if os.path.isdir(path):
        py_path = os.path.join(path, "constants.py")
        path = py_path if os.path.isfile(py_path) else os.path.join(path, "constants_values.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No constants file at {path}")
    if path.endswith(".py"):
        return _load_constants_py(path)

    values = {}
    with open(path) as f:
        for line in f:
            name, sep, raw = line.partition(" = ")
            if not sep:
                continue
            name = name.strip()
            if not name.isidentifier():
                continue  # e.g. the appended "Ran as debug mode = False" line
            try:
                values[name] = ast.literal_eval(raw.strip())
            except (ValueError, SyntaxError):
                # Strings are written without quotes (e.g. "ENC_INPUT_VERSION = old"); module reprs are skipped.
                if re.fullmatch(r"[\w.\-]+", raw.strip()):
                    values[name] = raw.strip()
    return values


def get_constant(values: dict, name: str):
    """Look ``name`` up in a parsed constants file, falling back to the live one.

    Args:
        values: Output of :func:`load_constants_file`.
        name: Constant name.

    Returns:
        The recorded value, or the value in `src/constants.py` if the run
        predates the constant or wrote it in a form we cannot parse.
    """
    if name in values:
        return values[name]
    if hasattr(live_constants, name):
        fallback = getattr(live_constants, name)
        print(f"  [constants] {name} not found in run config, using current default {fallback!r}")
        return fallback
    raise KeyError(f"Constant {name!r} is neither in the run config nor in src/constants.py")
