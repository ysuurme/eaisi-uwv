"""
Centralized logging module for the EAISI UWV project.

Provides:
- setup_logging()      : One-time configuration (called from main.py)
- f_log()              : Single logging function used by all modules
- f_log_start_end()    : Decorator for function start/end timing
- f_log_execution()    : Program-level execution timer

Logging profiles (set LOG_PROFILE in config.py):
- PRD   : Emoji stage markers + warnings/errors only
- TEST  : All INFO-level messages
- DEBUG : Full execution detail with module names
"""

###############################################################################
# SYSTEM MODULES
###############################################################################

import logging
import textwrap
import time

from functools import wraps

###############################################################################
# CONFIGURATION
###############################################################################

from src.config import (
    LOG_PROFILE,
    LOG_SEPARATOR_WIDTH,
    LOG_LINE_WIDTH,
    DIR_LOG,
)

###############################################################################
# CONSTANTS
###############################################################################

# Custom log level for pipeline stage markers (between INFO=20 and WARNING=30)
STAGE = 25
logging.addLevelName(STAGE, "STAGE")

# Semantic emoji vocabulary — auto-prepended by f_log() when c_type is a key
STAGE_EMOJI = {
    "start":     "🚀",   # Pipeline or stage kickoff
    "process":   "⚙️",    # Setup, loading, training, tuning, transformation
    "success":   "✅",   # Step completed
    "store":     "💾",   # Data/model persisted to DB
    "register":  "📦",   # Model registered or promoted
    "complete":  "🎉",   # Final pipeline success
    "gate_fail": "🚫",   # Quality gate rejection
}

# Maps LOG_PROFILE to console handler level
_PROFILE_LEVELS = {
    "PRD":   STAGE,           # 25: stage markers + warning + error + critical
    "TEST":  logging.INFO,    # 20: all info messages + above
    "DEBUG": logging.DEBUG,   # 10: everything
}

# Console format per profile
_PROFILE_FORMATS = {
    "PRD":   "%(message)s",
    "TEST":  "%(asctime)s - %(levelname)s - %(message)s",
    "DEBUG": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
}

# Maps standard c_type strings to logging levels
_TYPE_TO_LEVEL = {
    "debug":    logging.DEBUG,
    "info":     logging.INFO,
    "warning":  logging.WARNING,
    "error":    logging.ERROR,
    "critical": logging.CRITICAL,
}

# Module-level flag to prevent double configuration
_is_configured = False

# Logger instance used by all f_log calls
_logger = logging.getLogger("eaisi_uwv")

###############################################################################
# FORMATTER
###############################################################################

class IndentedFormatter(logging.Formatter):
    """
    Custom formatter that wraps long lines and indents continuation lines
    for readability in log files.
    """

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)

        # Split by the standard " - " delimiter to find message start
        parts = formatted.split(" - ", 2)
        if len(parts) < 3 or len(formatted) <= LOG_LINE_WIDTH:
            return formatted

        prefix = f"{parts[0]} - {parts[1]} - "
        message = parts[2]
        available_width = LOG_LINE_WIDTH - len(prefix)

        wrapped = textwrap.fill(
            message,
            width=available_width,
            initial_indent="",
            subsequent_indent=" " * len(prefix),
            break_long_words=False,
            break_on_hyphens=False,
        )

        # Re-attach prefix to first line only
        lines = wrapped.split("\n")
        lines[0] = prefix + lines[0]
        return "\n".join(lines)


###############################################################################
# SETUP
###############################################################################

def setup_logging(profile: str = None) -> None:
    """
    One-time logging configuration. Call from main.py before any f_log() calls.

    Sets up two handlers:
    - Console: level and format determined by LOG_PROFILE
    - File: always captures DEBUG-level to log/application.log

    Parameters
    ----------
    profile : str, optional
        Override for LOG_PROFILE from config.py. One of "PRD", "TEST", "DEBUG".
    """
    global _is_configured

    if _is_configured:
        return

    active_profile = profile or LOG_PROFILE

    # Validate profile
    if active_profile not in _PROFILE_LEVELS:
        active_profile = "PRD"

    # Clear any existing handlers on root logger to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Set root logger to DEBUG so handlers control filtering
    logging.root.setLevel(logging.DEBUG)

    # --- Console Handler ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(_PROFILE_LEVELS[active_profile])
    console_handler.setFormatter(logging.Formatter(
        fmt=_PROFILE_FORMATS[active_profile],
        datefmt="%H:%M:%S",
    ))
    logging.root.addHandler(console_handler)

    # --- File Handler ---
    DIR_LOG.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        filename=DIR_LOG / "application.log",
        mode="a",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(IndentedFormatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.root.addHandler(file_handler)

    # --- Suppress noisy third-party loggers ---
    for noisy_logger in ("mlflow", "urllib3", "git", "docker"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _is_configured = True


def _ensure_setup() -> None:
    """Lazy initialization guard for standalone script execution."""
    if not _is_configured:
        setup_logging()


###############################################################################
# PUBLIC API
###############################################################################

def f_log(
    c_message: str,
    c_type:    str  = "info",
    b_raise:   bool = False,
    c_before:  str  = None,
    c_after:   str  = None,
) -> None:
    """
    Central logging function for all project modules.

    Parameters
    ----------
    c_message : str
        Message to log.
    c_type : str
        Log type. Standard: "debug", "info", "warning", "error", "critical".
        Stage (emoji auto-prepended): "start", "process", "success",
        "store", "register", "complete", "gate_fail".
    b_raise : bool
        Raise Exception after logging error/critical messages.
    c_before : str
        Character for separator line before the message (e.g. "=", "-").
    c_after : str
        Character for separator line after the message (e.g. "=", "-").
    """
    _ensure_setup()

    # Determine log level and format message
    if c_type in STAGE_EMOJI:
        level = STAGE
        formatted_message = f"{STAGE_EMOJI[c_type]} {c_message}"
    elif c_type in _TYPE_TO_LEVEL:
        level = _TYPE_TO_LEVEL[c_type]
        formatted_message = c_message
    else:
        # Fallback: unknown c_type defaults to INFO
        level = logging.INFO
        formatted_message = c_message

    # Log separator before
    if c_before is not None:
        _logger.log(level, c_before * LOG_SEPARATOR_WIDTH)

    # Log the message
    _logger.log(level, formatted_message)

    # Log separator after
    if c_after is not None:
        _logger.log(level, c_after * LOG_SEPARATOR_WIDTH)

    # Raise exception for error/critical when requested
    if c_type in ("error", "critical") and b_raise:
        raise Exception(c_message)


###############################################################################
# DECORATORS
###############################################################################

def f_log_start_end(c_separator: str = "-"):
    """
    Decorator to log the start and end of a function call.

    Parameters
    ----------
    c_separator : str
        Character for separator lines around the log messages.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            f_log(f"Start {func.__name__}()", c_before=c_separator)
            result = func(*args, **kwargs)
            f_log(f"Ended {func.__name__}()", c_after=c_separator)
            return result
        return wrapper
    return decorator


###############################################################################
# EXECUTION TIMER
###############################################################################

# Stores start times per project for f_log_execution
_start_times: dict[str, float] = {}


def f_log_execution(c_project: str, b_start: bool = True) -> None:
    """
    Log start or end of a program execution with elapsed time.

    Parameters
    ----------
    c_project : str
        Project name (displayed in uppercase).
    b_start : bool
        True to log start and begin timing; False to log end with elapsed time.
    """
    action = "Start" if b_start else "End"

    if b_start:
        _start_times[c_project] = time.time()
        f_log(
            f"{action} of the {c_project.upper()} program.",
            c_before="=", c_after="=",
        )
        return

    f_log(
        f"{action} of the {c_project.upper()} program.",
        c_before="=", c_after="-",
    )

    if c_project not in _start_times:
        f_log(
            "No start time recorded. Call f_log_execution with b_start=True first.",
            c_type="warning", c_after="=",
        )
        return

    elapsed = time.time() - _start_times.pop(c_project)
    f_log(
        f"Total execution time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)",
        c_after="=",
    )