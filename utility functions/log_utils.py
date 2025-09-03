import os
import inspect
from datetime import datetime

# ---- Configuration ----
LOG_DIR = os.path.join(os.getcwd(), "../../../logs")
os.makedirs(LOG_DIR, exist_ok=True)


# Daily log file
LOG_FILE_PATH = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y%m%d')}.log")

# ---- Global Log Source for Notebooks ----
GLOBAL_LOG_SOURCE = None


def set_log_source(source_name: str):
    """Set the source name for Jupyter notebook logs."""
    global GLOBAL_LOG_SOURCE
    GLOBAL_LOG_SOURCE = source_name


def is_jupyter():
    """Detect if running inside Jupyter notebook."""
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except:
        return False


def get_caller_filename():
    """Return the filename of the caller."""
    if is_jupyter() and GLOBAL_LOG_SOURCE:
        return GLOBAL_LOG_SOURCE
    else:
        # Go up the call stack to find the actual calling .py file
        frame = inspect.stack()[2]
        module = inspect.getmodule(frame[0])
        if module and hasattr(module, "__file__"):
            return os.path.basename(module.__file__)
        else:
            return "Unknown"


def log(message, source=None, log_to_file=True):
    """Logs a message with timestamp and optional file/module source."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    caller = source if source else get_caller_filename()
    formatted_message = f"{timestamp} | {caller} | {message}"
    print(formatted_message)

    if log_to_file:
        with open(LOG_FILE_PATH, "a+", encoding="utf-8") as f:
            f.write(formatted_message + "\n")
