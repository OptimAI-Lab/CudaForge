def print_bold(text: str):
    """Prints text in bold."""
    print(f"\033[1m{text}\033[0m")

def print_warning(text: str):
    """Prints text in yellow (warning color)."""
    # 1=bold, 33=yellow
    print(f"\033[1;33m{text}\033[0m")
