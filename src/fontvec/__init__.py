import sys


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: fontvec [path/to/fonts]", file=sys.stderr)
        exit(1)

    folder = sys.argv[1]
