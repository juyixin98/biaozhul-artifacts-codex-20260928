"""Local synthetic target program for the delta debugging fixture.

Reads a document (canonical text form) from the file named in argv[1].

Exit protocol:
  0  PASS            document valid, trigger conditions not all met
  2  OTHER_ERROR     document malformed in a way the program rejects
  3  TARGET_ERROR    the specific bug: requires ALL of
                       mode=strict, level>=3, debug=on
  (hang)             `simulate_hang=1` sleeps forever -> the runner's
                     timeout classifies this as INDETERMINATE
"""

import sys
import time


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: target_program.py <input-file>", file=sys.stderr)
        return 2
    try:
        with open(sys.argv[1], "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
    except OSError as exc:
        print(f"OTHER_ERROR: cannot read input: {exc}", file=sys.stderr)
        return 2

    if not lines or not lines[0].startswith("BEGIN"):
        print("OTHER_ERROR: missing BEGIN header", file=sys.stderr)
        return 2
    if not lines[-1].startswith("END"):
        print("OTHER_ERROR: missing END footer", file=sys.stderr)
        return 2

    settings = {}
    for ln in lines[1:-1]:
        if "=" not in ln:
            print(f"OTHER_ERROR: malformed setting line: {ln!r}", file=sys.stderr)
            return 2
        key, _, value = ln.partition("=")
        settings[key.strip()] = value.strip()

    if settings.get("simulate_hang") == "1":
        while True:  # pragma: no cover - killed by runner timeout
            time.sleep(1)

    if "level" in settings:
        try:
            level = int(settings["level"])
        except ValueError:
            print("OTHER_ERROR: level must be an integer", file=sys.stderr)
            return 2
    else:
        level = 0

    if (
        settings.get("mode") == "strict"
        and level >= 3
        and settings.get("debug") == "on"
    ):
        print(
            "TARGET_ERROR: strict mode with level>=3 and debug=on "
            "trips the parser bug",
            file=sys.stderr,
        )
        return 3

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
