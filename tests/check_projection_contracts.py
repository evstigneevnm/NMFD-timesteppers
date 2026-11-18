"""A rejected projector counts as a pass only for the expected interface diagnostic."""
from pathlib import Path
import subprocess
import sys

from timestepper_ci import HERE, PROJECTION_FAMILIES


def main(directory, compiler, *includes):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    success = True
    for family in PROJECTION_FAMILIES:
        for invalid in ("BAD_RETURN", "MISSING_APPLY"):
            log = directory/f"{family}_{invalid}.log"
            command = [compiler, "-std=c++17", "-fopenmp", "-fsyntax-only", f"-DTEST_PROJECTION_{invalid}_MUST_FAIL",
                       *includes, str(HERE/f"test_{family}_projection_lifecycle.cpp")]
            with log.open("w") as out:
                result = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
            diagnostic = "Operator apply(vector&) must return void or bool"
            passed = result.returncode > 0 and diagnostic in log.read_text(errors="replace")
            print(f"{family}/{invalid}: {'passed' if passed else 'failed'} ({log})", flush=True)
            success &= passed
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
