"""Start the independent 15-minute scheduler. Run in its own terminal; Ctrl+C to stop.

    python scripts/run_scheduler.py
    python scripts/run_scheduler.py --no-immediate   # wait for the first interval
"""

import argparse

import _bootstrap  # noqa: F401

from voterpulse.config import get_config
from voterpulse.logging_utils import setup_from_config
from voterpulse.monitoring.scheduler import run_forever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-immediate", action="store_true", help="do not run once at start-up")
    args = parser.parse_args()
    setup_from_config()
    run_forever(get_config(), run_immediately=not args.no_immediate)


if __name__ == "__main__":
    main()
