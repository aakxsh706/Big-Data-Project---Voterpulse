"""Create or upgrade the SQLite database(s).

    python scripts/init_db.py            # live database
    python scripts/init_db.py --demo     # separate demo database
"""

import argparse

import _bootstrap  # noqa: F401

from voterpulse.config import get_config
from voterpulse.database import init_db, schema_version
from voterpulse.logging_utils import fields, get_logger, setup_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", action="store_true", help="initialise the demo database instead")
    args = parser.parse_args()
    setup_from_config()
    cfg = get_config()
    path = cfg.demo_db_path if args.demo else cfg.live_db_path
    conn = init_db(path, "demo" if args.demo else "live")
    get_logger("scripts.init_db").info("database_ready", extra=fields(path=path, schema_version=schema_version(conn)))
    conn.close()


if __name__ == "__main__":
    main()
