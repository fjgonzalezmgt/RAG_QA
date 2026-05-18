"""Initialize the Quality Intelligence PostgreSQL schema.

This script applies ``sql/001_quality_intelligence_schema.sql`` to the database
configured in ``.env``. Use ``--db-name RAG_DB`` when the local ``.env`` points
to another database.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_books.config import get_settings


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Initialize the Quality Intelligence schema.")
    parser.add_argument("--db-name", help="Target PostgreSQL database name. Defaults to DB_NAME from .env.")
    parser.add_argument(
        "--create-db",
        action="store_true",
        help="Create the target database first when it does not exist.",
    )
    parser.add_argument(
        "--maintenance-db",
        default="postgres",
        help="Database used to create --db-name when --create-db is set.",
    )
    parser.add_argument(
        "--sql-file",
        type=Path,
        default=ROOT / "sql" / "001_quality_intelligence_schema.sql",
        help="SQL migration file to execute.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the schema migration."""

    args = parse_args()
    settings = get_settings()
    db = settings.db
    if args.db_name:
        db = replace(db, name=args.db_name)

    sql_path = args.sql_file.resolve()
    sql_text = sql_path.read_text(encoding="utf-8")
    sql_text = sql_text.replace(
        "extensions.vector(2000)",
        f"extensions.vector({settings.openai.embedding_dim})",
    )

    print(f"Applying {sql_path}")
    print(f"Database: {db.name}")
    print("Schema: quality_intelligence")

    try:
        if args.create_db:
            ensure_database(db, maintenance_db=args.maintenance_db)

        with psycopg.connect(
            host=db.host,
            port=db.port,
            dbname=db.name,
            user=db.user,
            password=db.password,
            sslmode=db.sslmode,
            row_factory=dict_row,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_text)
            conn.commit()
    except psycopg.errors.InsufficientPrivilege as exc:
        print(f"Cannot create or modify the database with the configured user: {exc}")
        return 2
    except psycopg.OperationalError as exc:
        print(f"Cannot connect to database '{db.name}': {exc}")
        return 2

    print("Quality Intelligence schema is ready.")
    return 0


def ensure_database(db, maintenance_db: str) -> None:
    """Create the target database if it does not exist."""

    with psycopg.connect(
        host=db.host,
        port=db.port,
        dbname=maintenance_db,
        user=db.user,
        password=db.password,
        sslmode=db.sslmode,
        row_factory=dict_row,
        autocommit=True,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db.name,))
            if cur.fetchone():
                print(f"Database already exists: {db.name}")
                return
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db.name)))
            print(f"Database created: {db.name}")


if __name__ == "__main__":
    raise SystemExit(main())
