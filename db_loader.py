import os
import re
import sqlite3
from typing import List, Tuple


VALID_CHINESE_RE = re.compile(r"^[\u4e00-\u9fff]+$")
BLACKLIST_TOKENS = {"某", "不知", "无名"}


def get_cbdb_names(
    db_path: str = "latest.db", dynasty_id: int | None = None, limit: int | None = None
) -> Tuple[List[str], List[str]]:
    """
    Load and clean surname + given-name pairs from a CBDB SQLite database.
    """
    if not os.path.exists(db_path):
        return [], []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    except sqlite3.Error:
        return [], []

    sql = """
    SELECT c_surname_chn, c_mingzi_chn
    FROM BIOG_MAIN
    WHERE c_mingzi_chn IS NOT NULL AND c_mingzi_chn != ''
      AND c_surname_chn IS NOT NULL AND c_surname_chn != ''
    """

    params = []
    if dynasty_id is not None:
        sql += " AND c_dy = ?"
        params.append(dynasty_id)

    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    try:
        cursor.execute(sql, params)
        rows = cursor.fetchall()
    except sqlite3.Error:
        conn.close()
        return [], []

    conn.close()

    cleaned_surnames: List[str] = []
    cleaned_names: List[str] = []

    for surname, name in rows:
        if not (isinstance(surname, str) and isinstance(name, str)):
            continue

        if any(token in surname for token in BLACKLIST_TOKENS):
            continue
        if any(token in name for token in BLACKLIST_TOKENS):
            continue

        if VALID_CHINESE_RE.fullmatch(surname) and VALID_CHINESE_RE.fullmatch(name):
            cleaned_surnames.append(surname)
            cleaned_names.append(name)

    return cleaned_surnames, cleaned_names
