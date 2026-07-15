"""
因子缓存 SQLite 元数据管理
"""
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

# ================================================================
# 常量 & 建表 SQL
# ================================================================

META_DB_NAME = "factor_cache_meta.db"

META_INIT_SQL = """
CREATE TABLE IF NOT EXISTS factor_status (
    stock_code    TEXT NOT NULL,
    factor_col    TEXT NOT NULL,
    max_time      TEXT,
    factor_name   TEXT,
    computed_at   TEXT,
    file_mtime    REAL,
    file_size     INTEGER,
    data_max_time TEXT,
    PRIMARY KEY (stock_code, factor_col)
);
CREATE TABLE IF NOT EXISTS factor_hashes (
    factor_name TEXT NOT NULL,
    code_hash   TEXT NOT NULL,
    changed_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fh_name ON factor_hashes (factor_name);
"""


# ================================================================
# 连接 & 初始化
# ================================================================

def init_meta_db(db_path: Path) -> sqlite3.Connection:
    """打开（或创建）meta 数据库并初始化表结构"""
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(META_INIT_SQL)
    return conn


# ================================================================
# factor_status CRUD
# ================================================================

def load_factor_status(
    conn: sqlite3.Connection,
) -> Dict[str, Dict[str, Tuple[str, Optional[float], Optional[int], Optional[str]]]]:
    """加载 factor_status。

    Returns:
        {stock_code: {factor_col: (max_time, file_mtime, file_size, data_max_time)}}
        max_time: 因子实际输出的最大交易日期（受 end_date 裁切影响）
        data_max_time: 源 K 线数据的最大交易日期（不受 end_date 裁切影响）
    """
    rows = conn.execute(
        "SELECT stock_code, factor_col, max_time, file_mtime, file_size, data_max_time FROM factor_status"
    ).fetchall()
    result: Dict[str, Dict[str, Tuple[str, Optional[float], Optional[int], Optional[str]]]] = {}
    for stock_code, factor_col, max_time, file_mtime, file_size, data_max_time in rows:
        result.setdefault(stock_code, {})[factor_col] = (max_time, file_mtime, file_size, data_max_time)
    return result


def save_factor_status_batch(conn: sqlite3.Connection, rows: list):
    """批量写入 factor_status 条目
    rows: [(stock_code, factor_col, max_time, factor_name, file_mtime, file_size, data_max_time), ...]"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    expanded = [
        (r[0], r[1], r[2], r[3], now, r[4], r[5], r[6])
        for r in rows
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO factor_status "
        "(stock_code, factor_col, max_time, factor_name, computed_at, file_mtime, file_size, data_max_time) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        expanded,
    )
    conn.commit()


def delete_factor_status_by_factor_name(conn: sqlite3.Connection, factor_name: str):
    """删除某个 factor_name 对应的所有 factor_status 条目（代码哈希变化时使用）"""
    conn.execute("DELETE FROM factor_status WHERE factor_name = ?", (factor_name,))
    conn.commit()


def delete_all_factor_status(conn: sqlite3.Connection):
    """清空所有 factor_status（全量重建时使用）"""
    conn.execute("DELETE FROM factor_status")
    conn.commit()


def delete_factor_status_by_col(conn: sqlite3.Connection, factor_col: str):
    """删除某个 factor_col 对应的所有 factor_status 条目（清理孤儿缓存时使用）"""
    conn.execute("DELETE FROM factor_status WHERE factor_col = ?", (factor_col,))
    conn.commit()


# ================================================================
# factor_hashes CRUD
# ================================================================

def load_meta_hashes(conn: sqlite3.Connection) -> Dict[str, str]:
    """读取每个因子最新的代码 hash（从 append-log 中取最新行）"""
    rows = conn.execute(
        "SELECT factor_name, code_hash FROM factor_hashes "
        "WHERE rowid IN (SELECT MAX(rowid) FROM factor_hashes GROUP BY factor_name)"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def save_meta_hashes(conn: sqlite3.Connection, hashes: Dict[str, str]):
    """仅 append 有变化的因子 hash（对比当前最新记录，只写入差异行）"""
    old = load_meta_hashes(conn)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    diff = [(name, h, now) for name, h in hashes.items() if old.get(name) != h]
    if diff:
        conn.executemany(
            "INSERT INTO factor_hashes (factor_name, code_hash, changed_at) VALUES (?, ?, ?)",
            diff,
        )
        conn.commit()


# ================================================================
# 文件 hash
# ================================================================

def compute_file_hash(file_path: Path) -> Optional[str]:
    """计算文件的 SHA256"""
    if file_path is None or not file_path.exists():
        return None
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
