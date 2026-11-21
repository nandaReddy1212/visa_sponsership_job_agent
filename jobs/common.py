# jobs/common.py
import threading
from pathlib import Path

import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CONFIG_PATH = Path("config/ats_boards.yml")
VALID_ATS = {"greenhouse", "lever"}
BOARDS_PATH = Path("data/interim/ats_boards.parquet")     # accepted boards (output of discovery)

_local = threading.local()


def load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = yaml.safe_load(path.read_text())
    cfg["discovery"]["generic_words"] = set(cfg["discovery"]["generic_words"])
    cfg["overrides"] = cfg.get("overrides") or []
    cfg["rejects"] = cfg.get("rejects") or []
    for o in cfg["overrides"]:
        if o["ats"] not in VALID_ATS:
            raise ValueError(f"unknown ats '{o['ats']}' for {o['company_key']}")
    for r in cfg["rejects"]:
        if not r.get("company_key") or not r.get("slug"):
            raise ValueError(f"reject needs company_key and slug: {r}")
    return cfg


CONFIG = load_config()


def session() -> requests.Session:
    """One HTTP session per thread, with retries on 429/5xx."""
    if not hasattr(_local, "session"):
        s = requests.Session()
        retry = Retry(total=3, backoff_factor=1,
                      status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.headers["User-Agent"] = "h1b-job-agent/0.1 (portfolio project)"
        _local.session = s
    return _local.session


def get_json(url: str, timeout: int):
    """JSON body, or None on 404."""
    r = session().get(url, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()