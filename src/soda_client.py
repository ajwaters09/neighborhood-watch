"""A thin client for the Chicago Data Portal's Socrata (SODA) API.

One client covers every portal dataset the project reads: crimes (`ijzp-q8t2`), 311 service
requests (`v6vf-nfxy`), and the CDOT and Park District permits. An API key ID/secret pair (HTTP
basic auth) lifts the anonymous throttling. Notebooks pass it from the secret scope; locally it
comes from SOCRATA_API_KEY_ID / SOCRATA_API_KEY_SECRET (.env.example).

Pure Python (requests only), so it works anywhere, including the tutorials.
"""

from __future__ import annotations

import os
import time
from typing import Any, Iterator

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:      # not installed on Databricks, where keys come from the secret scope
    pass

DOMAIN = "data.cityofchicago.org"
BASE_URL = f"https://{DOMAIN}/resource"

CRIMES_DATASET_ID = "ijzp-q8t2"
SERVICE_REQUESTS_DATASET_ID = "v6vf-nfxy"

DEFAULT_PAGE_SIZE = 50_000
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


class SocrataClient:
    def __init__(
        self,
        key_id: str | None = None,
        key_secret: str | None = None,
        timeout: int = 60,
    ) -> None:
        self.key_id = key_id or os.environ.get("SOCRATA_API_KEY_ID")
        self.key_secret = key_secret or os.environ.get("SOCRATA_API_KEY_SECRET")
        self.timeout = timeout
        self.session = requests.Session()
        if self.key_id and self.key_secret:
            self.session.auth = (self.key_id, self.key_secret)

    def _url(self, dataset_id: str) -> str:
        return f"{BASE_URL}/{dataset_id}.json"

    def _request(self, dataset_id: str, params: dict[str, Any]) -> list[dict]:
        url = self._url(dataset_id)
        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 429:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                    continue
                if 400 <= resp.status_code < 500:
                    # Bad SoQL or a bad dataset id won't fix itself: fail now, with Socrata's message.
                    raise RuntimeError(
                        f"SODA request to {dataset_id} rejected ({resp.status_code}): {resp.text[:500]}\n"
                        f"params: {params}"
                    )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        raise RuntimeError(f"SODA request to {dataset_id} failed after {MAX_RETRIES} attempts") from last_exc

    def query(
        self,
        dataset_id: str,
        where: str | None = None,
        select: str | None = None,
        order: str | None = None,
        group: str | None = None,
        limit: int | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> Iterator[dict]:
        """Yield a dataset's rows, paging with $limit/$offset.

        `where`, `select`, `order` and `group` are raw SoQL clause bodies (where="year=2025").
        `limit` caps the total rows; None means every matching row.
        """
        offset = 0
        returned = 0
        while True:
            page_limit = page_size
            if limit is not None:
                page_limit = min(page_size, limit - returned)
                if page_limit <= 0:
                    return
            params: dict[str, Any] = {"$limit": page_limit, "$offset": offset}
            if where:
                params["$where"] = where
            if select:
                params["$select"] = select
            if order:
                params["$order"] = order
            if group:
                params["$group"] = group

            rows = self._request(dataset_id, params)
            if not rows:
                return
            for row in rows:
                yield row
            returned += len(rows)
            offset += len(rows)
            if len(rows) < page_limit:
                return

    def count(self, dataset_id: str, where: str | None = None) -> int:
        params: dict[str, Any] = {"$select": "count(*) as c"}
        if where:
            params["$where"] = where
        rows = self._request(dataset_id, params)
        return int(rows[0]["c"]) if rows else 0

    # -- dataset shortcuts --------------------------------------------------

    def get_crimes(
        self,
        where: str | None = None,
        select: str | None = None,
        order: str = "date DESC",
        limit: int | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> Iterator[dict]:
        return self.query(CRIMES_DATASET_ID, where=where, select=select, order=order, limit=limit, page_size=page_size)

    def get_service_requests(
        self,
        where: str | None = None,
        select: str | None = None,
        order: str = "created_date DESC",
        limit: int | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> Iterator[dict]:
        return self.query(
            SERVICE_REQUESTS_DATASET_ID, where=where, select=select, order=order, limit=limit, page_size=page_size
        )
