
import json
import logging
import threading
import time
import uuid
from typing import Any, Optional
import requests
import os

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1

import json
import logging
import threading
import time
import uuid
from typing import Any, Optional
import requests
import os

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1

def call_search_api(
    retrieval_service_url: str,
    query: str,
    topk: int = 5,
    timeout: int = DEFAULT_TIMEOUT
) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:

    IDEALAB_AK = os.getenv("Web_search_AK", "")
    IDEALAB_URL = retrieval_service_url 

    if not IDEALAB_AK:
        return None, "Idealab API Key 未配置，请设置环境变量 IDEALAB_AK"

    headers = {
        "X-AK": IDEALAB_AK,
        "Content-Type": "application/json"
    }

    results = []
    error_msg = None
    request_id = str(uuid.uuid4())
    log_prefix = f"[Idealab Request ID: {request_id}] "

    for attempt in range(MAX_RETRIES):
        try:
            payload = {
                "query": query,
                "num": topk,
                "extendParams": {"country": "us", "locale": "en-us"},
                "platformInput": {"model": "google-search"}
            }
            logger.info(f"{log_prefix}Attempt {attempt+1}/{MAX_RETRIES}: Query='{query}'")
            resp = requests.post(IDEALAB_URL, json=payload, headers=headers, timeout=timeout)

            if resp.status_code in [500, 502, 503, 504]:
                error_msg = f"{log_prefix}Server Error {resp.status_code}"
                logger.warning(error_msg)
                continue    

            resp.raise_for_status()

            try:
                data = resp.json()
            except json.JSONDecodeError:
                error_msg = f"{log_prefix}JSON Decode Error: {resp.text[:200]}"
                continue

            organic_list = data.get("data", {}).get("originalOutput", {}).get("organic", [])
            for item in organic_list:
                results.append({
                    "title": item.get("title", "") or item.get("link", "") or "<No Title>",
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "date": item.get("date", ""),
                })

            if results:
                logger.info(f"{log_prefix}Search successful on attempt {attempt+1}")
                return results, None

        except requests.RequestException as e:
            error_msg = f"{log_prefix}Request Error: {e}"

        # 重试等待
        if attempt < MAX_RETRIES - 1:
            delay = INITIAL_RETRY_DELAY * (attempt + 1)
            logger.info(f"{log_prefix}Retrying after {delay} seconds...")
            time.sleep(delay)

    logger.error(f"{log_prefix}Failed after retries. Last error: {error_msg}")
    return None, error_msg or "API Call Failed after retries"



def perform_single_search(
    retrieval_service_url: str,
    query: str,
    topk: int = 5,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[str, dict[str, Any]]:
    results_list, error_msg = call_search_api(
        retrieval_service_url=retrieval_service_url,
        query=query,
        topk=topk,
        timeout=timeout,
    )

    metadata = {
        "query_count": 1,
        "queries": [query],
        "api_request_error": error_msg,
        "status": "unknown",
        "total_results": 0,
        "formatted_result": None,
        "api_response": None
    }

    if error_msg:
        readable_text = f"Search error: {error_msg}"
        metadata["status"] = "api_error"
    elif results_list:
        readable_text = "\n\n".join(
            f"[{i}] \"{doc['title']}\n{doc['snippet']}\" {doc['date']}\n{doc['url']}"
            for i, doc in enumerate(results_list, 1)
        )
        metadata["status"] = "success"
        metadata["total_results"] = len(results_list)
        metadata["formatted_result"] = readable_text
    else:
        readable_text = "No results found."
        metadata["status"] = "no_results"

    return readable_text, metadata


if __name__ == "__main__":
    results, meta = perform_single_search(
        retrieval_service_url="",
        query="Hotel Argentina architect",
        topk=5
    )
    print(results)
    print(meta)
