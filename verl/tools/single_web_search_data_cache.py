import json
import logging
import os
import threading
import sqlite3
from datetime import datetime, timedelta
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray
import ray.actor

from verl.tools.utils.search_r1_like_utils_api_single_query import  perform_single_search
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


class PoolMode(Enum):
    """Execution pool mode enumeration."""
    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class SearchExecutionWorker:
    """Worker for executing search operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    logger.warning(f"Error when executing search: {e}")
        else:
            return fn(*fn_args, **fn_kwargs)


def init_search_execution_pool(
    num_workers: int,
    enable_global_rate_limit=True,
    rate_limit=10,
    mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize search execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(SearchExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class SearchTool(BaseTool):
    """Search tool with local SQLite caching to reduce API usage."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        self.num_workers = config.get("num_workers", 120)
        self.rate_limit = config.get("rate_limit", 120)
        self.timeout = config.get("timeout", 30)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_search_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )

        self.retrieval_service_url = config.get("retrieval_service_url")
        assert self.retrieval_service_url, "Configuration must include 'retrieval_service_url'"
        self.topk = config.get("topk", 5)

        # 缓存相关
        self.cache_db_path = config.get("cache_db_path", "search_cache.db")
        self.cache_expire_hours = config.get("cache_expire_hours", None)  # None 表示永不过期
        self._init_cache_db()

        if self.retrieval_service_url == "":
            raise ValueError("retrieval_service_url is not set")

        logger.info(f"Initialized SearchTool with config: {config}")

    def _init_cache_db(self):
        """初始化缓存数据库"""
        conn = sqlite3.connect(self.cache_db_path)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS search_cache (
            query TEXT PRIMARY KEY,
            result TEXT,
            metadata TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        conn.close()

    def _get_cache(self, query: str):
        """从缓存读取搜索结果"""
        conn = sqlite3.connect(self.cache_db_path)
        c = conn.cursor()
        c.execute("SELECT result, metadata, updated_at FROM search_cache WHERE query = ?", (query,))
        row = c.fetchone()
        conn.close()

        if row:
            result_text, metadata_json, updated_at = row
            metadata = json.loads(metadata_json) if metadata_json else {}
            return result_text, metadata
        return None

    def _set_cache(self, query: str, result_text: str, metadata: dict):
        """写入搜索结果到缓存"""
        conn = sqlite3.connect(self.cache_db_path)
        c = conn.cursor()
        c.execute("""
        INSERT OR REPLACE INTO search_cache (query, result, metadata, updated_at)
        VALUES (?, ?, ?, ?)
        """, (query, result_text, json.dumps(metadata), datetime.now().isoformat()))
        conn.commit()
        conn.close()
        logger.info(f"[Cache] 保存搜索结果: {query}")
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"response": "", "reward": []}
        return instance_id, ToolResponse()

    def execute_search(self, instance_id: str, query: str, retrieval_service_url: str, topk: int, timeout: int):
        result_text, metadata = perform_single_search(
            retrieval_service_url=retrieval_service_url,
            query=query,
            topk=topk,
            concurrent_semaphore=None,
            timeout=timeout
        )
        logger.debug(f"Search result for instance {instance_id}: {result_text}")
        return result_text, metadata

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        timeout = self.timeout
        query_str = parameters.get("query")

        if not query_str or not isinstance(query_str, str):
            error_msg = "Error: 'query' is missing, empty, or not a string."
            logger.error(f"[SearchTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=json.dumps({"result": error_msg})), 0.0, {}

        try:
            # 1️⃣ 检查缓存
            cache_res = self._get_cache(query_str)
            if cache_res:
                result_text, metadata = cache_res
                self._instance_dict[instance_id]["reward"].append(result_text.strip())
                return ToolResponse(text=result_text), 0.0, metadata

            # 2️⃣ 缓存未命中 → 调 API
            result_text, metadata = await self.execution_pool.execute.remote(
                self.execute_search, instance_id, query_str, self.retrieval_service_url, self.topk, timeout
            )
            self._instance_dict[instance_id]["reward"].append(result_text.strip())

            # 3️⃣ 写入缓存
            self._set_cache(query_str, result_text, metadata)

            return ToolResponse(text=result_text), 0.0, metadata

        except Exception as e:
            error_result = json.dumps({"result": f"Search execution failed: {e}"})
            logger.error(f"[SearchTool] Execution failed: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
