import ast
import asyncio
import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Iterable, List, Optional

from dmSQLAlchemy import CollectionSchema, DataType, FieldSchema, dmVecClient
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from qanything_kernel.configs.model_config import (
    DAMENG_DATABASE_LOCAL,
    DAMENG_HOST_LOCAL,
    DAMENG_PASSWORD_LOCAL,
    DAMENG_PORT_LOCAL,
    DAMENG_USER_LOCAL,
    MILVUS_COLLECTION_NAME,
)
from qanything_kernel.connector.embedding.embedding_for_online_client import YouDaoEmbeddings
from qanything_kernel.utils.custom_log import debug_logger, insert_logger
from qanything_kernel.utils.general_utils import get_time


class SelfDMVectorStore(VectorStore):
    def __init__(self, *args, **kwargs):
        self.embedding_func = kwargs.get("embedding_function")
        if self.embedding_func is None:
            raise ValueError("embedding_function is required")

        connection_args = kwargs.get("connection_args") or {}
        host = connection_args.get("host", DAMENG_HOST_LOCAL)
        port = connection_args.get("port", DAMENG_PORT_LOCAL)

        self._dm_uri = connection_args.get("uri", f"{host}:{port}")
        self._dm_user = connection_args.get("user", DAMENG_USER_LOCAL)
        self._dm_password = connection_args.get("password", DAMENG_PASSWORD_LOCAL)
        self._dm_db_name = connection_args.get("db_name", DAMENG_DATABASE_LOCAL)
        self.timeout = kwargs.get("timeout", 10)

        self.collection_name = kwargs.get("collection_name", MILVUS_COLLECTION_NAME)
        self.metric_type = (
            (kwargs.get("search_params") or {}).get("metric_type")
            or kwargs.get("metric_type")
            or "COSINE"
        ).upper()

        self.auto_id = kwargs.get("auto_id", True)

        self._primary_field = kwargs.get("primary_field", "id")
        self._text_field = kwargs.get("text_field", "text")
        self._vector_field = kwargs.get("vector_field", "vector")
        self._doc_id_field = "doc_id"
        self._kb_id_field = "kb_id"
        self._file_id_field = "file_id"
        self._metadata_field = "metadata_json"

        self.last_flush_time = 0.0
        self.inserted_since_last_flush = 0
        self.flush_interval = 600
        self.flush_threshold = 10000

        self._vector_dim: Optional[int] = None
        self._collection_ready = False
        self._create_lock = threading.Lock()
        self._client_local = threading.local()

    @property
    def embeddings(self):
        return self.embedding_func

    def _run_coro_sync(self, coro):
        try:
            asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(asyncio.run, coro).result()
        except RuntimeError:
            return asyncio.run(coro)

    def _get_client(self) -> dmVecClient:
        client = getattr(self._client_local, "client", None)
        if client is None:
            client = dmVecClient(
                uri=self._dm_uri,
                user=self._dm_user,
                password=self._dm_password,
                db_name=self._dm_db_name,
                timeout=self.timeout,
            )
            self._client_local.client = client
        return client

    def _check_collection_exists(self) -> bool:
        client = self._get_client()
        exists = client.has_collection(self.collection_name)
        if exists:
            self._collection_ready = True
        return exists

    def _ensure_collection(self, vector_dim: Optional[int] = None) -> None:
        if self._collection_ready:
            return

        with self._create_lock:
            if self._collection_ready:
                return

            client = self._get_client()
            if client.has_collection(self.collection_name):
                self._collection_ready = True
                return

            if vector_dim is None:
                raise ValueError("Vector dimension is required when creating a new DM collection.")

            fields = [
                FieldSchema(self._primary_field, DataType.VARCHAR, is_primary=True, auto_id=False),
                FieldSchema(self._doc_id_field, DataType.VARCHAR),
                FieldSchema(self._kb_id_field, DataType.VARCHAR),
                FieldSchema(self._file_id_field, DataType.VARCHAR),
                FieldSchema(self._text_field, DataType.TEXT),
                FieldSchema(self._metadata_field, DataType.TEXT),
                FieldSchema(self._vector_field, DataType.FLOAT_VECTOR, dim=vector_dim),
            ]
            schema = CollectionSchema(fields=fields, description="QAnything DM vector collection")
            index_params = client.prepare_index_params()
            index_name = f"{self.collection_name}_{self._vector_field}_hnsw_idx"
            index_params.add_index(
                field_name=self._vector_field,
                index_type="HNSW",
                index_name=index_name,
                metric_name=self.metric_type,
            )
            client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                metric_type=self.metric_type,
                index_params=index_params,
            )
            client.commit()
            self._collection_ready = True
            self._vector_dim = vector_dim
            debug_logger.info(
                f"created DM vector collection: {self.collection_name}, dim={vector_dim}, metric={self.metric_type}"
            )

    def _load_table(self):
        try:
            return self._get_client().load_table(self.collection_name)
        except Exception:
            return None

    def _parse_literal(self, value: str) -> Any:
        text = value.strip()
        if not text:
            return text

        try:
            return ast.literal_eval(text)
        except Exception:
            pass

        low = text.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        return text

    def _build_filter(self, expr: Optional[str], table) -> Optional[List[Any]]:
        if not expr or not expr.strip():
            return None

        clauses = []
        conds = re.split(r"\s+and\s+", expr.strip(), flags=re.IGNORECASE)

        for cond in conds:
            cond = cond.strip()
            if not cond:
                continue

            in_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s+in\s+(.+)$", cond, flags=re.IGNORECASE)
            eq_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*==\s*(.+)$", cond)

            if in_match:
                field = in_match.group(1)
                raw_value = in_match.group(2)
                if field not in table.c:
                    raise ValueError(f"Unsupported filter field: {field}")
                parsed = self._parse_literal(raw_value)
                if isinstance(parsed, (list, tuple, set)):
                    values = list(parsed)
                else:
                    values = [parsed]
                clauses.append(table.c[field].in_(values))
                continue

            if eq_match:
                field = eq_match.group(1)
                raw_value = eq_match.group(2)
                if field not in table.c:
                    raise ValueError(f"Unsupported filter field: {field}")
                clauses.append(table.c[field] == self._parse_literal(raw_value))
                continue

            raise ValueError(f"Unsupported expression condition: {cond}")

        return clauses or None

    def _normalize_vector(self, vector: List[float]) -> List[float]:
        return [float(v) for v in vector]

    def _row_to_document(self, row: dict) -> Document:
        metadata = {}
        raw_metadata = row.get(self._metadata_field)
        if isinstance(raw_metadata, dict):
            metadata.update(raw_metadata)
        elif isinstance(raw_metadata, str) and raw_metadata:
            try:
                metadata.update(json.loads(raw_metadata))
            except Exception:
                metadata = {}

        for field in (self._doc_id_field, self._kb_id_field, self._file_id_field):
            value = row.get(field)
            if value is not None and field not in metadata:
                metadata[field] = value

        content = row.get(self._text_field) or ""
        return Document(page_content=content, metadata=metadata)

    def _should_flush(self) -> bool:
        return False

    @get_time
    def _milvus_flush(self):
        self.last_flush_time = time.time()
        self.inserted_since_last_flush = 0
        insert_logger.info(f"DM vectorstore flush noop at {self.last_flush_time}")

    def get_expr_result(self, expr: str, output_fields: List[str]) -> List[dict] | None:
        if not self._check_collection_exists():
            debug_logger.debug("No existing collection to query.")
            return None

        table = self._load_table()
        if table is None:
            return None

        filters = self._build_filter(expr, table)
        return self._get_client().query(
            collection_name=self.collection_name,
            filter=filters,
            output_fields=output_fields,
            timeout=self.timeout,
        )

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        timeout: Optional[int] = None,
        batch_size: int = 1000,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        time_record = kwargs.get("time_record", {})
        texts = list(texts)

        if not texts:
            insert_logger.info("Nothing to insert, skipping.")
            return []

        if metadatas is None:
            metadatas = [{} for _ in texts]
        if len(metadatas) != len(texts):
            raise ValueError("metadatas length must be equal to texts length")

        if not self.auto_id:
            assert isinstance(ids, list), "A list of valid ids are required when auto_id is False."
            assert len(set(ids)) == len(texts), "Different lengths of texts and unique ids are provided."
        elif ids is not None and len(ids) != len(texts):
            raise ValueError("ids length must be equal to texts length")

        embedding_start = time.perf_counter()
        try:
            embeddings = await self.embedding_func.aembed_documents(texts)
        except NotImplementedError:
            embeddings = [await self.embedding_func.aembed_query(x) for x in texts]
        time_record["milvus_embedding_time"] = round(time.perf_counter() - embedding_start, 2)

        if not embeddings:
            insert_logger.info("No embeddings generated, skipping.")
            return []

        self._ensure_collection(vector_dim=len(embeddings[0]))

        row_ids = []
        rows = []
        for idx, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
            row_id = ids[idx] if ids else uuid.uuid4().hex
            row_ids.append(row_id)
            metadata = dict(metadata or {})

            rows.append(
                {
                    self._primary_field: row_id,
                    self._doc_id_field: str(metadata.get(self._doc_id_field, "")),
                    self._kb_id_field: str(metadata.get(self._kb_id_field, "")),
                    self._file_id_field: str(metadata.get(self._file_id_field, "")),
                    self._text_field: text,
                    self._metadata_field: json.dumps(metadata, ensure_ascii=False, default=str),
                    self._vector_field: self._normalize_vector(embedding),
                }
            )

        insert_start = time.perf_counter()
        client = self._get_client()
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            client.insert(
                collection_name=self.collection_name,
                data=rows[start:end],
                timeout=timeout if timeout is not None else self.timeout,
            )
            self.inserted_since_last_flush += end - start
        client.commit()

        time_record["milvus_insert_time"] = round(time.perf_counter() - insert_start, 2)
        return row_ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        return self._run_coro_sync(self.aadd_texts(texts=texts, metadatas=metadatas, **kwargs))

    async def aadd_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return await self.aadd_texts(texts=texts, metadatas=metadatas, **kwargs)

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        expr: Optional[str] = None,
        **kwargs: Any,
    ) -> List[tuple[Document, float]]:
        if not self._check_collection_exists():
            return []

        table = self._load_table()
        if table is None:
            return []

        query_embedding = await self.embedding_func.aembed_query(query)
        filters = self._build_filter(expr, table)

        output_fields = kwargs.get("output_fields") or [
            self._text_field,
            self._metadata_field,
            self._doc_id_field,
            self._kb_id_field,
            self._file_id_field,
        ]

        search_res = self._get_client().search(
            collection_name=self.collection_name,
            data=self._normalize_vector(query_embedding),
            filter=filters,
            limit=k,
            with_dist=True,
            output_fields=output_fields,
            search_params={"metric_type": self.metric_type},
            timeout=kwargs.get("timeout", self.timeout),
            anns_field=self._vector_field,
        )

        result: List[tuple[Document, float]] = []
        for row in search_res:
            doc = self._row_to_document(row)
            score = float(row.get("distance", 0.0))
            result.append((doc, score))
        return result

    def similarity_search_with_score(self, *args: Any, **kwargs: Any) -> List[tuple[Document, float]]:
        return self._run_coro_sync(self.asimilarity_search_with_score(*args, **kwargs))

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        result = self.similarity_search_with_score(query=query, k=k, **kwargs)
        return [doc for doc, _ in result]

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        expr: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        result = await self.asimilarity_search_with_score(query=query, k=k, expr=expr, **kwargs)
        return [doc for doc, _ in result]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        return self._run_coro_sync(
            self.amax_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                **kwargs,
            )
        )

    def get_pks(self, expr: str, timeout: int = 10) -> List[str]:
        if not self._check_collection_exists():
            return []

        table = self._load_table()
        if table is None:
            return []

        filters = self._build_filter(expr, table)
        rows = self._get_client().query(
            collection_name=self.collection_name,
            filter=filters,
            output_fields=[self._primary_field],
            timeout=timeout,
        )
        return [row[self._primary_field] for row in rows if row.get(self._primary_field) is not None]

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        expr = kwargs.get("expr", "")
        timeout = kwargs.get("timeout", 10)

        if not expr and ids:
            expr = f"{self._primary_field} in {ids}"

        if not expr or not str(expr).strip():
            return True

        if not self._check_collection_exists():
            return True

        table = self._load_table()
        if table is None:
            return True

        filters = self._build_filter(expr, table)
        if not filters:
            return True

        res = self._get_client().delete(
            collection_name=self.collection_name,
            filter=filters,
            timeout=timeout,
        )
        self._get_client().commit()
        delete_count = int(res.get("delete_count", 0)) if isinstance(res, dict) else 0
        return delete_count >= 0

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ):
        instance = cls(embedding_function=embedding, **kwargs)
        instance.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return instance


class SelfMilvus(SelfDMVectorStore):
    """Compatibility alias kept for old imports."""


class VectorStoreMilvusClient:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.host = DAMENG_HOST_LOCAL
        self.port = DAMENG_PORT_LOCAL
        self.local_vectorstore: SelfDMVectorStore = SelfDMVectorStore(
            embedding_function=YouDaoEmbeddings(),
            connection_args={
                "host": self.host,
                "port": self.port,
                "user": DAMENG_USER_LOCAL,
                "password": DAMENG_PASSWORD_LOCAL,
                "db_name": DAMENG_DATABASE_LOCAL,
            },
            collection_name=MILVUS_COLLECTION_NAME,
            auto_id=True,
            search_params={"metric_type": "COSINE"},
        )
        debug_logger.info(f"init vectorstore dm {self.host}:{self.port}, {MILVUS_COLLECTION_NAME}")

    def get_local_chunks(self, expr, timeout=10):
        future = self.executor.submit(partial(self.local_vectorstore.get_pks, expr=expr, timeout=timeout))
        return future.result()

    @get_time
    def delete_expr(self, expr):
        try:
            chunks = self.get_local_chunks(expr)
        except Exception as e:
            debug_logger.error(f"failed to query chunks before delete, expr: {expr}, error: {e}")
            return

        if len(chunks) == 0:
            debug_logger.info(f"expr: {expr} not found in local vectorstore")
            return
        try:
            ok = self.local_vectorstore.delete(expr=expr, timeout=10)
            res = {"delete_count": len(chunks), "ok": ok}
            debug_logger.info(f"local vectorstore delete expr: {expr} res: {res}")
        except Exception as e:
            debug_logger.error(f"local vectorstore delete expr: {expr} error: {e}")
