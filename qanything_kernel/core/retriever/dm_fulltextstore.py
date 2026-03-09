from typing import List, Optional

from langchain_core.documents import Document

from qanything_kernel.configs.model_config import DB_TYPE, DM_FULLTEXT_TOP_K
from qanything_kernel.connector.database.db_client import KnowledgeBaseManager
from qanything_kernel.utils.custom_log import debug_logger


class StoreDMFullTextClient:
    def __init__(self, dm_client: Optional[KnowledgeBaseManager] = None):
        if dm_client is None and str(DB_TYPE).lower() not in ("dm", "dameng"):
            raise RuntimeError("StoreDMFullTextClient only supports DB_TYPE=dm/dameng")
        self.dm_client = dm_client or KnowledgeBaseManager()
        self.id_key = "doc_id"
        debug_logger.info("Init StoreDMFullTextClient")

    async def aadd_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        if ids is not None and len(ids) != len(documents):
            raise ValueError("`ids` should have the same length as `documents`")
        records = []
        inserted_ids = []
        for idx, doc in enumerate(documents):
            metadata = doc.metadata or {}
            doc_id = ids[idx] if ids is not None else metadata.get(self.id_key)
            kb_id = metadata.get("kb_id")
            file_id = metadata.get("file_id")
            if not doc_id or not kb_id or not file_id:
                continue
            records.append(
                {
                    "doc_id": str(doc_id),
                    "kb_id": str(kb_id),
                    "file_id": str(file_id),
                    "content": doc.page_content or "",
                }
            )
            inserted_ids.append(str(doc_id))
        if records:
            self.dm_client.upsert_fulltext_documents(records)
        return inserted_ids

    async def asimilarity_search(self, query: str, k: int = DM_FULLTEXT_TOP_K, kb_ids: Optional[List[str]] = None) -> List[Document]:
        limit = k if isinstance(k, int) and k > 0 else DM_FULLTEXT_TOP_K
        rows = self.dm_client.query_fulltext_documents(query, kb_ids or [], limit=limit)
        docs = []
        for row in rows:
            docs.append(
                Document(
                    page_content=row.get("content", ""),
                    metadata={
                        self.id_key: row.get("doc_id"),
                        "kb_id": row.get("kb_id"),
                        "file_id": row.get("file_id"),
                        "score": row.get("score", 0.0),
                    },
                )
            )
        return docs

    def delete(self, doc_ids: List[str]) -> None:
        self.dm_client.delete_fulltext_documents_by_ids(doc_ids or [])

    def delete_files(self, file_ids: List[str], file_chunks: Optional[List[int]] = None) -> None:
        _ = file_chunks
        self.dm_client.delete_fulltext_documents_by_file_ids(file_ids or [])
