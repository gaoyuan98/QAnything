import asyncio
import os
import unittest
import uuid

from langchain_core.documents import Document

from qanything_kernel.configs import model_config
from qanything_kernel.core.retriever.elasticsearchstore import StoreElasticSearchClient
#
# ES 接口使用情况（来源）
# -ES 连接与索引配置来源于 qanything_kernel/configs/model_config.py:173、qanything_kernel/configs/model_config.py:174、qanything_kernel/configs/model_config.py:175、qanything_kernel/configs/model_config.py:176、qanything_kernel/configs/model_config.py:178。
# -ES 客户端封装与 BM25 策略初始化在 qanything_kernel/core/retriever/elasticsearchstore.py:6、qanything_kernel/core/retriever/elasticsearchstore.py:8、qanything_kernel/core/retriever/elasticsearchstore.py:13。
# -ES 写入与检索的业务使用在 qanything_kernel/core/retriever/parent_retriever.py:143、qanything_kernel/core/retriever/parent_retriever.py:148、qanything_kernel/core/retriever/parent_retriever.py:231、qanything_kernel/core/retriever/parent_retriever.py:232。
# -ES 删除与 delete_files 的封装在 qanything_kernel/core/retriever/elasticsearchstore.py:17、qanything_kernel/core/retriever/elasticsearchstore.py:24。
# -ES delete_files 的调用点在 qanything_kernel/qanything_server/handler.py:483、qanything_kernel/qanything_server/handler.py:549。
# -业务初始化链路里创建 ES 客户端与检索器在 qanything_kernel/core/local_doc_qa.py:74、qanything_kernel/core/local_doc_qa.py:75。

class TestElasticSearchInterfaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        es_url = os.getenv("QANYTHING_ES_URL", model_config.ES_URL)
        es_user = os.getenv("QANYTHING_ES_USER", model_config.ES_USER)
        es_password = os.getenv("QANYTHING_ES_PASSWORD", model_config.ES_PASSWORD)
        index_name = f"ut_es_{uuid.uuid4().hex}"

        import qanything_kernel.core.retriever.elasticsearchstore as es_store_mod

        es_store_mod.ES_URL = es_url
        es_store_mod.ES_USER = es_user if es_user else None
        es_store_mod.ES_PASSWORD = es_password if es_password else None
        es_store_mod.ES_INDEX_NAME = index_name

        cls.index_name = index_name
        cls.es_client = None

        try:
            cls.es_client = StoreElasticSearchClient()
            if not cls.es_client.es_store.client.ping():
                raise RuntimeError("Elasticsearch ping failed")
        except Exception as exc:
            raise unittest.SkipTest(f"Elasticsearch 不可用: {exc}")

    @classmethod
    def tearDownClass(cls):
        if not cls.es_client:
            return
        try:
            cls.es_client.es_store.client.indices.delete(
                index=cls.index_name,
                ignore_unavailable=True,
            )
        except Exception:
            pass

    def _refresh_index(self):
        self.es_client.es_store.client.indices.refresh(index=self.index_name)

    def test_elasticsearchstore_interfaces(self):
        kb_id = "kb_ut"
        file_id = f"file_ut_{uuid.uuid4().hex[:8]}"
        docs = [
            Document(page_content="hello elasticsearch", metadata={"kb_id": kb_id, "file_id": file_id}),
            Document(page_content="hello bm25", metadata={"kb_id": kb_id, "file_id": file_id}),
        ]
        doc_ids = [f"{file_id}_{i}" for i in range(len(docs))]

        # 出处: qanything_kernel/core/retriever/elasticsearchstore.py:8-14 (ElasticsearchStore 初始化)
        # 出处: qanything_kernel/core/retriever/parent_retriever.py:147-149 (es_store.aadd_documents)
        inserted = asyncio.run(self.es_client.es_store.aadd_documents(docs, ids=doc_ids))
        self.assertTrue(inserted)
        self._refresh_index()

        # 出处: qanything_kernel/core/retriever/parent_retriever.py:231-233 (es_store.asimilarity_search + filter)
        filter_clause = [{"terms": {"metadata.kb_id.keyword": [kb_id]}}]
        found = asyncio.run(
            self.es_client.es_store.asimilarity_search("hello", k=2, filter=filter_clause)
        )
        self.assertTrue(found)

        # 出处: qanything_kernel/core/retriever/elasticsearchstore.py:17-20 (es_store.delete)
        self.es_client.delete(doc_ids)
        self._refresh_index()
        found_after_delete = asyncio.run(
            self.es_client.es_store.asimilarity_search("hello", k=2, filter=filter_clause)
        )
        self.assertFalse(found_after_delete)

        # 出处: qanything_kernel/core/retriever/elasticsearchstore.py:24-29 (delete_files 构造 doc_id)
        # 出处: qanything_kernel/qanything_server/handler.py:483 (es_client.delete_files)
        asyncio.run(self.es_client.es_store.aadd_documents(docs, ids=doc_ids))
        self._refresh_index()
        self.es_client.delete_files([file_id], [len(docs)])
        self._refresh_index()
        found_after_delete_files = asyncio.run(
            self.es_client.es_store.asimilarity_search("hello", k=2, filter=filter_clause)
        )
        self.assertFalse(found_after_delete_files)


if __name__ == "__main__":
    unittest.main()
