import asyncio
import json
import os
import time
import unittest
import uuid
from collections import OrderedDict

from pymilvus import connections, utility
from langchain_core.documents import Document

from qanything_kernel.configs import model_config
from qanything_kernel.core.retriever.vectorstore import SelfMilvus
from qanything_kernel.connector.database.milvus.milvus_cache import MilvusLRUCache
from qanything_kernel.connector.database.milvus.milvus_client import MilvusClient


class _DummyEmbeddings:
    """用于集成测试的简单向量化实现。"""

    def _vec(self, text: str, dim: int = 4):
        seed = sum(bytearray(text.encode("utf-8"))) % 10
        return [float(seed + i) for i in range(dim)]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)

    async def aembed_query(self, text):
        return self.embed_query(text)


class _EvictOnlyCache(MilvusLRUCache):
    """仅用于测试 evict/get/put，避免影响已有集合缓存。"""

    def update_cache(self):
        self.cache = OrderedDict()


def _make_vector(dim: int, seed: int = 1):
    return [float((seed + i) % 10) for i in range(dim)]


class TestMilvusInterfaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        host = os.getenv("QANYTHING_MILVUS_HOST", model_config.MILVUS_HOST_LOCAL)
        port = int(os.getenv("QANYTHING_MILVUS_PORT", model_config.MILVUS_PORT))
        cls.host = host
        cls.port = port
        # 出处: qanything_kernel/connector/database/milvus/milvus_client.py:125
        # 出处: qanything_kernel/connector/database/milvus/milvus_cache.py:18
        try:
            connections.connect(host=host, port=port)
            utility.list_collections()
        except Exception as exc:
            raise unittest.SkipTest(f"Milvus 不可用: {exc}")

        # 将运行时连接参数同步到模块变量（这些变量在类初始化时读取）
        import qanything_kernel.core.retriever.vectorstore as vectorstore_mod
        import qanything_kernel.connector.database.milvus.milvus_client as milvus_client_mod
        import qanything_kernel.connector.database.milvus.milvus_cache as milvus_cache_mod

        vectorstore_mod.MILVUS_HOST_LOCAL = host
        vectorstore_mod.MILVUS_PORT = port
        milvus_client_mod.MILVUS_HOST_ONLINE = host
        milvus_client_mod.MILVUS_PORT = port
        milvus_cache_mod.MILVUS_HOST_ONLINE = host
        milvus_cache_mod.MILVUS_PORT = port

    @classmethod
    def tearDownClass(cls):
        try:
            connections.disconnect("default")
        except Exception:
            pass

    def test_langchain_selfmilvus_interfaces(self):
        # ---------- LangChain Milvus / SelfMilvus ----------
        lc_collection = f"ut_langchain_{uuid.uuid4().hex}"
        store = SelfMilvus(
            embedding_function=_DummyEmbeddings(),
            connection_args={"host": self.host, "port": self.port},
            collection_name=lc_collection,
            partition_key_field="kb_id",
            auto_id=True,
            # 出处: qanything_kernel/core/retriever/vectorstore.py:122 (set_properties)
            collection_properties={"collection.ttl.seconds": 3600},
        )

        try:
            texts = ["hello", "world"]
            metadatas = [
                {"kb_id": "kb1", "num": 1, "tag": "a"},
                {"kb_id": "kb1", "num": 2, "tag": "b"},
            ]
            # 出处: qanything_kernel/core/retriever/vectorstore.py:59 (infer_dtype_bydata)
            # 出处: qanything_kernel/core/retriever/vectorstore.py:73/76/80/85/91/101 (FieldSchema/DataType)
            # 出处: qanything_kernel/core/retriever/vectorstore.py:105 (CollectionSchema)
            # 出处: qanything_kernel/core/retriever/vectorstore.py:113 (Collection)
            # 出处: qanything_kernel/core/retriever/vectorstore.py:243 (col.insert)
            pks = asyncio.run(store.aadd_texts(texts, metadatas=metadatas, batch_size=2))
            self.assertTrue(pks)

            # 出处: qanything_kernel/core/retriever/vectorstore.py:147 (col.query)
            expr = 'kb_id == "kb1"'
            res = store.get_expr_result(expr=expr, output_fields=[store._primary_field])
            self.assertTrue(res)

            docs = [Document(page_content="doc one", metadata={"kb_id": "kb1", "num": 3, "tag": "c"})]
            # 出处: qanything_kernel/core/retriever/parent_retriever.py:144 (aadd_documents)
            asyncio.run(store.aadd_documents(docs))

            # 出处: qanything_kernel/core/retriever/parent_retriever.py:49 (asimilarity_search_with_score)
            sim_res = asyncio.run(store.asimilarity_search_with_score("hello", k=1, expr=expr))
            self.assertTrue(sim_res)

            # 出处: qanything_kernel/core/retriever/parent_retriever.py:45 (amax_marginal_relevance_search)
            mmr_res = asyncio.run(store.amax_marginal_relevance_search("hello", k=1, expr=expr))
            self.assertTrue(mmr_res)

            # 出处: qanything_kernel/core/retriever/vectorstore.py:284 (get_pks)
            pks2 = store.get_pks(expr=expr, timeout=10)
            self.assertTrue(pks2)

            # 出处: qanything_kernel/core/retriever/vectorstore.py:298 (local_vectorstore.delete)
            store.delete(expr=expr, timeout=10)

            # 出处: qanything_kernel/core/retriever/vectorstore.py:30/257 (col.flush)
            async def _flush_once():
                store._milvus_flush()
                await asyncio.sleep(0.1)

            asyncio.run(_flush_once())

            # ---------- MilvusLRUCache 操作 ----------
            safe_cache = _EvictOnlyCache(capacity=1)
            # 出处: qanything_kernel/connector/database/milvus/milvus_cache.py:65 (collection.load)
            safe_cache.put(lc_collection, store.col, _async=False)
            # 出处: qanything_kernel/connector/database/milvus/milvus_cache.py:46-47 (utility.load_state/LoadState)
            # 出处: qanything_kernel/connector/database/milvus/milvus_cache.py:54 (collection.load)
            safe_cache.get(lc_collection)
            # 出处: qanything_kernel/connector/database/milvus/milvus_cache.py:79 (collection.release)
            safe_cache.evict()
        finally:
            try:
                utility.drop_collection(lc_collection)
            except Exception:
                pass

    def test_milvusclient_interfaces(self):
        cache = MilvusLRUCache(capacity=2)
        # 出处: qanything_kernel/connector/database/milvus/milvus_cache.py:26 (utility.list_collections)
        # 出处: qanything_kernel/connector/database/milvus/milvus_cache.py:29 (utility.load_state)
        # 出处: qanything_kernel/connector/database/milvus/milvus_cache.py:31 (Collection(name=...))

        # ---------- MilvusClient (pymilvus) ----------
        user_id = f"ut_user_{uuid.uuid4().hex}"
        kb_id = "kb_ut_1"
        # 出处: qanything_kernel/connector/database/milvus/milvus_client.py:50 (utility.has_collection)
        # 出处: qanything_kernel/connector/database/milvus/milvus_client.py:51 (CollectionSchema)
        # 出处: qanything_kernel/connector/database/milvus/milvus_client.py:53/57 (Collection)
        # 出处: qanything_kernel/connector/database/milvus/milvus_client.py:54 (create_index)
        # 出处: qanything_kernel/connector/database/milvus/milvus_client.py:65-72 (FieldSchema/DataType)
        # 出处: qanything_kernel/connector/database/milvus/milvus_client.py:128-130 (has_partition/create_partition/Partition)
        client = MilvusClient(user_id=user_id, kb_ids=[kb_id], milvus_cache=cache)

        file_id = "file_ut_1"
        chunk_id = f"{file_id}_0"
        vector = _make_vector(768, seed=3)
        rows = [
            [chunk_id],
            [file_id],
            ["a.pdf"],
            ["/tmp/a.pdf"],
            [time.strftime("%Y%m%d%H%M%S")],
            ["content"],
            [vector],
            [json.dumps({"doc_id": "doc_1"})],
        ]
        client.sess.insert(rows, partition_name=kb_id)
        client.sess.flush()
        client.sess.load()

        # 出处: qanything_kernel/connector/database/milvus/milvus_client.py:118 (schema.fields)
        _ = client.output_fields

        # 出处: qanything_kernel/connector/database/milvus/milvus_client.py:143 (sess.search)
        search_res = client.search_emb_async([vector], expr="", top_k=1, client_timeout=5)
        self.assertTrue(search_res)

        # 出处: qanything_kernel/connector/database/milvus/milvus_client.py:176 (sess.query)
        query_res = client.query_expr_async(expr=f'file_id in ["{file_id}"]', output_fields=["file_id"])
        self.assertTrue(query_res)

        # 出处: qanything_kernel/connector/database/milvus/milvus_client.py:200 (sess.delete)
        client.delete_files_batch([file_id])

        # 出处: qanything_kernel/connector/database/milvus/milvus_client.py:186-187 (drop_partition)
        client.delete_partition(kb_id)

        # 出处: qanything_kernel/connector/database/milvus/milvus_client.py:182 (drop_collection)
        client.delete_collection()

    def test_milvus_cache_disconnect(self):
        cache = MilvusLRUCache(capacity=1)
        # 出处: qanything_kernel/connector/database/milvus/milvus_cache.py:85 (connections.disconnect)
        cache.clear()
        # 断开后重新连接，避免影响其它用例
        connections.connect(host=self.host, port=self.port)


if __name__ == "__main__":
    unittest.main()
