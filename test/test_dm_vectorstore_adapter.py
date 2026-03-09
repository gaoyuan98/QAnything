import unittest

from sqlalchemy import Column, MetaData, String, Table, Text

from qanything_kernel.core.retriever.vectorstore import SelfDMVectorStore


class _DummyEmbeddings:
    async def aembed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]

    async def aembed_query(self, text):
        return [0.1, 0.2]


class TestDMVectorStoreAdapter(unittest.TestCase):
    def setUp(self):
        self.store = SelfDMVectorStore(
            embedding_function=_DummyEmbeddings(),
            connection_args={"uri": "127.0.0.1:5236", "user": "u", "password": "p", "db_name": "s"},
            collection_name="ut_collection",
        )
        metadata = MetaData()
        self.table = Table(
            "ut_collection",
            metadata,
            Column("kb_id", String),
            Column("file_id", String),
            Column("doc_id", String),
            Column("metadata_json", Text),
            Column("text", Text),
        )

    def test_build_filter_eq_and_in(self):
        expr = 'kb_id == "KB1" and file_id in ["f1", "f2"]'
        clauses = self.store._build_filter(expr, self.table)
        self.assertEqual(len(clauses), 2)

    def test_build_filter_invalid_field(self):
        with self.assertRaises(ValueError):
            self.store._build_filter('missing == "x"', self.table)

    def test_row_to_document(self):
        row = {
            "text": "hello",
            "metadata_json": '{"kb_id": "KB1", "file_id": "f1"}',
            "doc_id": "d1",
        }
        doc = self.store._row_to_document(row)
        self.assertEqual(doc.page_content, "hello")
        self.assertEqual(doc.metadata["kb_id"], "KB1")
        self.assertEqual(doc.metadata["file_id"], "f1")
        self.assertEqual(doc.metadata["doc_id"], "d1")


if __name__ == "__main__":
    unittest.main()
