import argparse

from qanything_kernel.configs.model_config import DB_TYPE
from qanything_kernel.connector.database.db_client import KnowledgeBaseManager
from qanything_kernel.utils.custom_log import debug_logger


def run_backfill(batch_size: int) -> None:
    if str(DB_TYPE).lower() not in ("dm", "dameng"):
        raise RuntimeError("backfill_dm_fulltext_index.py only supports DB_TYPE=dm/dameng")
    manager = KnowledgeBaseManager()
    source_docs = manager.get_fulltext_backfill_source_documents()
    total = len(source_docs)
    debug_logger.info(f"start dm fulltext backfill, total documents: {total}, batch_size: {batch_size}")
    upserted = 0
    for i in range(0, total, batch_size):
        batch = source_docs[i:i + batch_size]
        upserted += manager.upsert_fulltext_documents(batch)
        debug_logger.info(f"backfill progress: {min(i + batch_size, total)}/{total}")
    debug_logger.info(f"dm fulltext backfill done, upserted rows: {upserted}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill Documents -> DocumentFulltextIndex for Dameng")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for upsert")
    args = parser.parse_args()
    run_backfill(batch_size=args.batch_size)
