from qanything_kernel.configs.model_config import (
    DAMENG_HOST_LOCAL,
    DAMENG_PORT_LOCAL,
    DAMENG_USER_LOCAL,
    DAMENG_PASSWORD_LOCAL,
    DAMENG_DATABASE_LOCAL,
)
from qanything_kernel.utils.custom_log import debug_logger, insert_logger
from qanything_kernel.connector.database.mysql.mysql_client import KnowledgeBaseManager as MysqlKnowledgeBaseManager
from dbutils.pooled_db import PooledDB
import json
import uuid
from datetime import datetime, timedelta
import re
from typing import Dict, List


class KnowledgeBaseManager(MysqlKnowledgeBaseManager):

    def __init__(self, pool_size=12):
        self.pool_size = pool_size
        self.dm_config = {
            "host": DAMENG_HOST_LOCAL,
            "port": DAMENG_PORT_LOCAL,
            "user": DAMENG_USER_LOCAL,
            "password": DAMENG_PASSWORD_LOCAL,
            "database": DAMENG_DATABASE_LOCAL,
        }
        self.check_database_(
            DAMENG_HOST_LOCAL,
            DAMENG_PORT_LOCAL,
            DAMENG_USER_LOCAL,
            DAMENG_PASSWORD_LOCAL,
            DAMENG_DATABASE_LOCAL,
        )
        self._init_dm_pool()
        self.create_tables_()
        debug_logger.info("[SUCCESS] Dameng database {} connected".format(DAMENG_DATABASE_LOCAL))

    def _user_table(self):
        return "\"USER\""

    def _quote_qalog_columns(self, columns):
        quoted = []
        for column in columns:
            if column.lower() == "model":
                quoted.append("\"MODEL\"")
            else:
                quoted.append(column)
        return ", ".join(quoted)

    def _dm_prepare_query(self, query):
        return query.replace("%s", "?")


    def _init_dm_pool(self):
        try:
            import dmPython
        except ImportError as exc:
            raise RuntimeError("dmPython is required when DB_TYPE is dameng") from exc
        self.cnxpool = PooledDB(
            creator=dmPython,
            maxconnections=self.pool_size,
            mincached=self.pool_size,
            maxcached=self.pool_size,
            maxshared=0,
            blocking=True,
            ping=1,
            user=self.dm_config["user"],
            password=self.dm_config["password"],
            server=self.dm_config["host"],
            port=self.dm_config["port"],
            schema=self.dm_config.get("database"),
        )
        self.free_cnx = self.pool_size
        self.used_cnx = 0

    def _dm_connect(self):
        try:
            import dmPython
        except ImportError as exc:
            raise RuntimeError("dmPython is required when DB_TYPE is dameng") from exc
        host = self.dm_config["host"]
        port = self.dm_config["port"]
        user = self.dm_config["user"]
        password = self.dm_config["password"]
        database = self.dm_config.get("database")
        return dmPython.connect(user=user, password=password, server=host, port=port, schema=database)

    def _dm_get_connection(self):
        conn = self.cnxpool.connection()
        self.used_cnx += 1
        self.free_cnx -= 1
        if self.free_cnx < 4:
            debug_logger.info(
                "Get connection success. Pool status: free {} used {}".format(self.free_cnx, self.used_cnx)
            )
        return conn

    def _dm_release_connection(self, conn):
        self.used_cnx -= 1
        self.free_cnx += 1
        if self.free_cnx <= 4:
            debug_logger.info(
                "Release connection. Pool status: free {} used {}".format(self.free_cnx, self.used_cnx)
            )
        conn.close()

    def _is_ddl_query(self, query):
        return bool(re.match(r"\s*(CREATE|ALTER|DROP)\b", query, flags=re.IGNORECASE))

    def _is_ignorable_dm_error(self, err):
        msg = str(err).lower()
        return any(token in msg for token in ("already exists", "duplicate", "exists", "not exists", "does not exist"))

    @staticmethod
    def _normalize_dm_text(value):
        if value is None:
            return ""
        if hasattr(value, "read"):
            return value.read()
        return str(value)

    @staticmethod
    def _split_fulltext_terms(query_text: str) -> List[str]:
        terms = [term.strip() for term in re.split(r"[\s,;，。！？!?.]+", query_text or "") if term.strip()]
        if not terms and query_text:
            terms = [query_text.strip()]
        return terms[:8]

    def check_database_(self, host, port, user, password, database_name):
        conn = self._dm_connect()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            debug_logger.info("[SUCCESS] Dameng database {} check passed".format(database_name))
        finally:
            if cursor is not None:
                cursor.close()
            conn.close()

    def execute_query_(self, query, params, commit=False, fetch=False, check=False, user_dict=False):
        conn = None
        cursor = None
        result = None
        try:
            conn = self._dm_get_connection()
            cursor = conn.cursor()
            query = self._dm_prepare_query(query)
            cursor.execute(query, params or ())

            if commit:
                conn.commit()

            if fetch:
                rows = cursor.fetchall()
                if user_dict and cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    result = [dict(zip(columns, row)) for row in rows]
                else:
                    result = rows
            elif check:
                result = cursor.rowcount
        except Exception as err:
            if self._is_ddl_query(query) and self._is_ignorable_dm_error(err):
                debug_logger.info("DDL already applied (this is okay): {}".format(query))
            else:
                debug_logger.error("Execute query failed: {} SQL: {}".format(err, query))
            if commit and conn is not None:
                try:
                    conn.rollback()
                except Exception:
                    pass
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                if not commit:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                self._dm_release_connection(conn)
        return result

    def delete_documents(self, file_ids):
        total_deleted = 0
        for file_id in file_ids:
            query = "SELECT doc_id FROM Documents WHERE doc_id LIKE %s"
            doc_ids = self.execute_query_(query, (f"{file_id}_%",), fetch=True)
            debug_logger.info(f"Found documents to delete: {doc_ids}, {file_id}")

            if doc_ids:
                doc_ids = [doc_id[0] for doc_id in doc_ids]
                batch_size = 100
                for i in range(0, len(doc_ids), batch_size):
                    batch_doc_ids = doc_ids[i:i + batch_size]
                    placeholders = ','.join(['%s'] * len(batch_doc_ids))
                    delete_query = "DELETE FROM Documents WHERE doc_id IN ({})".format(placeholders)
                    res = self.execute_query_(delete_query, batch_doc_ids, commit=True, check=True)
                    total_deleted += res
        debug_logger.info(f"Deleted documents count: {total_deleted}")


    def create_tables_(self):
        user_table = self._user_table()
        query = f"""
            CREATE TABLE {user_table} (
                id INTEGER IDENTITY(1,1) PRIMARY KEY,
                user_id VARCHAR(255) UNIQUE,
                user_name VARCHAR(255),
                creation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        self.execute_query_(query, (), commit=True)
        query = """
            CREATE TABLE KnowledgeBase (
                id INTEGER IDENTITY(1,1) PRIMARY KEY,
                kb_id VARCHAR(255) UNIQUE,
                user_id VARCHAR(255),
                kb_name VARCHAR(255),
                deleted INTEGER DEFAULT 0,
                latest_qa_time TIMESTAMP,
                latest_insert_time TIMESTAMP
            )
        """
        self.execute_query_(query, (), commit=True)
        query = """
            CREATE TABLE File (
                id INTEGER IDENTITY(1,1) PRIMARY KEY,
                file_id VARCHAR(255) UNIQUE,
                user_id VARCHAR(255) DEFAULT 'unknown',
                kb_id VARCHAR(255),
                file_name VARCHAR(255),
                status VARCHAR(255),
                msg VARCHAR(255) DEFAULT 'success',
                transfer_status VARCHAR(255),
                deleted INTEGER DEFAULT 0,
                file_size INTEGER DEFAULT -1,
                content_length INTEGER DEFAULT -1,
                chunks_number INTEGER DEFAULT -1,
                file_location VARCHAR(255) DEFAULT 'unknown',
                file_url VARCHAR(2048) DEFAULT '',
                upload_infos CLOB,
                chunk_size INTEGER DEFAULT -1,
                timestamp VARCHAR(255) DEFAULT '197001010000'
            )
        """
        self.execute_query_(query, (), commit=True)
        query = """
            CREATE TABLE Faqs (
                id INTEGER IDENTITY(1,1) PRIMARY KEY,
                faq_id  VARCHAR(255) UNIQUE,
                user_id VARCHAR(255) NOT NULL,
                kb_id VARCHAR(255) NOT NULL,
                question VARCHAR(512) NOT NULL,
                answer VARCHAR(2048) NOT NULL,
                nos_keys VARCHAR(768)
            )
        """
        self.execute_query_(query, (), commit=True)
        query = """
            CREATE TABLE Documents (
                id INTEGER IDENTITY(1,1) PRIMARY KEY,
                doc_id VARCHAR(255) UNIQUE,
                json_data CLOB
            )
        """
        self.execute_query_(query, (), commit=True)
        query = """
            CREATE TABLE DocumentFulltextIndex (
                id INTEGER IDENTITY(1,1) PRIMARY KEY,
                doc_id VARCHAR(255) UNIQUE,
                kb_id VARCHAR(255) NOT NULL,
                file_id VARCHAR(255) NOT NULL,
                content CLOB,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        self.execute_query_(query, (), commit=True)
        query = """
            CREATE TABLE QaLogs (
                id INTEGER IDENTITY(1,1) PRIMARY KEY,
                qa_id VARCHAR(255) UNIQUE,
                user_id VARCHAR(255) NOT NULL,
                bot_id VARCHAR(255),
                kb_ids VARCHAR(2048) NOT NULL,
                query VARCHAR(512) NOT NULL,
                "MODEL" VARCHAR(64) NOT NULL,
                product_source VARCHAR(64) NOT NULL,
                time_record VARCHAR(512) NOT NULL,
                history CLOB NOT NULL,
                condense_question VARCHAR(1024) NOT NULL,
                prompt CLOB NOT NULL,
                result CLOB NOT NULL,
                retrieval_documents CLOB NOT NULL,
                source_documents CLOB NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        self.execute_query_(query, (), commit=True)
        query = """
            CREATE TABLE FileImages (
                id INTEGER IDENTITY(1,1) PRIMARY KEY,
                image_id VARCHAR(255) UNIQUE,
                file_id VARCHAR(255) NOT NULL,
                user_id VARCHAR(255) NOT NULL,
                kb_id VARCHAR(255) NOT NULL,
                nos_key VARCHAR(255) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        self.execute_query_(query, (), commit=True)
        query = """
            CREATE TABLE QanythingBot (
                id INTEGER IDENTITY(1,1) PRIMARY KEY,
                bot_id          VARCHAR(64) UNIQUE,
                user_id         VARCHAR(255),
                bot_name        VARCHAR(512),
                description     VARCHAR(512),
                head_image      VARCHAR(512),
                prompt_setting  CLOB,
                welcome_message CLOB,
                kb_ids_str      VARCHAR(1024),
                deleted         INTEGER DEFAULT 0,
                create_time     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                update_time     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                llm_setting     VARCHAR(512) DEFAULT '{}'
            )
        """
        self.execute_query_(query, (), commit=True)

        index_queries = [
            "CREATE INDEX index_kb_id_deleted ON File (kb_id, deleted)",
            "CREATE INDEX idx_user_id_status ON File (user_id, status)",
            "CREATE INDEX index_bot_id ON QaLogs (bot_id)",
            "CREATE INDEX index_query ON QaLogs (query)",
            "CREATE INDEX index_timestamp ON QaLogs (timestamp)",
            "CREATE INDEX idx_fulltext_kb_file ON DocumentFulltextIndex (kb_id, file_id)",
            "CREATE CONTEXT INDEX idx_fulltext_content ON DocumentFulltextIndex(content)",
            "ALTER TABLE QanythingBot ADD COLUMN llm_setting VARCHAR(512) DEFAULT '{}'",
            "ALTER TABLE QanythingBot DROP COLUMN \"MODEL\"",
        ]

        for ddl in index_queries:
            self.execute_query_(ddl, (), commit=True)

        debug_logger.info("All tables and indexes checked/created successfully.")

    def add_user_(self, user_id, user_name):
        query = f"INSERT INTO {self._user_table()} (user_id, user_name) VALUES (%s, %s)"
        self.execute_query_(query, (user_id, user_name), commit=True)
        debug_logger.info("Add user: {} {}".format(user_id, user_name))

    def check_user_exist_(self, user_id):
        query = f"SELECT user_id FROM {self._user_table()} WHERE user_id = %s"
        result = self.execute_query_(query, (user_id,), fetch=True)
        debug_logger.info("check_user_exist {}".format(result))
        return result is not None and len(result) > 0

    def get_users(self):
        query = f"SELECT user_id FROM {self._user_table()}"
        return self.execute_query_(query, (), fetch=True)

    def add_qalog(self, user_id, bot_id, kb_ids, query, model, product_source, time_record, history,
                  condense_question, prompt, result, retrieval_documents, source_documents):
        debug_logger.info("add_qalog: {}".format(query))
        qa_id = uuid.uuid4().hex
        kb_ids = json.dumps(kb_ids, ensure_ascii=False)
        retrieval_documents = json.dumps(retrieval_documents, ensure_ascii=False)
        source_documents = json.dumps(source_documents, ensure_ascii=False)
        history = json.dumps(history, ensure_ascii=False)
        time_record = json.dumps(time_record, ensure_ascii=False)
        insert_query = (
            "INSERT INTO QaLogs (qa_id, user_id, bot_id, kb_ids, query, \"MODEL\", product_source, time_record, "
            "history, condense_question, prompt, result, retrieval_documents, source_documents) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")
        self.execute_query_(insert_query, (qa_id, user_id, bot_id, kb_ids, query, model, product_source, time_record,
                                           history, condense_question, prompt, result, retrieval_documents,
                                           source_documents), commit=True)

    def get_qalog_by_filter(self, need_info, user_id=None, query=None, bot_id=None, time_range=None, any_kb_id=None,
                            qa_ids=None):
        need_info_columns = self._quote_qalog_columns(need_info)
        if qa_ids is not None:
            dm_query = f"SELECT {need_info_columns} FROM QaLogs WHERE qa_id IN ({','.join(['%s'] * len(qa_ids))})"
            qa_infos = self.execute_query_(dm_query, qa_ids, fetch=True)
        else:
            dm_query = f"SELECT {need_info_columns} FROM QaLogs WHERE timestamp BETWEEN %s AND %s"
            params = list(time_range)
            if user_id:
                dm_query += " AND user_id = %s"
                params.append(user_id)
            if any_kb_id:
                dm_query += " AND kb_ids LIKE %s"
                params.append(f"%{any_kb_id}%")
            if bot_id:
                dm_query += " AND bot_id = %s"
                params.append(bot_id)
            if query:
                dm_query += " AND query = %s"
                params.append(query)
            debug_logger.info("get_qalog_by_filter: {}".format(params))
            qa_infos = self.execute_query_(dm_query, params, fetch=True)
        qa_infos = [dict(zip(need_info, qa_info)) for qa_info in qa_infos]
        for qa_info in qa_infos:
            if "timestamp" in qa_info:
                qa_info["timestamp"] = qa_info["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            if "kb_ids" in qa_info:
                qa_info["kb_ids"] = json.loads(qa_info["kb_ids"])
            if "time_record" in qa_info:
                qa_info["time_record"] = json.loads(qa_info["time_record"])
            if "retrieval_documents" in qa_info:
                qa_info["retrieval_documents"] = json.loads(qa_info["retrieval_documents"])
            if "source_documents" in qa_info:
                qa_info["source_documents"] = json.loads(qa_info["source_documents"])
            if "history" in qa_info:
                qa_info["history"] = json.loads(qa_info["history"])
        if "timestamp" in need_info:
            qa_infos = sorted(qa_infos, key=lambda x: x["timestamp"], reverse=True)
        return qa_infos

    def get_qalog_by_ids(self, ids, need_info):
        placeholders = ",".join(["%s"] * len(ids))
        need_info_columns = self._quote_qalog_columns(need_info)
        query = "SELECT {} FROM QaLogs WHERE qa_id IN ({})".format(need_info_columns, placeholders)
        return self.execute_query_(query, ids, fetch=True)

    def get_random_qa_infos(self, limit=10, time_range=None, need_info=None):
        if need_info is None:
            need_info = ["qa_id", "user_id", "kb_ids", "query", "result", "timestamp"]
        if "qa_id" not in need_info:
            need_info.append("qa_id")
        if "user_id" not in need_info:
            need_info.append("user_id")
        if "timestamp" not in need_info:
            need_info.append("timestamp")
        need_info_columns = self._quote_qalog_columns(need_info)
        query = f"SELECT {need_info_columns} FROM QaLogs WHERE timestamp BETWEEN %s AND %s ORDER BY RAND() LIMIT %s"
        qa_infos = self.execute_query_(query, (time_range[0], time_range[1], limit), fetch=True)
        qa_infos = [dict(zip(need_info, qa_info)) for qa_info in qa_infos]
        for qa_info in qa_infos:
            qa_info["timestamp"] = qa_info["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        return qa_infos

    def get_related_qa_infos(self, qa_id, need_info=None, need_more=False):
        if need_info is None:
            need_info = ["user_id", "kb_ids", "query", "condense_question", "result", "timestamp", "product_source"]
        if "user_id" not in need_info:
            need_info.append("user_id")
        if "kb_ids" not in need_info:
            need_info.append("kb_ids")
        need_info_columns = self._quote_qalog_columns(need_info)
        query = f"SELECT {need_info_columns} FROM QaLogs WHERE qa_id = %s"
        qa_log = self.execute_query_(query, (qa_id,), fetch=True)
        qa_log = dict(zip(need_info, qa_log[0]))
        qa_log["timestamp"] = qa_log["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        user_id = qa_log["user_id"]
        current_time = datetime.utcnow()
        seven_days_ago = current_time - timedelta(days=7)
        if not need_more:
            return qa_log, [], []

        recent_logs = []
        offset = 0
        limit = 50
        while True:
            query_recent_logs = f"""
                SELECT {need_info_columns}
                FROM QaLogs
                WHERE user_id = %s AND timestamp >= %s
                ORDER BY timestamp
                LIMIT %s OFFSET %s
            """
            logs = self.execute_query_(query_recent_logs, (user_id, seven_days_ago, limit, offset), fetch=True)
            if not logs:
                break
            logs = [dict(zip(need_info, log)) for log in logs]
            for log in logs:
                log["timestamp"] = log["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            recent_logs.extend(logs)
            offset += limit
            break

        older_logs = []
        offset = 0
        while True:
            query_older_logs = f"""
                SELECT {need_info_columns}
                FROM QaLogs
                WHERE user_id = %s AND timestamp < %s
                ORDER BY timestamp
                LIMIT %s OFFSET %s
            """
            logs = self.execute_query_(query_older_logs, (user_id, seven_days_ago, limit, offset), fetch=True)
            if not logs:
                break
            logs = [dict(zip(need_info, log)) for log in logs]
            for log in logs:
                log["timestamp"] = log["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            older_logs.extend(logs)
            offset += limit
            break
        return qa_log, recent_logs, older_logs

    def add_document(self, doc_id, json_data):
        json_data = json.dumps(json_data, ensure_ascii=False)
        query = "INSERT INTO Documents (doc_id, json_data) VALUES (%s, %s)"
        self.execute_query_(query, (doc_id, json_data), commit=True, check=True)

    def upsert_fulltext_documents(self, records: List[Dict]) -> int:
        upserted = 0
        for item in records:
            doc_id = item.get("doc_id")
            kb_id = item.get("kb_id")
            file_id = item.get("file_id")
            content = item.get("content", "")
            if not doc_id or not kb_id or not file_id:
                continue
            update_query = """
                UPDATE DocumentFulltextIndex
                SET kb_id = %s, file_id = %s, content = %s, updated_at = CURRENT_TIMESTAMP
                WHERE doc_id = %s
            """
            updated = self.execute_query_(update_query, (kb_id, file_id, content, doc_id), commit=True, check=True)
            if updated is None:
                continue
            if updated == 0:
                insert_query = """
                    INSERT INTO DocumentFulltextIndex (doc_id, kb_id, file_id, content)
                    VALUES (%s, %s, %s, %s)
                """
                inserted = self.execute_query_(insert_query, (doc_id, kb_id, file_id, content), commit=True, check=True)
                if inserted:
                    upserted += 1
            else:
                upserted += int(updated)
        return upserted

    def query_fulltext_documents(self, query_text: str, kb_ids: List[str], limit: int = 30) -> List[Dict]:
        if not query_text or not kb_ids:
            return []
        placeholders = ",".join(["%s"] * len(kb_ids))

        contains_sqls = [
            (
                f"""
                SELECT doc_id, kb_id, file_id, content, CONTAINS(content, %s) AS score
                FROM DocumentFulltextIndex
                WHERE kb_id IN ({placeholders}) AND CONTAINS(content, %s) > 0
                ORDER BY score DESC
                FETCH FIRST %s ROWS ONLY
                """,
                [query_text, *kb_ids, query_text, limit],
            ),
            (
                f"""
                SELECT doc_id, kb_id, file_id, content, CONTAINS(content, %s) AS score
                FROM DocumentFulltextIndex
                WHERE kb_id IN ({placeholders}) AND CONTAINS(content, %s) > 0
                ORDER BY score DESC
                LIMIT %s
                """,
                [query_text, *kb_ids, query_text, limit],
            ),
        ]
        for sql, params in contains_sqls:
            rows = self.execute_query_(sql, tuple(params), fetch=True)
            if rows:
                return [
                    {
                        "doc_id": row[0],
                        "kb_id": row[1],
                        "file_id": row[2],
                        "content": self._normalize_dm_text(row[3]),
                        "score": float(row[4]) if row[4] is not None else 0.0,
                    }
                    for row in rows
                ]
            if rows is None:
                continue

        terms = self._split_fulltext_terms(query_text)
        if not terms:
            return []
        score_expr = " + ".join(["CASE WHEN content LIKE %s THEN 1 ELSE 0 END" for _ in terms])
        like_filter = " OR ".join(["content LIKE %s" for _ in terms])
        like_params = [f"%{term}%" for term in terms]
        like_sqls = [
            (
                f"""
                SELECT doc_id, kb_id, file_id, content, ({score_expr}) AS score
                FROM DocumentFulltextIndex
                WHERE kb_id IN ({placeholders}) AND ({like_filter})
                ORDER BY score DESC
                FETCH FIRST %s ROWS ONLY
                """,
                [*like_params, *kb_ids, *like_params, limit],
            ),
            (
                f"""
                SELECT doc_id, kb_id, file_id, content, ({score_expr}) AS score
                FROM DocumentFulltextIndex
                WHERE kb_id IN ({placeholders}) AND ({like_filter})
                ORDER BY score DESC
                LIMIT %s
                """,
                [*like_params, *kb_ids, *like_params, limit],
            ),
        ]
        for sql, params in like_sqls:
            rows = self.execute_query_(sql, tuple(params), fetch=True)
            if rows:
                return [
                    {
                        "doc_id": row[0],
                        "kb_id": row[1],
                        "file_id": row[2],
                        "content": self._normalize_dm_text(row[3]),
                        "score": float(row[4]) if row[4] is not None else 0.0,
                    }
                    for row in rows
                ]
            if rows is None:
                continue
        return []

    def delete_fulltext_documents_by_ids(self, doc_ids: List[str]) -> int:
        if not doc_ids:
            return 0
        deleted = 0
        batch_size = 200
        for i in range(0, len(doc_ids), batch_size):
            batch_doc_ids = doc_ids[i:i + batch_size]
            placeholders = ",".join(["%s"] * len(batch_doc_ids))
            query = f"DELETE FROM DocumentFulltextIndex WHERE doc_id IN ({placeholders})"
            res = self.execute_query_(query, tuple(batch_doc_ids), commit=True, check=True)
            if res:
                deleted += int(res)
        return deleted

    def delete_fulltext_documents_by_file_ids(self, file_ids: List[str]) -> int:
        if not file_ids:
            return 0
        deleted = 0
        batch_size = 200
        for i in range(0, len(file_ids), batch_size):
            batch_file_ids = file_ids[i:i + batch_size]
            placeholders = ",".join(["%s"] * len(batch_file_ids))
            query = f"DELETE FROM DocumentFulltextIndex WHERE file_id IN ({placeholders})"
            res = self.execute_query_(query, tuple(batch_file_ids), commit=True, check=True)
            if res:
                deleted += int(res)
        return deleted

    def get_fulltext_backfill_source_documents(self) -> List[Dict]:
        rows = self.execute_query_("SELECT doc_id, json_data FROM Documents", (), fetch=True) or []
        results = []
        for row in rows:
            doc_id = row[0]
            raw_json = self._normalize_dm_text(row[1])
            try:
                doc_json = json.loads(raw_json)
            except Exception:
                continue
            kwargs = doc_json.get("kwargs", {})
            metadata = kwargs.get("metadata", {})
            kb_id = metadata.get("kb_id")
            file_id = metadata.get("file_id")
            content = kwargs.get("page_content", "")
            if not kb_id or not file_id:
                continue
            results.append(
                {
                    "doc_id": doc_id,
                    "kb_id": kb_id,
                    "file_id": file_id,
                    "content": content,
                }
            )
        return results
