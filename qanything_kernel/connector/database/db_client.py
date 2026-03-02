from qanything_kernel.configs.model_config import DB_TYPE

if str(DB_TYPE).lower() in ("dameng", "dm"):
    from qanything_kernel.connector.database.dameng.dameng_client import KnowledgeBaseManager
else:
    from qanything_kernel.connector.database.mysql.mysql_client import KnowledgeBaseManager

__all__ = ["KnowledgeBaseManager"]
