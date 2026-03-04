from copy import deepcopy
from enum import Enum
from typing import List, Dict
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.sql.ddl import CreateTable
from .vector import VectorAdaptor, VECTOR
from typing import Optional, Union, Any, Tuple
from .types import ROWID
from sqlalchemy.sql import text
from sqlalchemy.dialects.mysql import insert as mysql_insert
from .extensions import DMDialect_Adapter, DMMySQLDialect_Adapter, DMTSQLDialect_Adapter
from sqlalchemy import create_engine, Column, MetaData, Table, insert, delete, and_, select
from sqlalchemy import Boolean, Integer, BigInteger, String, Float, Double, Text, JSON, BLOB, Date, ARRAY, BINARY, Interval, TIMESTAMP

class IntEnum(int, Enum):
    pass

class ErrorCode(IntEnum):
    SUCCESS = 0
    UNEXPECTED_ERROR = 1
    INVALID_ARGUMENT = 2
    NOT_SUPPORTED = 3
    COLLECTION_NOT_FOUND = 100
    SEARCH_INDEX_ERROR = 700

class DataType(IntEnum):
    INT = 1
    BOOL = 2
    JSON = 3
    TEXT = 4
    BLOB = 5
    DATE = 6
    INT64 = 7
    ARRAY = 8
    FLOAT = 9
    ROWID = 10
    BIGINT = 11
    DOUBLE = 12
    STRING = 13
    BINARY = 14
    VARCHAR = 15
    INTERVAL = 16
    TIMESTAMP = 17
    INT8_VECTOR = 18
    FLOAT_VECTOR = 19
    FLOAT32_VECTOR = 20
    FLOAT64_VECTOR = 21
    SPARSE_FLOAT_VECTOR = 22  # ?
    UNKNOWN = 999

def convert_datatype_to_sqltype(datatype: DataType):
    if datatype == DataType.INT:
        return Integer
    if datatype == DataType.BOOL:
        return Boolean
    if datatype == DataType.JSON:
        return JSON
    if datatype == DataType.TEXT:
        return Text
    if datatype == DataType.BLOB:
        return BLOB
    if datatype == DataType.DATE:
        return Date
    if datatype == DataType.ARRAY:
        return ARRAY
    if datatype == DataType.FLOAT:
        return Float
    if datatype == DataType.INT64:
        return BigInteger
    if datatype == DataType.ROWID:
        return ROWID
    if datatype == DataType.BIGINT:
        return BigInteger
    if datatype == DataType.DOUBLE:
        return Double
    if datatype == DataType.STRING:
        return String
    if datatype == DataType.BINARY:
        return BINARY
    if datatype == DataType.VARCHAR:
        return String
    if datatype == DataType.INTERVAL:
        return Interval
    if datatype == DataType.TIMESTAMP:
        return TIMESTAMP
    if datatype == DataType.INT8_VECTOR:
        return VECTOR
    if datatype == DataType.FLOAT_VECTOR:
        return VECTOR
    if datatype == DataType.FLOAT32_VECTOR:
        return VECTOR
    if datatype == DataType.FLOAT64_VECTOR:
        return VECTOR
    if datatype == DataType.SPARSE_FLOAT_VECTOR:
        return VECTOR
    raise ValueError(f"Invalid DataType: {datatype}")

class dmException(Exception):
    def __init__(
        self,
        code: int = ErrorCode.UNEXPECTED_ERROR,
        message: str = "",
    ) -> None:
        super().__init__()
        self._code = code
        self._message = message


    @property
    def code(self):
        return self._code

    @property
    def message(self):
        return self._message

    def __str__(self) -> str:
        return f"<{type(self).__name__}: (code={self.code}, message={self.message})>"

class DataTypeNotSupportException(dmException):
    """Raise when datatype isn't supported"""

class PrimaryKeyException(dmException):
    """Raise when primarykey are invalid"""

class AutoIDException(dmException):
    """Raise when autoID is invalid"""

class PartitionKeyException(dmException):
    """Raise when partitionkey are invalid"""

class ClusteringKeyException(dmException):
    """Raise when clusteringkey are invalid"""

class ParamError(dmException):
    """Raise when params are incorrect"""

class VectorFieldParamException(dmException):
    """Raise when Vector Field parameters are invalid"""

class VarcharFieldParamException(dmException):
    """Raise when Varchar Field parameters are invalid"""

class CollectionStatusException(dmException):
    """Raise when collection status is invalid"""

class VectorNotFound(dmException):
    """Raise when vector is not found"""

class FoundMultipleIndexes(dmException):
    """Raise when vector is not found"""

class VectorMetricTypeException(dmException):
    """Raise when vector metric type is invalid"""

COMMON_TYPE_PARAMS = (
    "dim",
    "max_length",
    "max_capacity",
    "enable_match",
    "enable_analyzer",
    "analyzer_params",
    "multi_analyzer_params",
)

class ExceptionsMessage:
    IndexNotFound = "Index not found"
    FieldDtype = "Field dtype must be of DataType"
    AutoIDType = "Param auto_id must be bool type."
    CollectionNotExists = "Collection does not exist."
    IsPrimaryType = "Param is_primary must be bool type."
    IsPartitionKeyType = "Param is_partition_key must be bool type"
    PrimaryFieldType = "Param primary_field must be int or str type."
    IsClusteringKeyType = "Param is_clustering_key must be bool type."
    MetricTypeParamTypeInvalid = "MetricType param type should be string."
    VectorFieldMissingDimParam = "Param 'dim' must be set for vector field."
    AutoStringType = "The string type cannot be set as an auto-increment column"
    AutoIDOnlyOnPK = "The auto_id can only be specified on the primary key field"
    VarcharFieldMissingLengthParam = "Param 'max_length' must be set for varchar field."
    DefaultValueInvalid = "Default value cannot be None for a field that is defined as nullable == false."
    TooManyIndexFound = "More than one index has been found. Please confirm whether the schema name needs to be specified"
    MetricTypeValueInvalid = "MetricType should be 'IP'/'COSINE'/'HAMMING'/'L2'/'MANHATTAN'/'EUCLIDEAN_SQUARED' in ann search."

class FieldSchema:
    def __init__(
        self,
        name: str,
        dtype: DataType,
        description: str = "",
        is_primary: bool = False,
        auto_id: bool = False,
        nullable: bool = False,
        **kwargs,
    ) -> None:
        self.name = name
        self.dtype = dtype
        self.description = description
        self.is_primary = is_primary
        self.auto_id = auto_id
        self.nullable = nullable
        self.column_schema = None
        self.kwargs = kwargs
        self.type_params = {}
        self._check_primary_key_datatype()
        self._parse_type_params()

    def _check_primary_key_datatype(self):
        if not self.is_primary:
            return
        if self.dtype not in (
            DataType.INT,
            DataType.BIGINT,
            DataType.INT64,
            DataType.ROWID,
            DataType.FLOAT,
            DataType.STRING,
            DataType.VARCHAR,
        ):
            raise PrimaryKeyException(
                code=ErrorCode.INVALID_ARGUMENT,
                message=ExceptionsMessage.PrimaryFieldType,
            )

    def _parse_type_params(self):
        if self.dtype in [DataType.FLOAT_VECTOR, DataType.FLOAT32_VECTOR, DataType.FLOAT64_VECTOR]:
            if "dim" not in self.kwargs:
                raise VectorFieldParamException(
                    code=ErrorCode.INVALID_ARGUMENT,
                    message=ExceptionsMessage.VectorFieldMissingDimParam,
                )
            self.type_params["dim"] = self.kwargs["dim"]

    def parse_to_sql_column(self):
        self.column_schema = Column(
            self.name,
            convert_datatype_to_sqltype(self.dtype)(**self.type_params),
            primary_key=self.is_primary,
            autoincrement=self.auto_id,
            nullable=self.nullable,
        )

class VecIndexType(Enum):
    HNSW = 0
    IVF_FLAT = 1

def isVectorDataType(datatype: DataType) -> bool:
    return datatype in (
        DataType.INT8_VECTOR,
        DataType.FLOAT_VECTOR,
        DataType.FLOAT32_VECTOR,
        DataType.FLOAT64_VECTOR,
        DataType.SPARSE_FLOAT_VECTOR,
    )

class CollectionSchema:

    def __init__(
            self,
            fields: List,
            description: str = "",
            **kwargs,
    ):
        self.kwargs = deepcopy(kwargs)
        self.description = description
        if fields is not None:
            self.fields = [deepcopy(field) for field in fields]
        else:
            self.fields = []
        self.adjust_fields()

    def add_field(self, field_name: str, datatype: DataType, **kwargs):
        field = FieldSchema(field_name, datatype, **kwargs)
        cur_idx = len(self.fields)
        self.fields.append(field)
        self.fields[cur_idx].parse_to_sql_column()

    def adjust_fields(self):
        for field in self.fields:
            field.parse_to_sql_column()

class IndexParam:
    def __init__(self, field_name: str, index_type: str, index_name: str, **kwargs):

        self.field_name = field_name
        self.index_name = index_name
        self.is_primary = False
        self.metric_name = kwargs.get("metric_name", None)
        self.skip_existing = kwargs.get("skip_existing", False)
        self.percentage_value = kwargs.get("percentage_value", 90)
        if type(index_type) is str:
            self.index_type = index_type.upper()
        else:
            self.index_type = index_type.name.upper()
        if self.index_type == "IVF_FLAT":
            self.num_of_partitions = kwargs.get("num_of_partitions", None)
        elif self.index_type == "HNSW":
            self.ef_construction = kwargs.get("ef_construction", None)
            self.max_connection = kwargs.get("max_connection", None)
        else:
            raise ValueError("Currently, only the reconstruction of IVF_FLAT index and HNSW index is supported in DM")

    def to_dict(self) -> Dict:
        if self.index_type == "IVF_FLAT":
            return {
                "field_name": self.field_name,
                "index_type": self.index_type,
                "index_name": self.index_name,
                "is_primary": False,
                "percentage_value": self.percentage_value,
                "num_of_partitions": self.num_of_partitions,
            }
        elif self.index_type == "HNSW":
            return {
                "field_name": self.field_name,
                "index_type": self.index_type,
                "index_name": self.index_name,
                "is_primary": False,
                "percentage_value": self.percentage_value,
                "ef_construction": self.ef_construction,
                "max_connection": self.max_connection,
            }
        else:
            raise ValueError("Currently only IVF_FLAT index and HNSW index are supported in DM")

    def __str__(self):
        return str(self.to_dict())

    __repr__ = __str__

    def __eq__(self, other: None):
        if isinstance(other, self.__class__):
            return dict(self) == dict(other)

        if isinstance(other, dict):
            return dict(self) == other
        return False

class IndexParams(list):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_index(self, field_name: str, index_type: str = "", index_name: str = "", **kwargs):
        index_param = IndexParam(field_name, index_type, index_name, **kwargs)
        super().append(index_param)

def validate_param(
    param_name: str, param: Any, expected_type: Union[type, Tuple[type, ...]]
) -> None:
    if param is None:
        msg = f"missing required argument: [{param_name}]"
        raise ParamError(message=msg)

    if not isinstance(param, expected_type):
        msg = (
            f"wrong type of argument [{param_name}], "
            f"expected type: [{expected_type.__name__}], "
            f"got type: [{type(param).__name__}]"
        )
        raise ParamError(message=msg)

class dmVecClient:

    def __init__(
        self,
        uri: str = "localhost:5236",
        user: str = "SYSDBA",
        password: str = "",
        db_name: str = "", #schema
        echo: bool = False,
        connect_args: dict = {},
        token: str = "", # not used in dm
        timeout: Optional[float] = 0,
        **kwargs,
    ) -> None:
        connection_str = (
            f"dm+dmPython://{user}:{password}@{uri}/{db_name}"
        )
        self.engine = create_engine(connection_str, echo=echo, connect_args=connect_args)
        self.conn = self.engine.connect()
        self.connection = self.conn.connection.connection
        self.connection.connection_timeout = timeout
        self.connection_timeout = deepcopy(timeout)
        self.vec_adapter = VectorAdaptor(self.engine)
        self.metadata_obj = MetaData()
        self.dialect = self.engine.dialect

    @classmethod
    def create_schema(cls, **kwargs):
        kwargs["check_fields"] = False
        return CollectionSchema([], **kwargs)

    def _set_timeout(self, timeout):
        if timeout != self.connection_timeout:
            self.connection.connection_timeout = timeout

    def _reset_timeout(self, timeout):
        if timeout != self.connection_timeout:
            self.connection.connection_timeout = self.connection_timeout

    def _execute_sql(self, sql, params, timeout):
        self._set_timeout(timeout)

        if params is None:
            res = self.conn.execute(text(sql))
        else:
            res = self.conn.execute(text(sql), parameters=params)

        self._reset_timeout(timeout)

        return res

    def _get_func(self, metric_type, column):
        if metric_type == "IP":
            return column.inner_product_negative
        elif metric_type == "COSINE":
            return column.cosine_distance
        elif metric_type == "HAMMING":
            return column.hamming_distance
        elif metric_type == "L2":
            return column.l2_distance
        elif metric_type == "MANHATTAN":
            return column.l1_distance
        elif metric_type == "EUCLIDEAN_SQUARED":
            return column.l2_s_distance
        else:  # default to cosine
            return column.cosine_distance

    def create_collection(
        self,
        collection_name: str,
        dimension: Optional[int] = None,
        primary_field_name: str = "id",  # default is "id"
        id_type: str = "int",  # or "string",
        metric_type: str = "COSINE",
        vector_field_name: str = "vector",  # default is  "vector"
        auto_id: bool = False,
        timeout: Optional[float] = 0,
        schema: Optional[CollectionSchema] = None,
        index_params: Optional[IndexParams] = None,
        **kwargs,
    ):
        self._set_timeout(timeout)
        if schema is None:
            self._create_collection_without_schema(collection_name, dimension, primary_field_name, id_type, metric_type,
                                         vector_field_name, auto_id, index_params)
        else:
            self._create_collection_with_schema(collection_name, metric_type, schema, index_params)
        self._reset_timeout(timeout)

    def prepare_index_params(self):
        return IndexParams()

    def _create_collection_with_schema(
            self,
            collection_name: str,
            metric_type: str = "COSINE",
            schema: Optional[CollectionSchema] = None,
            index_params: Optional[IndexParams] = None
    ):
        columns = []
        for field in schema.fields:
            columns.append(field.column_schema)

        self.create_table(
            table_name=collection_name,
            columns=columns,
            metric_type=metric_type,
            vidxs=index_params,
        )

    def _create_collection_without_schema(
            self,
            collection_name: str,
            dimension: int,
            primary_field_name: str = "id",  # default is "id"
            id_type: Union[DataType, str] = DataType.INT,  # or "string",
            metric_type: str = "COSINE",
            vector_field_name: str = "vector",  # default is  "vector"
            auto_id: bool = False,
            index_params: Optional[IndexParams] = None,
    ):
        if dimension is None:
            raise VectorFieldParamException(
                code=ErrorCode.INVALID_ARGUMENT,
                message=ExceptionsMessage.VectorFieldMissingDimParam,
            )

        if id_type == DataType.INT:
            id_column = Column(
                primary_field_name,
                Integer(),
                primary_key=True,
                autoincrement=auto_id,
            )
        elif id_type == DataType.BIGINT:
            id_column = Column(
                primary_field_name,
                BigInteger(),
                primary_key=True,
                autoincrement=auto_id,
            )
        else:
            if auto_id:
                raise VectorFieldParamException(
                    code=ErrorCode.INVALID_ARGUMENT,
                    message=ExceptionsMessage.AutoStringType,
                )
            else:
                id_column = Column(
                primary_field_name,
                String(8188),
                primary_key=True,
            )
        vector_column = Column(vector_field_name, VECTOR(dimension))
        columns = [id_column, vector_column]
        self.create_table(
            table_name=collection_name,
            columns=columns,
            metric_type=metric_type,
            vidxs=index_params,
        )

    def check_idx(self, columns, vidxs: Optional[IndexParams], metric_type):
        name_lst = []
        for col in columns:
            name_lst.append(col.name)
        if vidxs is None:
            return None
        else:
            for vidx in vidxs:
                if vidx.field_name not in name_lst:
                    raise ParamError(message="The field_name of the index to be created is not within the range of column names in the table")
                else:
                    idx = name_lst.index(vidx.field_name)
                    vidx.column = columns[idx]
                vidx.index_type = vidx.index_type.upper()
                if vidx.index_type not in ('HNSW', 'IVF_FLAT'):
                    raise ParamError(
                        message="Currently only IVF_FLAT index and HNSW index are supported in DM")
                if vidx.metric_name is None:
                    vidx.metric_name = metric_type

    def create_table(self, table_name, columns, metric_type, vidxs):
        table = Table(table_name, self.metadata_obj, *columns)
        create_table_sql = str(CreateTable(table).compile(self.engine))
        self.conn.execute(text(create_table_sql))
        self.check_idx(columns, vidxs, metric_type)
        if vidxs is not None:
            for vidx in vidxs:
                if vidx is not None:
                    if vidx.index_type == 'HNSW':
                        self.vec_adapter.client_create_vector_hnsw_index(vidx)
                    else:
                        self.vec_adapter.client_create_vector_ivf_index(vidx)

    def insert(
        self,
        collection_name: str,
        data: Union[dict, list[dict]],
        timeout: Optional[float] = 0,
        partition_name: Optional[str] = "", # not used in dm
    ):
        if isinstance(data, dict):
            data = [data]
        if len(data) == 0:
            return
        table = self._set_timeout_and_reflect(collection_name, timeout)

        self.conn.execute(insert(table).values(data))
        self._reset_timeout(timeout)

    def get_collection_stats(self, collection_name: str, timeout: Optional[float] = 0) -> Dict:

        temp_table_name = deepcopy(collection_name)

        temp_table_name = self.dialect.denormalize_name(self.dialect.identifier_preparer.quote(temp_table_name))

        get_stats_sql = f"SELECT COUNT(*) as row_count FROM {temp_table_name}"

        res = self._execute_sql(get_stats_sql, None, timeout)

        cnt = [r[0] for r in res][0]
        return {"row_count": cnt}

    def commit(self):
        self.conn.commit()

    def drop_collection(self, collection_name: str, timeout: Optional[float] = 0, **kwargs):
        self._set_timeout(timeout)
        self.drop_table_if_exist(table_name=collection_name)
        self._reset_timeout(timeout)

    def drop_table_if_exist(self, table_name: str):
        try:
            table = Table(table_name, self.metadata_obj, autoload_with=self.engine)
        except NoSuchTableError as e:
            return

        table.drop(self.engine, checkfirst=True)
        self.metadata_obj.remove(table)

    def has_collection(self, collection_name: str, timeout: Optional[float] = 0, **kwargs):
        get_if_exists_sql = "SELECT a_objects.object_name, a_objects.object_type\n"\
                "FROM all_objects a_objects\n"\
                "WHERE a_objects.owner = :schema\n"\
                "AND a_objects.object_type IN ('VIEW', 'TABLE', 'MATERIALIZED VIEW')\n"\
                "AND a_objects.object_name = :table_name;"
        collection_name = self.dialect.denormalize_name(collection_name)
        schema_name = self.dialect.denormalize_name(self.dialect.default_schema_name)
        params = {"table_name": collection_name}
        params['schema'] = schema_name

        res = self._execute_sql(get_if_exists_sql, params, timeout)
        if res.rowcount == 0:
            return False
        else:
            return True

    def rename_collection(
            self, old_name: str, new_name: str, timeout: Optional[float] = 0  # pylint: disable=unused-argument
    ) -> None:

        old_name = self.dialect.denormalize_name(self.dialect.identifier_preparer.quote(old_name))
        new_name = self.dialect.denormalize_name(self.dialect.identifier_preparer.quote(new_name))
        rename_table_sql = f"ALTER TABLE {old_name} RENAME TO {new_name}"

        self._execute_sql(rename_table_sql, None, timeout)

    def load_table(self, collection_name: str):
        try:
            table = Table(collection_name, self.metadata_obj, autoload_with=self.engine)
        except NoSuchTableError as e:
            raise CollectionStatusException(
                code=ErrorCode.COLLECTION_NOT_FOUND,
                message=ExceptionsMessage.CollectionNotExists
            ) from e

        return table

    def create_index(
        self,
        collection_name: str,
        index_params: IndexParams,
        timeout: Optional[float] = 0,
        **kwargs,
    ):
        self._set_timeout(timeout)
        table = self.load_table(collection_name)
        columns = table.columns
        self.check_idx(columns, index_params, "COSINE")
        for index_param in index_params:
            if index_param is not None:
                if index_param.index_type == 'HNSW':
                    self.vec_adapter.client_create_vector_hnsw_index(index_param)
                elif index_param.index_type == 'IVF_FLAT':
                    self.vec_adapter.client_create_vector_ivf_index(index_param)
                else:
                    raise ValueError("Currently only IVF_FLAT index and HNSW index are supported in DM")

        self._reset_timeout(timeout)

    def drop_index(self, index_name: str, timeout: Optional[float] = 0, collection_name: str = None, **kwargs):
        index_name = self.dialect.denormalize_name(self.dialect.identifier_preparer.quote(index_name))
        drop_index_sql = f"DROP INDEX {index_name}"

        self._execute_sql(drop_index_sql, None, timeout)

    def rebuild_index(
            self,
            collection_name: str,
            index_name: str,
            timeout: Optional[float] = 0,
            schema_name: str = None,
            metric_name: str = None,
            target_accuracy: int = 0,
            cluster_centers: int = 0,
            percentage_value: int = 0,
            max_connection: int = 0,
            ef_construction: int = 0,
            **kwargs
    ):
        self._set_timeout(timeout)
        index_name = self.dialect.denormalize_name(index_name)
        collection_name = self.dialect.denormalize_name(collection_name)
        schema_name = self.dialect.denormalize_name(self.dialect.default_schema_name if schema_name is None else schema_name)
        index_type = self._check_vector_index(schema_name, index_name, collection_name)
        if index_type == 'HNSW':
            rebuild_sql = ("CALL SP_REBUILD_VECTOR_HNSW_INDEX(:schema_name, :index_name, "
                 ":metric_name, :percentage_value, :max_connection, :ef_construction);")
            self.conn.execute(
                text(rebuild_sql).bindparams(schema_name=schema_name, index_name=index_name, metric_name=metric_name,
                                           percentage_value=percentage_value, max_connection=max_connection,
                                           ef_construction=ef_construction))
        elif index_type == 'IVFFLAT':
            rebuild_sql = ("CALL SP_REBUILD_VECTOR_IVFFLAT_INDEX(:schema_name, :index_name, "
                         ":metric_name, :target_accuracy, :cluster_centers);")
            self.conn.execute(
                text(rebuild_sql).bindparams(schema_name=schema_name, index_name=index_name, metric_name=metric_name,
                                           target_accuracy=target_accuracy, cluster_centers=cluster_centers))
        else:
            raise ValueError("Currently, only the reconstruction of IVF_FLAT index and HNSW index is supported in DM")

        self._reset_timeout(timeout)

    def _check_vector_index(
            self,
            schema_name,
            index_name,
            collection_name,
    ):
        params = {"index_name": index_name}
        params['collection_name'] = collection_name
        check_sql = "SELECT INDEX_TYPE FROM ALL_INDEXES WHERE INDEX_NAME = :index_name AND TABLE_NAME = :collection_name"
        check_sql += " AND OWNER = :schema_name"
        params['schema_name'] = schema_name
        result = self.conn.execute(text(check_sql), parameters=params)
        if result.rowcount == 0:
            raise VectorNotFound(code=ErrorCode.SEARCH_INDEX_ERROR,message=ExceptionsMessage.IndexNotFound)
        else:
            index_type = result.first().index_type

        if index_type.upper().startswith('VECTOR '):
            index_type = index_type.upper()[7:]
        return index_type

    def _set_timeout_and_reflect(self, collection_name, timeout):
        self._set_timeout(timeout)

        try:
            table = Table(collection_name, self.metadata_obj, autoload_with=self.engine)
        except NoSuchTableError as e:
            raise CollectionStatusException(
                code=ErrorCode.COLLECTION_NOT_FOUND,
                message=ExceptionsMessage.CollectionNotExists
            ) from e

        return table

    def _compile_where_in_stmt(self, table, stmt, ids, filter):
        where_in_clause = None
        if ids is not None:
            primary_keys = table.primary_key
            pkey_names = [column.name for column in primary_keys]
            pkey_len = len(pkey_names)
            if isinstance(ids, list):
                where_in_clause = table.c[pkey_names[0]].in_(ids)
            elif isinstance(ids, (str, int)):
                where_in_clause = table.c[pkey_names[0]].in_([ids])
            elif isinstance(ids, dict):
                if len(ids) > pkey_len:
                    raise TypeError("The length of the 'ids' field of type 'dict' exceeds the number of primary keys in the composite primary key")
                else:
                    keys = list(ids.keys())
                    if keys[0] not in pkey_names:
                        raise ValueError("The name of the key is not within the scope of the composite primary key")
                    where_in_clause = table.c[keys[0]] == (ids[keys[0]])
                    for i in range(len(keys) - 1):
                        if keys[i + 1] not in pkey_names:
                            raise ValueError("The name of the key is not within the scope of the composite primary key")
                        where_in_clause = and_(where_in_clause, table.c[keys[i + 1]] == (ids[keys[i + 1]]))
            else:
                raise TypeError("'ids' is not a list/str/int")

        if where_in_clause is None and filter is None:
            result = stmt
        elif where_in_clause is not None and filter is None:
            result = stmt.where(where_in_clause)
        elif where_in_clause is None and filter is not None:
            result = stmt.where(*filter)
        else:
            result = stmt.where(and_(where_in_clause, *filter))

        return result

    def delete(
            self,
            collection_name: str,
            ids: Optional[Union[dict, list, str, int]] = None,
            timeout: Optional[float] = 0,
            filter=None,
            partition_name: Optional[str] = "",
            **kwargs,
    ) -> dict:
        table = self._set_timeout_and_reflect(collection_name, timeout)

        delete_clause = delete(table)

        stmt = self._compile_where_in_stmt(table, delete_clause, ids, filter)
        result = self.conn.execute(stmt)

        self._reset_timeout(timeout)
        return {"delete_count": result.rowcount}

    def upsert(
        self,
        collection_name: str,
        data: Union[dict, list[dict]],
        timeout: Optional[float] = 0,
        partition_name: Optional[str] = "",
    ) -> Dict:

        table = self._set_timeout_and_reflect(collection_name, timeout)
        if isinstance(data, dict):
            data = [data]

        if len(data) == 0:
            return None

        need_set_identity = False
        upsert_stmt = mysql_insert(table)
        stmt = upsert_stmt.values(data)
        update_dict = {}
        if isinstance(self.dialect.parse_module, DMDialect_Adapter):
            org_parse_type = 'DM'
        elif isinstance(self.dialect.parse_module, DMMySQLDialect_Adapter):
            org_parse_type = 'MySQL'
        elif isinstance(self.dialect.parse_module, DMTSQLDialect_Adapter):
            org_parse_type = 'TSQL'
        else:
            raise ValueError('Undefined parse_type')
        if org_parse_type != 'MySQL':
            for col in data[0].keys():
                if col in table.c:
                    update_dict[col] = stmt.inserted[col]
            if table.autoincrement_column is not None:
                identity_col = table.autoincrement_column.name
                if identity_col in data[0].keys():
                    need_set_identity = True
        table_name = self.dialect.denormalize_name(table.name)
        temp_table_name = deepcopy(table_name)
        temp_table_name = self.dialect.identifier_preparer.quote(temp_table_name)
        upsert_stmt = stmt.on_duplicate_key_update(**update_dict)
        if need_set_identity:
            self.conn.execute(text(f"SET IDENTITY_INSERT {temp_table_name} ON WITH REPLACE NULL;"))
        if org_parse_type != 'MySQL':
            self.conn.execute(text("SP_SET_SESSION_PARSE_TYPE('MySQL');"))
        result = self.conn.execute(upsert_stmt)
        res = {"upsert_count": result.rowcount}
        if org_parse_type != 'MySQL':
            self.conn.execute(text(f"SP_SET_SESSION_PARSE_TYPE('{org_parse_type}');"))
        if need_set_identity:
            self.conn.execute(text(f"SET IDENTITY_INSERT {temp_table_name} OFF;"))
        self._reset_timeout(timeout)

        return res

    def get(
        self,
        collection_name: str,
        ids: Union[list, str, int],
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = 0,
        partition_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[dict]:
        result = self.query(collection_name, "", output_fields, timeout, ids, partition_names, **kwargs)
        return result

    def query(
            self,
            collection_name: str,
            filter: str = "",
            output_fields: Optional[List[str]] = None,
            timeout: Optional[float] = 0,
            ids: Optional[Union[List, str, int]] = None,
            partition_names: Optional[List[str]] = None,
            **kwargs,
    ):
        table = self._set_timeout_and_reflect(collection_name, timeout)

        if output_fields is not None:
            columns = [table.c[column_name] for column_name in output_fields]
            select_clause = select(*columns)
        else:
            select_clause = select(table)

        if isinstance(ids, (int, str)):
            ids = [ids]

        stmt = self._compile_where_in_stmt(table, select_clause, ids, filter)
        exec_res = self.conn.execute(stmt)
        data_res = exec_res.fetchall()
        columns = list(exec_res.keys())

        return [
            {
                columns[i]:  value
                for i, value in enumerate(row)
            }
            for row in data_res
        ]

    def search(
        self,
        collection_name: str,
        data: Optional[Union[List[list], list]] = None, #运算的数据
        filter: str = "",
        limit: int = 10,
        with_dist: bool = False,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[dict] = None,
        timeout: Optional[float] = 0,
        partition_names: Optional[List[str]] = None, #not used in dm
        anns_field: Optional[str] = None,
        ids: Optional[Union[List[int], List[str], str, int]] = None,
        **kwargs,
    ) -> list[dict]:

        if not (isinstance(data, list) or isinstance(data, dict)):
            raise ValueError("'data' type must be 'list'")

        table = self._set_timeout_and_reflect(collection_name, timeout)

        metric_type_str = "COSINE"
        if search_params is not None:
            if "metric_type" in search_params:
                if not isinstance(search_params["metric_type"], str):
                    raise VectorMetricTypeException(
                        code=ErrorCode.INVALID_ARGUMENT,
                        message=ExceptionsMessage.MetricTypeParamTypeInvalid,
                    )
                metric_type_str = search_params["metric_type"].upper()
                if metric_type_str not in (
                    "IP", "COSINE", "HAMMING", "L2", "MANHATTAN", "EUCLIDEAN_SQUARED"
                ):
                    raise VectorMetricTypeException(
                        code=ErrorCode.INVALID_ARGUMENT,
                        message=ExceptionsMessage.MetricTypeValueInvalid,
                    )

        if anns_field is None and search_params is not None and isinstance(search_params["anns_field"], str):
            anns_field = search_params["anns_field"]

        vec_col = table.c[anns_field]
        vec_sel_con = self._get_func(metric_type_str, vec_col)
        order_col = vec_sel_con(data)

        if output_fields is not None:
            columns = [table.c[column_name] for column_name in output_fields]
        else:
            columns = [table.c[column.name] for column in table.columns]

        if with_dist:
            if isinstance(data, list):
                columns.append(order_col.label("distance"))

        select_clause = select(*columns)

        stmt = self._compile_where_in_stmt(table, select_clause, ids, filter)
        stmt = stmt.order_by(order_col).limit(limit)
        limit_clause = stmt.selectable._limit_clause
        if limit_clause is not None:
            limit_clause.approx_select = True

        exec_res = self.conn.execute(stmt)
        data_res = exec_res.fetchall()
        columns = list(exec_res.keys())

        res = [
            {
                columns[i]:  value
                for i, value in enumerate(row)
            }
            for row in data_res
        ]
        if with_dist:
            return sorted(res, key=lambda x: x[columns[-1]])
        return res
