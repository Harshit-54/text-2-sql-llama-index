# import os
# from IPython.display import Markdown, display
# from sqlalchemy import (
#     create_engine,
#     MetaData,
#     Table,
#     Column,
#     String,
#     Integer,
#     select,
# )

# from llama_index.core import SQLDatabase
# from sqlalchemy import insert
# from llama_index.core.settings import Settings

# engine = create_engine("sqlite:///:memory:")
# metadata_obj = MetaData()

# # create city SQL table
# table_name = "city_stats"
# city_stats_table = Table(
#     table_name,
#     metadata_obj,
#     Column("city_name", String(16), primary_key=True),
#     Column("population", Integer),
#     Column("country", String(16), nullable=False),
# )

# # create users SQL table
# users_table = Table(
#     "users",
#     metadata_obj,
#     Column("user_id", String(16), primary_key=True),
#     Column("first_name", String(16), nullable=False),
#     Column("last_name", String(16), nullable=False),
#     Column("address", String(16)),
#     Column("email", String(16))
# )

# metadata_obj.create_all(engine)


# sql_database = SQLDatabase(engine, include_tables=["city_stats", "users"])

# rows = [
#     {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
#     {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
#     {
#         "city_name": "Chicago",
#         "population": 2679000,
#         "country": "United States",
#     },
#     {"city_name": "Seoul", "population": 9776000, "country": "South Korea"},
# ]
# for row in rows:
#     stmt = insert(city_stats_table).values(**row)
#     with engine.begin() as connection:
#         cursor = connection.execute(stmt)

# # metadata_obj.create_all(engine)

# from llama_index.core.indices.struct_store.sql_query import (
#     SQLTableRetrieverQueryEngine,
# )
# from llama_index.core.objects import (
#     SQLTableNodeMapping,
#     ObjectIndex,
#     SQLTableSchema,
# )
# from llama_index.core import VectorStoreIndex

# table_node_mapping = SQLTableNodeMapping(sql_database)
# table_schema_objs = [
#     (SQLTableSchema(table_name="city_stats")),
#     (SQLTableSchema(table_name="users"))
# ]  # add a SQLTableSchema for each table

# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# Settings.embed_model = HuggingFaceEmbedding(
#         model_name="BAAI/bge-small-en-v1.5"
#     )
# from llama_index.llms.llama_api import LlamaAPI
# Settings.llm = LlamaAPI(api_key=os.getenv("LLAMAAPI_KEY"))

# obj_index = ObjectIndex.from_objects(
#     table_schema_objs,
#     table_node_mapping,
#     VectorStoreIndex,
#     embed_model=Settings.embed_model
# )
# query_engine = SQLTableRetrieverQueryEngine(
#     sql_database, obj_index.as_retriever(similarity_top_k=1), llm=Settings.llm
# )

# for t in metadata_obj.sorted_tables:
#      print(t.name)

