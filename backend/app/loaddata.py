import pandas as pd
from pathlib import Path
import glob
from llama_index.core.settings import Settings
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.program import LLMTextCompletionProgram
from app.createDatabase import create_table_from_dataframe
from sqlalchemy.orm import session
from sqlalchemy import (
    create_engine,
    MetaData)

from app import database_models


# def init_oracledb():
#     oracledb_settings = get_oracledb_settings()

#     username = oracledb_settings.ORACLE_DB_USERNAME
#     password = oracledb_settings.ORACLE_DB_PASSWORD
#     dsn = oracledb_settings.ORACLE_DB_DSN

#     engine_oracle = create_engine("oracle+oracledb://{username}:{password}@{dsn}")

#     def get_db():
#         # init_oracledb()
#         db = Session(engine_oracle)
#         try:
#             yield db
#         finally:
#             db.close()


class TableInfo(BaseModel):
        """Information regarding a structured table."""

        table_name: str = Field(
            ..., description="table name (must be underscores and NO spaces)"
        )
        table_summary: str = Field(
            ..., description="short, concise summary/caption of the table"
        )

tableinfo_dir = "WikiTableQuestions_TableInfo"
# !mkdir {tableinfo_dir}



from app.settings  import get_oracledb_settings
from sqlalchemy import (
    create_engine,
    MetaData)
from app import database_models

from sqlalchemy.orm import Session

oracledb_settings = get_oracledb_settings()


username = oracledb_settings.ORACLE_DB_USERNAME
password = oracledb_settings.ORACLE_DB_PASSWORD
dsn = oracledb_settings.ORACLE_DB_DSN
print("*********************")
print(username+ password)

engine = create_engine(f"oracle+oracledb://{username}:{password}@{dsn}")
# database_models.Base.metadata.create_all(bind=engine_oracledb)

# engine = create_engine("sqlite:///:memory:")

def getEngine(): 
    return engine

def _get_tableinfo_with_index(idx: int) -> str:
    results_gen = Path(tableinfo_dir).glob(f"{idx}_*")
    results_list = list(results_gen)
    if len(results_list) == 0:
        return None
    elif len(results_list) == 1:
        path = results_list[0]
        return TableInfo.parse_file(path)
    else:
        raise ValueError(
            f"More than one file matching index: {list(results_gen)}"
        )

def loadData():
    data_dir = Path("./WikiTableQuestions/csv/200-csv")
    csv_files = sorted([f for f in data_dir.glob("*.csv")])
    print("result")
    dfs = []
    # for csv_file in csv_files:
    #     print(f"processing file: {csv_file}")
    #     try:
    #         df = pd.read_csv(csv_file)
    #         dfs.append(df)
    #     except Exception as e:
    #         print(f"Error parsing {csv_file}: {str(e)}")

    prompt_str = """\
    Give me a summary of the table with the following JSON format.

    - The table name must be unique to the table and describe it while being concise. 
    - Do NOT output a generic table name (e.g. table, my_table).

    Do NOT make the table name one of the following: {exclude_table_name_list}

    Table:
    {table_str}

    Summary: """

    program = LLMTextCompletionProgram.from_defaults(
        output_cls=TableInfo,
        llm=Settings.llm,
        prompt_template_str=prompt_str,
    )

    table_names = set()
    table_infos = []
    # for idx, df in enumerate(dfs):
    #     table_info = _get_tableinfo_with_index(idx)
    #     if table_info:
    #         table_infos.append(table_info)
    #     else:
    #         while True:
    #             df_str = df.head(10).to_csv()
    #             table_info = program(
    #                 table_str=df_str,
    #                 exclude_table_name_list=str(list(table_names)),
    #             )
    #             table_name = table_info.table_name
    #             print(f"Processed table: {table_name}")
    #             if table_name not in table_names:
    #                 table_names.add(table_name)
    #                 break
    #             else:
    #                 # try again
    #                 print(f"Table name {table_name} already exists, trying again.")
    #                 break 

    #         out_file = f"{tableinfo_dir}/{idx}_{table_name}.json"
    #         json.dump(table_info.dict(), open(out_file, "w"))
    #     table_infos.append(table_info)
    # engine = create_engine("sqlite:///:memory:")
    metadata_obj = MetaData()
    for idx, df in enumerate(dfs):
        tableinfo = _get_tableinfo_with_index(idx)
        print(f"Creating table: {tableinfo.table_name}")
        create_table_from_dataframe(df, tableinfo.table_name, engine, metadata_obj)
    return createObjectIndex(table_infos, engine, metadata_obj)

    
import json

def createObjectIndex(table_infos, engine, metadata_obj): 
    from llama_index.core.objects import (
        SQLTableNodeMapping,
        ObjectIndex,
        SQLTableSchema,
    )
    from llama_index.core import SQLDatabase, VectorStoreIndex
    
    sql_database = SQLDatabase(engine)

    
    table_node_mapping = SQLTableNodeMapping(sql_database)
    table_schema_objs = [
        SQLTableSchema(table_name=t.table_name, context_str=t.table_summary)
        for t in table_infos
    ]  # add a SQLTableSchema for each table
    addMoreTables(metadata_obj, engine, table_schema_objs)

    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
    )
    obj_retriever = obj_index.as_retriever(similarity_top_k=3)

    from llama_index.core.retrievers import SQLRetriever
    from typing import List
    from llama_index.core.query_pipeline import FnComponent

    sql_retriever = SQLRetriever(sql_database)


    def get_table_context_str(table_schema_objs: List[SQLTableSchema]):
        """Get table context string."""
        context_strs = []
        for table_schema_obj in table_schema_objs:
            table_info = sql_database.get_single_table_info(
                table_schema_obj.table_name
            )
            print("table = "+table_schema_obj.table_name + "\n")
            print(table_info)

            if table_schema_obj.context_str:
                table_opt_context = " The table description is: "
                table_opt_context += table_schema_obj.context_str
                table_info += table_opt_context

            context_strs.append(table_info)
        return "\n\n".join(context_strs)


    table_parser_component = FnComponent(fn=get_table_context_str)

    from llama_index.core.retrievers import SQLRetriever
    from typing import List
    from llama_index.core.query_pipeline import FnComponent

    sql_retriever = SQLRetriever(sql_database)

    from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
    from llama_index.core import PromptTemplate
    from llama_index.core.query_pipeline import FnComponent
    from llama_index.core.llms import ChatResponse


    def parse_response_to_sql(response: ChatResponse) -> str:
        """Parse response to SQL."""
        response = response.message.content
        sql_query_start = response.find("SQLQuery:")
        if sql_query_start != -1:
            response = response[sql_query_start:]
            # TODO: move to removeprefix after Python 3.9+
            if response.startswith("SQLQuery:"):
                response = response[len("SQLQuery:") :]
        sql_result_start = response.find("SQLResult:")
        if sql_result_start != -1:
            response = response[:sql_result_start]
        return response.strip().strip("```").strip()


    sql_parser_component = FnComponent(fn=parse_response_to_sql)

    DEFAULT_TEXT_TO_SQL_TMPL = (
    "Given an input question, first create a syntactically correct {dialect} "
    "query to run, then look at the results of the query and return the answer. "
    "You can order the results by a relevant column to return the most "
    "interesting examples in the database.\n\n"
    "Never query for all the columns from a specific table, only ask for a "
    "few relevant columns given the question.\n\n"
    "Relevant columns can be found using the Column's name, it's doc, comment and info's label property.\n"
    "Pay attention to use only the column names that you can see in the schema "
    "description. "
    "Be careful to not query for columns that do not exist. "
    "Pay attention to which column is in which table. "
    "Also, qualify column names with the table name when needed. "
    "You are required to use the following format, each taking one line:\n\n"
    "Question: Question here\n"
    "SQLQuery: SQL Query to run\n"
    "SQLResult: Result of the SQLQuery\n"
    "Answer: Final answer here\n\n"
    "Only use tables listed below.\n"
    "{schema}\n\n"
    "Question: {query_str}\n"
    "SQLQuery: "
)


    # text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(
    #     dialect=engine.dialect.name
    # )

    text2sql_prompt= PromptTemplate(DEFAULT_TEXT_TO_SQL_TMPL).partial_format(dialect=engine.dialect.name)
    print(text2sql_prompt.template)

    response_synthesis_prompt_str = (
        "Given an input question, synthesize a response from the query results. \n"
        "Query: {query_str}\n"
        "SQL: {sql_query}\n"
        "SQL Response: {context_str}\n"
        "Response: "
    )
    response_synthesis_prompt = PromptTemplate(
        response_synthesis_prompt_str,
    )

    sql_response_synthesis_prompt_str = (
        "Given an input question, synthesize a response from the query results.\n"
        "Query: {query_str}\n"
        "SQL: {sql_query}\n"
        "Response: "
    )
    sql_response_synthesis_prompt = PromptTemplate(
        sql_response_synthesis_prompt_str,
    )


    from llama_index.core.query_pipeline import (
        QueryPipeline as QP,
        Link,
        InputComponent,
        CustomQueryComponent,
    )
    input_prompt_str = "create an sql query for the question {question} and "
    input_prompt_templ = PromptTemplate(input_prompt_str)
    

    with_retriver = False
    # with_retriver = True
    modules = {
            "input": input_prompt_templ,
            "text2sql_llm": Settings.llm,
            "text2sql_prompt": text2sql_prompt,
            "sql_output_parser": sql_parser_component,
            "table_retriever": obj_retriever,
            "table_output_parser": table_parser_component,
            "response_synthesis_llm": Settings.llm,
        }
    
    if(with_retriver is False):
        modules["sql_response_synthesis_prompt"]= sql_response_synthesis_prompt
    else:
        modules["response_synthesis_prompt"]= response_synthesis_prompt
        modules["sql_retriever"] = sql_retriever


    qp = QP(
        modules= modules,
        verbose=True,
    )

    if( with_retriver is False):
        qp.add_chain(["input", "table_retriever", "table_output_parser"])
        qp.add_link("table_output_parser", "text2sql_prompt", dest_key="schema")
        qp.add_link("input", "text2sql_prompt", dest_key="query_str")
        qp.add_chain(["text2sql_prompt", "text2sql_llm", "sql_output_parser"])
        qp.add_link(
            "sql_output_parser", "sql_response_synthesis_prompt", dest_key="sql_query"
        )
        qp.add_link("input", "sql_response_synthesis_prompt", dest_key="query_str")
        qp.add_link(
            "sql_output_parser", "sql_response_synthesis_prompt", dest_key="sql_query"
        )
        qp.add_link("sql_response_synthesis_prompt", "response_synthesis_llm")
    else:
        qp.add_chain(["input", "table_retriever", "table_output_parser"])
        qp.add_link("input", "text2sql_prompt", dest_key="query_str")
        qp.add_link("table_output_parser", "text2sql_prompt", dest_key="schema")
        qp.add_chain(
            ["text2sql_prompt", "text2sql_llm", "sql_output_parser", "sql_retriever"]
        )
        qp.add_link(
            "sql_output_parser", "response_synthesis_prompt", dest_key="sql_query"
        )
        qp.add_link(
            "sql_retriever", "response_synthesis_prompt", dest_key="context_str"
        )
        qp.add_link("input", "response_synthesis_prompt", dest_key="query_str")
        qp.add_link("response_synthesis_prompt", "response_synthesis_llm")
    

    response = qp.run(
        # query="who is the current officer in the office of State Auditor?"
        query="who is the customer and his mother's name and daughter's name from USA with sales amount greater than 50000?"
        # query= "Generate a comprehensive report on Tom's banking activities, including their account balances, recent transactions, loan details, and payments, all tied to the specific branch and employees managing their accounts."
        # query="give me list of all the movies which have won an award"
        # query = "give me list of all people who have won an award for \"Best Actor\" category  "
        # query="Tell me information about notting hill movie"
        # query="who is the Public Regulation Commissioner"
    )
    print(str(response))
    return qp



def addMoreTables(metadata_obj, engine, table_schema_objs): 
    from llama_index.core.objects import (
        SQLTableSchema,
    )
    from sqlalchemy import (
            Table,
            Column,
            String,
            Integer,
            ForeignKey,
            Float,
            Date
        )
    city_stats_table = Table(
        "city_stats",
        metadata_obj,
        Column("city_name", String(16), primary_key=True),
        Column("population", Integer),
        Column("country", String(16), nullable=False),
    )

    # Define the Country table
    country_table = Table(
        'Country',
        metadata_obj,
        Column("country_code", String(16), primary_key=True),
        Column("country_name", String(50), nullable=False)
    )

    # Define the Year table
    year_table = Table(
        'Year',
        metadata_obj,
        Column("year_id", Integer, primary_key=True),
        Column("year", Integer, nullable=False, unique=True)
    )

    # Define the Customer table with a foreign key to Country
    customer_table = Table(
        'SalesCustomer',
        metadata_obj,
        Column("customer_id", Integer, primary_key=True),
        Column("customer_name", String(50), nullable=False),
        Column("f_01", String(50), comment="Gardener's name"),
        Column("f_04", String(50), comment="son's name"),
        Column("f_07", String(50), comment="daughter's name"),
        Column("f_08", String(50), comment="spouse's name"),
        Column("f_09", String(50), comment="mother's name"),
        Column("country", String(16), ForeignKey('Country.country_code'), nullable=False)
    )

    # Define the Sales table with foreign keys to Customer, Country, and Year
    sales_table = Table(
        'Sales',
        metadata_obj,
        Column("sale_id", Integer, primary_key=True),
        Column("customer_id", Integer, ForeignKey('SalesCustomer.customer_id'), nullable=False),
        Column("country", String(16), ForeignKey('Country.country_code'), nullable=False),
        Column("year_id", Integer, ForeignKey('Year.year_id'), nullable=False),
        Column("amount", Integer, nullable=False)
    )
    
    metadata_obj.create_all(engine)
    table_schema_objs.append(SQLTableSchema(table_name="Country" ))
    table_schema_objs.append(SQLTableSchema(table_name="Year" ))
    table_schema_objs.append(SQLTableSchema(table_name="SalesCustomer" ))
    table_schema_objs.append(SQLTableSchema(table_name="Sales" ))


    customers_table = Table(
        'Customers',
        metadata_obj,
        Column("customer_id", Integer, primary_key=True),
        Column("name", String(50), nullable=False),
        Column("address", String(100), nullable=False),
        Column("phone_number", String(15), nullable=False, unique=True),
        Column("email", String(50), nullable=False, unique=True)
    )

    # Define the Branches table
    branches_table = Table(
        'Branches',
        metadata_obj,
        Column("branch_id", Integer, primary_key=True),
        Column("branch_name", String(50), nullable=False),
        Column("address", String(100), nullable=False)
    )

    # Define the Accounts table with a foreign key to Customers and Branches
    accounts_table = Table(
        'Accounts',
        metadata_obj,
        Column("account_id", Integer, primary_key=True),
        Column("account_number", String(20), nullable=False, unique=True),
        Column("customer_id", Integer, ForeignKey('Customers.customer_id'), nullable=False),
        Column("branch_id", Integer, ForeignKey('Branches.branch_id'), nullable=False),
        Column("account_type", String(10), nullable=False),  # e.g., savings, checking
        Column("balance", Float, nullable=False)
    )

    # Define the Transactions table with foreign keys to Accounts
    transactions_table = Table(
        'Transactions',
        metadata_obj,
        Column("transaction_id", Integer, primary_key=True),
        Column("account_id", Integer, ForeignKey('Accounts.account_id'), nullable=False),
        Column("transaction_type", String(10), nullable=False),  # e.g., debit, credit
        Column("amount", Float, nullable=False),
        Column("transaction_date", Date, nullable=False)
    )

    # Define the Employee table with a foreign key to Branches
    employee_table = Table(
        'Employee',
        metadata_obj,
        Column("employee_id", Integer, primary_key=True),
        Column("name", String(50), nullable=False),
        Column("position", String(50), nullable=False),  # e.g., teller, manager
        Column("branch_id", Integer, ForeignKey('Branches.branch_id'), nullable=False)
    )

    # Define the Loans table with a foreign key to Customers and Branches
    loans_table = Table(
        'Loans',
        metadata_obj,
        Column("loan_id", Integer, primary_key=True),
        Column("loan_number", String(20), nullable=False, unique=True),
        Column("customer_id", Integer, ForeignKey('Customers.customer_id'), nullable=False),
        Column("branch_id", Integer, ForeignKey('Branches.branch_id'), nullable=False),
        Column("loan_type", String(10), nullable=False),  # e.g., personal, home, auto
        Column("loan_amount", Float, nullable=False),
        Column("loan_date", Date, nullable=False),
        Column("interest_rate", Float, nullable=False)
    )

    # Define the LoanPayments table with a foreign key to Loans
    loan_payments_table = Table(
        'LoanPayments',
        metadata_obj,
        Column("payment_id", Integer, primary_key=True),
        Column("loan_id", Integer, ForeignKey('Loans.loan_id'), nullable=False),
        Column("payment_date", Date, nullable=False),
        Column("payment_amount", Float, nullable=False)
    )
    # metadata_obj.drop_all(engine)
    metadata_obj.create_all(engine)
    table_schema_objs.append(SQLTableSchema(table_name="Customers"))
    table_schema_objs.append(SQLTableSchema(table_name="Branches"))
    table_schema_objs.append(SQLTableSchema(table_name="Accounts"))
    table_schema_objs.append(SQLTableSchema(table_name="Transactions"))
    table_schema_objs.append(SQLTableSchema(table_name="Employee"))
    table_schema_objs.append(SQLTableSchema(table_name="Loans"))
    table_schema_objs.append(SQLTableSchema(table_name="LoanPayments"))

    