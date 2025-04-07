import os
import ast
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import psycopg
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain

# ----------------- MongoDB Setup ----------------- #
MONGO_CLIENT = MongoClient("mongodb://localhost:27017/")
DB = MONGO_CLIENT["chatwithdata"]
EXPLANATIONS_COLLECTION = DB["db-explanation"]

def verify_db_connection():
    try:
        MONGO_CLIENT.admin.command('ismaster')
        print("✅ MongoDB connection successful")
        return True
    except Exception as e:
        print("❌ MongoDB connection failed:", str(e))
        return False

# ----------------- LLM Init ----------------- #
from groq import Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

llm = ChatGroq(
    model="llama3-groq-70b-8192-tool-use-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# ----------------- MongoDB Schema Extraction ----------------- #
def extract_mongo_schema(db_uri, db_name):
    client = MongoClient(db_uri)
    db = client[db_name]
    schema_data = []

    def extract_nested_schema(document, path=''):
        schema = {}
        for key, value in document.items():
            full_path = f"{path}.{key}" if path else key
            if isinstance(value, dict):
                schema[key] = extract_nested_schema(value, full_path)
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    schema[key] = extract_nested_schema(value[0], f"{full_path}[]")
                else:
                    schema[key] = 'array'
            else:
                schema[key] = 'field'
        return schema

    for collection_name in db.list_collection_names():
        collection = db[collection_name]
        samples = list(collection.find().limit(2))
        if samples:
            full_schema = extract_nested_schema(samples[0])
            schema_entry = {
                "schema": {
                    "collection": collection_name,
                    "full_schema": full_schema
                },
                "samples": samples
            }
            schema_data.append(schema_entry)

    return schema_data

# ----------------- PostgreSQL Schema Extraction ----------------- #
def extract_postgres_schema():
    conn = psycopg.connect("dbname=dvdrental user=postgres password=Faisal host=localhost port=5432")
    cur = conn.cursor()
    schema_data = []

    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public';
    """)
    tables = cur.fetchall()

    for table in tables:
        table_name = table[0]
        cur.execute(f"SELECT * FROM {table_name} LIMIT 2;")
        samples = cur.fetchall()
        col_names = [desc[0] for desc in cur.description]

        schema_data.append({
            "table": table_name,
            "columns": col_names,
            "samples": [dict(zip(col_names, row)) for row in samples]
        })

    cur.close()
    conn.close()
    return schema_data

# ----------------- LLM-based Explanation Generation ----------------- #
def generate_explanations_with_llama(schema_data, db_type='mongo'):
    explanations = []

    for entry in schema_data:
        if db_type == 'mongo':
            schema = entry['schema']
            samples = entry['samples']
            prompt = (
                f"Given the schema: {schema} and the following sample data: {samples}, "
                "write a short explanation of what this collection represents, including any nested structures."
            )
        else:  # postgres
            schema = {
                "table": entry["table"],
                "columns": entry["columns"]
            }
            samples = entry["samples"]
            prompt = (
                f"Given the SQL table: {entry['table']} with columns: {entry['columns']} and sample rows: {samples}, "
                "write a short explanation of what this table represents."
            )

        completion = client.chat.completions.create(
            model="llama3-groq-70b-8192-tool-use-preview",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.5,
            max_tokens=1024,
            stream=True,
        )

        explanation = ""
        for chunk in completion:
            explanation += chunk.choices[0].delta.content or ""

        explanations.append({"schema": schema, "explanation": explanation.strip()})

    return explanations

# ----------------- MongoDB Explanation Storage ----------------- #
def save_explanations_to_mongodb(user_id, db_name, explanations):
    explanation_doc = {
        "user_id": user_id,
        "db_name": db_name,
        "is_finalized": False,
        "schemas": []
    }
    for entry in explanations:
        explanation_doc["schemas"].append({
            "schema": entry["schema"],
            "explanation": entry["explanation"]
        })
    EXPLANATIONS_COLLECTION.insert_one(explanation_doc)

def get_database_explanation(database_name):
    explanation = EXPLANATIONS_COLLECTION.find_one({"db_name": database_name})
    if explanation:
        return explanation
    else:
        raise ValueError("No explanation found for the given database.")

# ----------------- MongoDB Query Execution ----------------- #
def execute_query(database_name, query):
    database = MONGO_CLIENT[database_name]
    try:
        if hasattr(query, "content"):
            query = query.content
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")
        if query.startswith("db."):
            parts = query.split(".", 2)
            _, collection_name, operation_with_args = parts
            operation, args = operation_with_args.split("(", 1)
            args = args.rstrip(")")
            parsed_args = ast.literal_eval(args) if args.strip() else {}
            collection = database[collection_name]
            if operation == "find":
                result = list(collection.find(parsed_args))
            elif operation == "count_documents":
                result = collection.count_documents(parsed_args)
            elif operation == "aggregate":
                result = list(collection.aggregate(parsed_args))
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            return result
        else:
            raise ValueError("Invalid MongoDB query format.")
    except PyMongoError as e:
        raise ValueError(f"MongoDB error: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error executing query: {str(e)}")

# ----------------- PostgreSQL Query Execution ----------------- #
def execute_postgres_query(query):
    conn = psycopg.connect("dbname=dvdrental user=postgres password=Faisal host=localhost port=5432")
    cur = conn.cursor()
    try:
        cur.execute(query)
        if cur.description:
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        return {"rows_affected": cur.rowcount}
    except Exception as e:
        raise ValueError(f"PostgreSQL execution failed: {str(e)}")
    finally:
        cur.close()
        conn.close()

# ----------------- LLM Query Generator ----------------- #
def generate_query(user_query, explanation, db_type='mongo'):
    template = """
        Using the following database explanation:
        {explanation}

        Generate a {db_type} query for this user question:
        {user_query}

        Return just the query or aggregation pipeline, no explanations.
    """
    prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template(template)
    ])
    chain = prompt | llm
    response = chain.invoke({
        "explanation": explanation,
        "user_query": user_query,
        "db_type": "MongoDB" if db_type == 'mongo' else "SQL"
    })
    return response

# ----------------- LLM Output Refinement ----------------- #
def refine_output(user_query, query_result):
    prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("""
            Based on the following:
            User Query: {user_query}
            Query Result: {query_result}
            Provide the response in a user-friendly format.
        """)
    ])
    chain = prompt | llm
    refined_response = chain.invoke({
        "user_query": user_query,
        "query_result": query_result
    })
    return refined_response.content
