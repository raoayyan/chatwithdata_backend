# === Imports ===
# from pymongo import MongoClient
# from pymongo.errors import PyMongoError
import psycopg
import json
import re
import ast
import os
import bcrypt
import jwt
from datetime import datetime, timedelta
from typing import Union
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
from groq import Groq
# from bson.objectid import ObjectId

# === MongoDB Setup (commented) ===
# MONGO_CLIENT = MongoClient("mongodb://localhost:27017/")
# DB = MONGO_CLIENT["chatwithdata"]
# EXPLANATIONS_COLLECTION = DB["db-explanation"]

# === PostgreSQL Setup ===
POSTGRES_URI = "dbname=dvdrental user=postgres password=Faisal host=localhost port=5433"

# === LLM Setup ===
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

SECRET_KEY = "432874u5872"

# === PostgreSQL Functions ===
def extract_postgres_schema():
    conn = psycopg.connect(POSTGRES_URI)
    cur = conn.cursor()
    schema_data = []

    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
    tables = cur.fetchall()

    for table in tables:
        table_name = table[0]
        cur.execute(f"SELECT * FROM {table_name} LIMIT 2;")
        rows = cur.fetchall()
        col_names = [desc[0] for desc in cur.description]
        schema_data.append({
            "table": table_name,
            "columns": col_names,
            "samples": [dict(zip(col_names, row)) for row in rows]
        })

    cur.close()
    conn.close()
    return schema_data

def execute_postgres_query(sql_query):
    conn = psycopg.connect(POSTGRES_URI)
    cur = conn.cursor()
    try:
        cur.execute(sql_query)
        if cur.description:
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        else:
            return {"rows_affected": cur.rowcount}
    finally:
        cur.close()
        conn.close()

def generate_postgres_explanations(schema_data):
    explanations = []
    for table in schema_data:
        prompt = (
            f"Table: {table['table']}\n"
            f"Columns: {table['columns']}\n"
            f"Sample Rows: {table['samples']}\n"
            "Explain what this table represents in detail."
        )

        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.5,
            max_tokens=1024,
            stream=True,
        )

        explanation = ""
        for chunk in completion:
            explanation += chunk.choices[0].delta.content or ""

        explanations.append({
            "schema": {
                "table": table['table'],
                "columns": table['columns']
            },
            "explanation": explanation.strip()
        })

    return explanations

def generate_postgres_query(user_query, explanation):
    prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("""
        Given the following PostgreSQL database explanation:
        {explanation}

        Your task is to translate the user's natural language request into a valid PostgreSQL SQL query.

        ### Instructions:
        - Only return the SQL query string (no explanations).
        - Use proper table names and fields from the explanation.
        - Avoid placeholders unless necessary.
        - Do not add any markdown or commentary.

        ### Example
        User query: Find all customers from Canada.
        Output: SELECT * FROM customer WHERE country = 'Canada';

        Now, generate a SQL query for:
        {user_query}
        """)
    ])

    chain = prompt | llm
    response = chain.invoke({"explanation": explanation, "user_query": user_query})
    return response.content.strip()

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
    response = chain.invoke({
        "user_query": user_query,
        "query_result": query_result
    })
    return response.content
# === Auth for PostgreSQL (simple example) ===

USERS = []  # Replace this with a real PostgreSQL-backed model later

def signup(name: str, email: str, password: str):
    if any(u["email"] == email for u in USERS):
        return {"error": "Email already exists"}

    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    USERS.append({"name": name, "email": email, "password": hashed.decode()})
    return {"message": "User signed up successfully!"}

def sign_in(email: str, password: str):
    user = next((u for u in USERS if u["email"] == email), None)
    if not user or not bcrypt.checkpw(password.encode(), user["password"].encode()):
        return {"error": "Invalid email or password"}

    token = jwt.encode({
        "user_id": user["email"],  # or use a UUID if you prefer
        "email": user["email"],
        "exp": datetime.utcnow() + timedelta(hours=24)
    }, SECRET_KEY, algorithm="HS256")
    return {"message": "Sign-in successful", "token": token}

# === MongoDB-related Functions (commented out) ===
# def extract_mongo_schema(db_uri, db_name): ...
# def execute_query(database_name, query: Union[str, object]): ...
# def all_databases(): ...
# def check_database_exists(database_name: str) -> bool: ...
# def save_explanations_to_mongodb(user_id, db_name, explanations): ...
# def get_database_explanation(database_name): ...
# def generate_query(user_query, explanation): ...
# def store_chat(chat_id: str, query: str = None, response: str = None): ...
# def signup(name: str, email: str, password: str): ...
# def sign_in(email: str, password: str): ...
# def update_explanation_text(explanation_id, new_explanation): ...



# from pymongo import MongoClient
# from pymongo.errors import PyMongoError
# import psycopg
# import json, re, ast, os, bcrypt, jwt
# from datetime import datetime, timedelta
# from typing import Union
# from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
# from langchain_groq import ChatGroq
# from groq import Groq
# from bson.objectid import ObjectId


# # Mongo Setup
# MONGO_CLIENT = MongoClient("mongodb://localhost:27017/")
# DB = MONGO_CLIENT["chatwithdata"]
# EXPLANATIONS_COLLECTION = DB["db-explanation"]

# # PostgreSQL Setup
# POSTGRES_URI = "dbname=dvdrental user=postgres password=Faisal host=localhost port=5433"

# # LLM Client
# client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# llm = ChatGroq(
#     model="gemma2-9b-it",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )

# SECRET_KEY = "432874u5872"


# # def verify_db_connection():
# #     try:
# #         MONGO_CLIENT.admin.command('ismaster')
# #         print("✅ MongoDB connection successful")
# #         return True
# #     except Exception as e:
# #         print("❌ MongoDB connection failed:", str(e))
# #         return False


# # === MongoDB Functions ===

# def extract_mongo_schema(db_uri, db_name):
#     client = MongoClient(db_uri)
#     db = client[db_name]
#     schema_data = []

#     def extract_nested_schema(document, path=''):
#         schema = {}
#         for key, value in document.items():
#             full_path = f"{path}.{key}" if path else key
#             if isinstance(value, dict):
#                 schema[key] = extract_nested_schema(value, full_path)
#             elif isinstance(value, list):
#                 schema[key] = extract_nested_schema(value[0], f"{full_path}[]") if value and isinstance(value[0], dict) else 'array'
#             else:
#                 schema[key] = 'field'
#         return schema

#     for collection_name in db.list_collection_names():
#         collection = db[collection_name]
#         samples = list(collection.find().limit(2))
#         if samples:
#             full_schema = extract_nested_schema(samples[0])
#             schema_entry = {
#                 "schema": {
#                     "collection": collection_name,
#                     "full_schema": full_schema
#                 },
#                 "samples": samples
#             }
#             schema_data.append(schema_entry)

#     return schema_data


# def execute_query(database_name, query: Union[str, object]):
#     database = MONGO_CLIENT[database_name]

#     try:
#         if hasattr(query, "content"):
#             query = query.content

#         pattern = r'^db\.(\w+)\.(\w+)\((.*)\)\s*$'
#         match = re.match(pattern, query.strip(), re.DOTALL)
#         if not match:
#             raise ValueError("Query does not match expected MongoDB pattern.")

#         collection_name, operation, raw_args = match.groups()
#         raw_args = raw_args.strip()
#         try:
#             parsed_args = json.loads(raw_args) if raw_args else {}
#         except json.JSONDecodeError:
#             parsed_args = ast.literal_eval(raw_args) if raw_args else {}

#         collection = database[collection_name]
#         if operation == "find":
#             result = list(collection.find(parsed_args))
#         elif operation == "count_documents":
#             result = collection.count_documents(parsed_args)
#         elif operation == "aggregate":
#             if not isinstance(parsed_args, list):
#                 raise ValueError("Aggregate expects a list of pipeline stages.")
#             result = list(collection.aggregate(parsed_args))
#         else:
#             raise ValueError(f"Unsupported operation: {operation}")

#         return result

#     except PyMongoError as e:
#         raise ValueError(f"MongoDB error: {str(e)}")
#     except Exception as e:
#         raise ValueError(f"Error executing query: {str(e)}")


# # === PostgreSQL Functions ===

# def extract_postgres_schema():
#     conn = psycopg.connect(POSTGRES_URI)
#     cur = conn.cursor()
#     schema_data = []

#     cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
#     tables = cur.fetchall()

#     for table in tables:
#         table_name = table[0]
#         cur.execute(f"SELECT * FROM {table_name} LIMIT 2;")
#         rows = cur.fetchall()
#         col_names = [desc[0] for desc in cur.description]
#         schema_data.append({
#             "table": table_name,
#             "columns": col_names,
#             "samples": [dict(zip(col_names, row)) for row in rows]
#         })

#     cur.close()
#     conn.close()
#     return schema_data


# def execute_postgres_query(sql_query):
#     conn = psycopg.connect(POSTGRES_URI)
#     cur = conn.cursor()
#     try:
#         cur.execute(sql_query)
#         if cur.description:
#             columns = [desc[0] for desc in cur.description]
#             rows = cur.fetchall()
#             return [dict(zip(columns, row)) for row in rows]
#         else:
#             return {"rows_affected": cur.rowcount}
#     finally:
#         cur.close()
#         conn.close()


# def generate_postgres_explanations(schema_data):
#     explanations = []
#     for table in schema_data:
#         prompt = (
#             f"Table: {table['table']}\n"
#             f"Columns: {table['columns']}\n"
#             f"Sample Rows: {table['samples']}\n"
#             "Explain what this table represents in detail."
#         )

#         completion = client.chat.completions.create(
#             model="deepseek-r1-distill-llama-70b",
#             messages=[{"role": "system", "content": prompt}],
#             temperature=0.5,
#             max_tokens=1024,
#             stream=True,
#         )

#         explanation = ""
#         for chunk in completion:
#             explanation += chunk.choices[0].delta.content or ""

#         explanations.append({
#             "schema": {
#                 "table": table['table'],
#                 "columns": table['columns']
#             },
#             "explanation": explanation.strip()
#         })

#     return explanations


# # === Common Functionalities ===

# def all_databases():
#     db_names = EXPLANATIONS_COLLECTION.distinct("db_name")
#     return [{"db_name": name} for name in db_names if name]

# def check_database_exists(database_name: str) -> bool:
#     return any(db["db_name"] == database_name for db in all_databases())

# def save_explanations_to_mongodb(user_id, db_name, explanations):
#     doc = {
#         "user_id": user_id,
#         "db_name": db_name,
#         "is_finalized": False,
#         "schemas": [{"schema": e["schema"], "explanation": e["explanation"]} for e in explanations]
#     }
#     EXPLANATIONS_COLLECTION.insert_one(doc)

# def get_database_explanation(database_name):
#     doc = EXPLANATIONS_COLLECTION.find_one({"db_name": database_name})
#     if doc:
#         return doc
#     raise ValueError("No explanation found for the given database.")

# def generate_query(user_query, explanation):
#     prompt = ChatPromptTemplate.from_messages([
#         HumanMessagePromptTemplate.from_template("""
#         Using the following database explanation:
#         {explanation}

#         You have the schema. Generate a valid MongoDB aggregation pipeline ONLY for this user query:
#         {user_query}

#         Return ONLY the query starting with db.collection.aggregate([...]) and nothing else.
#         """)
#     ])
#     chain = prompt | llm
#     return chain.invoke({"explanation": explanation, "user_query": user_query})

# def refine_output(user_query, query_result):
#     prompt = ChatPromptTemplate.from_messages([
#         HumanMessagePromptTemplate.from_template("""
#         Based on the following:
#         User Query: {user_query}
#         Query Result: {query_result}

#         Provide the response in a user-friendly format.
#         """)
#     ])
#     chain = prompt | llm
#     return chain.invoke({
#         "user_query": user_query,
#         "query_result": query_result
#     }).content

# # === Auth and Chat Storage ===

# def store_chat(chat_id: str, query: str = None, response: str = None):
#     try:
#         collection = DB["chats"]
#         collection.update_one({"chat_id": chat_id}, {
#             "$setOnInsert": {"chat_id": chat_id, "queries": [], "responses": []}
#         }, upsert=True)
#         if query:
#             collection.update_one({"chat_id": chat_id}, {"$push": {"queries": query}})
#         if response:
#             collection.update_one({"chat_id": chat_id}, {"$push": {"responses": response}})
#         return {"message": "Chat stored successfully!"}
#     except Exception as e:
#         return {"error": str(e)}

# def signup(name: str, email: str, password: str):
#     users = DB["users"]
#     if users.find_one({"email": email}):
#         return {"error": "Email already exists"}
#     hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
#     users.insert_one({"name": name, "email": email, "password": hashed.decode()})
#     return {"message": "User signed up successfully!"}

# def sign_in(email: str, password: str):
#     user = DB["users"].find_one({"email": email})
#     if not user or not bcrypt.checkpw(password.encode(), user["password"].encode()):
#         return {"error": "Invalid email or password"}
#     token = jwt.encode({
#         "user_id": str(user["_id"]),
#         "email": user["email"],
#         "exp": datetime.utcnow() + timedelta(hours=24)
#     }, SECRET_KEY, algorithm="HS256")
#     return {"message": "Sign-in successful", "token": token}

# def generate_postgres_query(user_query, explanation):
#     """
#     Generate a PostgreSQL SQL query from a user query and explanation.
#     """
#     prompt = ChatPromptTemplate.from_messages([
#         HumanMessagePromptTemplate.from_template("""
#         Given the following PostgreSQL database explanation:
#         {explanation}

#         Your task is to translate the user's natural language request into a valid PostgreSQL SQL query.

#         ### Instructions:
#         - Only return the SQL query string (no explanations).
#         - Use proper table names and fields from the explanation.
#         - Avoid placeholders unless necessary.
#         - Do not add any markdown or commentary.

#         ### Example
#         User query: Find all customers from Canada.
#         Output: SELECT * FROM customer WHERE country = 'Canada';

#         Now, generate a SQL query for:
#         {user_query}
#         """)
#     ])

#     chain = prompt | llm
#     response = chain.invoke({"explanation": explanation, "user_query": user_query})

#     return response.content.strip()

# def update_explanation_text(explanation_id, new_explanation):
#     try:
#         result = EXPLANATIONS_COLLECTION.update_one(
#             {"_id": ObjectId(explanation_id)},
#             {"$set": {"schemas.0.explanation": new_explanation}}  # assumes explanation is inside schemas array
#         )
#         return result.modified_count > 0
#     except Exception as e:
#         print(f"Failed to update explanation: {e}")
#         return False
