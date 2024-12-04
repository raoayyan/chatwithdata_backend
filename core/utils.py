from pymongo import MongoClient
import os
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
import ast
from pymongo.errors import PyMongoError
def verify_db_connection():
    try:
        MONGO_CLIENT = MongoClient("mongodb://localhost:27017/")
        # The ismaster command is cheap and does not require auth.
        MONGO_CLIENT.admin.command('ismaster')
        print("✅ MongoDB connection successful")
        return True
    except Exception as e:
        print("❌ MongoDB connection failed:", str(e))
        return False

MONGO_CLIENT = MongoClient("mongodb://localhost:27017/")
DB = MONGO_CLIENT["chatwithdata"]
EXPLANATIONS_COLLECTION = DB["db-explanation"]

llm = ChatGroq(
    model="llama3-groq-70b-8192-tool-use-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    
    )

# ffor simple 1level schema extraction but providing sample it will get all the sense of data in nested
# def extract_mongo_schema(db_uri, db_name):
#     client = MongoClient(db_uri)
#     db = client[db_name]
#     schema_data = []

#     for collection_name in db.list_collection_names():
#         collection = db[collection_name]
#         samples = list(collection.find().limit(2))  # Fetch 2 sample documents
#         schema = {"collection": collection_name, "fields": list(samples[0].keys()) if samples else []}
#         schema_data.append({"schema": schema, "samples": samples})

#     return schema_data

def extract_mongo_schema(db_uri, db_name):
    client = MongoClient(db_uri)
    db = client[db_name]
    schema_data = []

    def extract_nested_schema(document, path=''):
        schema = {}
        for key, value in document.items():
            full_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                # Recursively extract nested schema
                schema[key] = extract_nested_schema(value, full_path)
            elif isinstance(value, list):
                # Handle array types
                if value and isinstance(value[0], dict):
                    schema[key] = extract_nested_schema(value[0], f"{full_path}[]")
                else:
                    schema[key] = 'array'
            else:
                schema[key] = 'field'
        
        return schema

    for collection_name in db.list_collection_names():
        collection = db[collection_name]
        samples = list(collection.find().limit(2))  # Fetch 2 sample documents
        
        if samples:
            # Extract full schema including nested structures
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

# import sqlite3

# def extract_sql_schema(db_path):
#     connection = sqlite3.connect(db_path)
#     cursor = connection.cursor()

#     schema_data = []
#     cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#     tables = cursor.fetchall()

#     for table in tables:
#         table_name = table[0]
#         cursor.execute(f"PRAGMA table_info({table_name});")
#         columns = [row[1] for row in cursor.fetchall()]
        
#         cursor.execute(f"SELECT * FROM {table_name} LIMIT 2;")
#         samples = cursor.fetchall()

#         schema_data.append({"table": table_name, "fields": columns, "samples": samples})

#     return schema_data

from groq import Groq

# Initialize the Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def generate_explanations_with_llama(schema_data):
    explanations = []

    for entry in schema_data:
        schema = entry['schema']
        samples = entry['samples']

        # Construct the prompt
        prompt = (
            f"Given the schema: {schema} and the following sample data: {samples}, "
            "write a short explanation of what this table or collection represents.original schema may contain short form like e_name for employee name e_no employee number. you  have to explain each using your best knowledge and by looking at sample data. and  if there is some nested schemas do explain them also "
        )

        # Groq API call
        completion = client.chat.completions.create(
            model="llama3-groq-70b-8192-tool-use-preview",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.5,
            max_tokens=1024,
            top_p=0.65,
            stream=True,
        )

        # Collect streamed response
        explanation = ""
        for chunk in completion:
            explanation += chunk.choices[0].delta.content or ""

        explanations.append({"schema": schema, "explanation": explanation.strip()})

    return explanations


def save_explanations_to_mongodb(user_id, db_name, explanations):
    # Prepare the document with user_id and db_name
    explanation_doc = {
        "user_id": user_id,
        "db_name": db_name,
        "is_finalized": False,
        "schemas": []  # This will be an array to store multiple schemas
    }
    
    # Populate the schemas array
    for entry in explanations:
        
        explanation_doc["schemas"].append({
            "schema": entry["schema"],
            "explanation": entry["explanation"]
        })
    
    # Insert the entire document
    EXPLANATIONS_COLLECTION.insert_one(explanation_doc)


def get_database_explanation(database_name):
    explanation = EXPLANATIONS_COLLECTION.find_one({"db_name": database_name})
    if explanation:
        return explanation
    else:
        raise ValueError("No explanation found for the given database.")


# def execute_query(database_name, query):
#     database = mongo_client[database_name]
#     try:
#         # Example for executing a find query
#         collection_name, query_details = parse_query(query)  # Define a function to parse query details
#         result = list(database[collection_name].find(query_details))
#         return result
#     except Exception as e:
#         raise ValueError(f"Error executing query: {str(e)}")

def execute_query(database_name, query):
    """
    Execute a dynamically generated MongoDB query string on the given database.

    Args:
        database_name (str): Name of the MongoDB database.
        query (Union[str, object]): Query string or object containing the MongoDB syntax.

    Returns:
        list or int: Query result.

    Raises:
        ValueError: If query format is invalid or execution fails.
    """
    database = MONGO_CLIENT[database_name]

    try:
        # Extract the query string if it's an object (e.g., AIMessage)
        if hasattr(query, "content"):
            query = query.content

        print("Query received:", query)

        # Ensure query is a string
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")

        # Split the query to identify the collection and operation
        if query.startswith("db."):
            # Extract collection name and operation
            parts = query.split(".", 2)  # Split only on the first two dots
            _, collection_name, operation_with_args = parts

            # Handle `collection.operation(args)`
            operation, args = operation_with_args.split("(", 1)
            args = args.rstrip(")")  # Remove closing parenthesis

            # Parse arguments safely
            if args.strip():  # Check if arguments are provided
                try:
                    parsed_args = ast.literal_eval(args)
                except (SyntaxError, ValueError):
                    raise ValueError(f"Invalid arguments: {args}")
            else:
                parsed_args = {}  # No arguments provided

            print(f"Parsed: collection={collection_name}, operation={operation}, arguments={parsed_args}")

            # Select the collection
            collection = database[collection_name]

            # Perform the requested operation
            if operation == "find":
                result = list(collection.find(parsed_args))
            elif operation == "count_documents":
                result = collection.count_documents(parsed_args)
            elif operation == "aggregate":
                result = list(collection.aggregate(parsed_args))
            else:
                raise ValueError(f"Unsupported operation: {operation}")

            print("Query Execution Result:", result)
            return result

        else:
            raise ValueError("Query does not start with 'db.' and is not valid MongoDB syntax.")

    except PyMongoError as e:
        raise ValueError(f"MongoDB error: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error executing query: {str(e)}")


def generate_query(user_query, explanation):
    
    prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("""
            Using the following database explanation:
            {explanation}

            In explanation you have all the data like database name collecion full schemas , Genearte me just one time full mongodb query or aggregation pipeline if you think ,now as you have all of this you have to generate accurate query that i can run on that database and get intented data output plus i am connected to database so dont worry about is we have to generate the mongo query and if you think query is complex do generate mongodb aggregation pipeline for given below user query whcih is in english. please dont provide any english just provide query or aggregation nothin else:
            {user_query}
            
            here is some example for your learning:
            query:Name of customers whose order total amount is 29.99.
            generated response: content='db.orders.aggregate([\n {\n "$match": {\n "total_amount": 29.99\n }\n },\n {\n "$lookup": {\n "from": "customers",\n "localField": "customer_id",\n "foreignField": "_id",\n "as": "customer_details"\n }\n },\n {\n "$unwind": {\n "path": "$customer_details"\n }\n },\n {\n "$project": {\n "customer_name": "$customer_details.name"\n }\n }\n])'


            query :all those custtomers name whose order total amount is grater than 30
            generated response:content='db.orders.aggregate([\n  {\n    "$lookup": {\n      "from": "customers",\n      "localField": "customer_id",\n      "foreignField": "_id",\n      "as": "customer_info"\n    }\n  },\n  {\n    "$match": {\n      "total_amount": {\n        "$gt": 30\n      }\n    }\n  },\n  {\n    "$project": {\n      "customer_name": "$customer_info.name"\n    }\n  }\n])'
        """)
    ])
    
    # Combine prompt and LLM into a sequence
    chain = prompt | llm

    # Use `invoke` instead of `run`
    response = chain.invoke({"explanation": explanation, "user_query": user_query})
    print("response of query generation :",response)
    return response

# def generate_query(user_query, explanation):
#     prompt = ChatPromptTemplate.from_messages([
#         HumanMessagePromptTemplate.from_template("""
#             Using the following database explanation:
#             {explanation}

#             In the explanation, you have all the data like database name, collection schemas, etc.
#             Now, generate a valid MongoDB query for the following user query in plain English:
#             {user_query}
#         """)
#     ])
    
#     chain = prompt | llm
#     response = chain.invoke({"explanation": explanation, "user_query": user_query})
    
#     # Extract the query from the response
#     if "```python" in response.content:
#         query = response.content.split("```python")[1].split("```")[0].strip()
#     else:
#         raise ValueError("Query not found in response content.")

#     print("Extracted Query:", query)
#     return query

def refine_output(user_query, query_result):
    prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("""
            Based on the following:
            User Query: {user_query}
            Query Result: {query_result}

            Provide the response in a user-friendly format.
        """)
    ])
    
     # Combine prompt and LLM into a sequence
    chain = prompt | llm
    refined_response = chain.invoke({
        "user_query": user_query,
        "query_result": query_result
    })

    # Extract the content from the LLM response
    return refined_response.content

    