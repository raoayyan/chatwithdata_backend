from pymongo import MongoClient
import os
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
import ast
from pymongo.errors import PyMongoError
import json
import re
from typing import Union
import bcrypt
import jwt
from datetime import datetime, timedelta
from bson.objectid import ObjectId 
from groq import Groq
import psycopg

SECRET_KEY = "432874u5872"


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
    model="gemma2-9b-it",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    
    )

# Initialize the Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def all_databases():
    """
    Retrieves all unique 'db_name' and their corresponding 'db_type' values 
    from the 'db-explanation' collection.

    Returns:
        List[Dict[str, str]]: List of unique database names and types in 
        {'db_name': value, 'db_type': value} format.
    """
    try:
        # Retrieve all documents with db_name and db_type
        databases = EXPLANATIONS_COLLECTION.find({}, {"db_name": 1, "db_type": 1, "_id": 0})
        
        # Filter out documents without db_name or db_type and return the result
        return [{"db_name": db.get("db_name"), "db_type": db.get("db_type")} for db in databases if db.get("db_name") and db.get("db_type")]
    except Exception as e:
        raise RuntimeError(f"Failed to fetch database values: {str(e)}")

def check_database_exists(database_name: str) -> bool:
    """
    Checks whether a database with the given name exists in the MongoDB instance.

    Args:
        database_name (str): The name of the database to check.

    Returns:
        bool: True if the database exists, False otherwise.
    """
    try:
        # Retrieve all database names
        db_list = all_databases()
        # Check if the given database name exists in the list
        return any(db["db_name"] == database_name for db in db_list)
    except Exception as e:
        print(f"Error while checking database existence: {str(e)}")
        return False


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


def generate_explanations_with_llama(schema_data):
    explanations = []

    for entry in schema_data:
        schema = entry['schema']
        samples = entry['samples']

        # Construct the prompt
        prompt = (
            f"Given the schema: {schema} and the following sample data: {samples}, "
            "write a short explanation of what this table or collection represents.original schema may contain short form like e_name for employee name e_no employee number. you  have to explain each using your best knowledge and by looking at sample data. and  if there is some nested schemas do explain them also. Remember please generate shortt and concise explanation to the point "
        )

        # Groq API call
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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

def save_explanations_to_mongodb(db_type, db_name, explanations):
    # Prepare the document with user_id and db_name
    explanation_doc = {
        "db_type": db_type,
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

def execute_query(database_name, query: Union[str, object]):
    """
    Execute a MongoDB query generated by an LLM or passed directly as a string.

    Args:
        database_name (str): Name of the MongoDB database.
        query (Union[str, object]): Query string or object containing MongoDB syntax.

    Returns:
        list or int: Query result.

    Raises:
        ValueError: If the query format is invalid or execution fails.
    """
    database = MONGO_CLIENT[database_name]

    try:
        # Extract query content if needed
        if hasattr(query, "content"):
            query = query.content

        print("Query received:", query)

        if not isinstance(query, str):
            raise ValueError("Query must be a string.")

        # Match query like: db.collection.operation(args)
        pattern = r'^db\.(\w+)\.(\w+)\((.*)\)\s*$'
        match = re.match(pattern, query.strip(), re.DOTALL)

        if not match:
            raise ValueError("Query does not match expected MongoDB pattern.")

        collection_name, operation, raw_args = match.groups()

        # Clean and parse arguments
        raw_args = raw_args.strip()

        # Attempt to parse arguments as JSON
        try:
            # Fix single quotes and convert to valid JSON
            if operation == "aggregate":
                parsed_args = json.loads(raw_args)
            else:
                # Try to parse single document
                parsed_args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            # Fallback: Try ast.literal_eval if JSON fails (less strict)
            try:
                parsed_args = ast.literal_eval(raw_args) if raw_args else {}
            except Exception:
                raise ValueError(f"Invalid arguments: {raw_args}")

        print(f"Parsed: collection={collection_name}, operation={operation}, arguments={parsed_args}")

        # Execute the query
        collection = database[collection_name]

        if operation == "find":
            result = list(collection.find(parsed_args))
        elif operation == "count_documents":
            result = collection.count_documents(parsed_args)
        elif operation == "aggregate":
            if not isinstance(parsed_args, list):
                raise ValueError("Aggregate expects a list of pipeline stages.")
            result = list(collection.aggregate(parsed_args))
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        print("Query Execution Result:", result)
        return result

    except PyMongoError as e:
        raise ValueError(f"MongoDB error: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error executing query: {str(e)}")

def generate_query(user_query, explanation):
    
    prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("""
            Using the following database explanation:
            {explanation}

            You have all the data, including database name, collections, and their full schemas. Generate an accurate MongoDB aggregation pipeline for the following user query written in English:
            {user_query}

            ### Instructions:
            1. Ensure the query strictly adheres to the schema provided in the explanation.
            2. Use MongoDB operators (e.g., `{{"$lookup"}}`, `{{"$match"}}`, `{{"$project"}}`) where necessary.
            3. If the query is complex, prefer generating a well-structured aggregation pipeline.
            4. **Do not output any text other than the MongoDB pipeline.**
            5. always start query like db. no other thing attached.
            6. ONLY OUTPUT A VALID MONGO AGGREGATION PIPELINE IN JSON AND NOTHING ELSE!

            ### Example:
            User query: Find the names of all customers whose order total amount is greater than 30.

            Output:
            db.orders.aggregate([
            {{
                "$lookup": {{
                "from": "customers",
                "localField": "customer_id",
                "foreignField": "_id",
                "as": "customer_info"
                }}
            }},
            {{
                "$match": {{
                "total_amount": {{
                    "$gt": 30
                }}
                }}
            }},
            {{
                "$project": {{
                "customer_name": "$customer_info.name"
                }}
            }}
            ])

            Query: all those customers name whose have not placed any order
            db.customers.aggregate([
            {{
                "$lookup": {{
                "from": "orders",
                "localField": "_id",
                "foreignField": "customer_id",
                "as": "order_info"
                }}
            }},
            {{
                "$match": {{
                "order_info": {{
                    "$size": 0
                }}
                }}
            }},
            {{
                "$project": {{
                "customer_name": "$name"
                }}
            }}
            ])

            Now process the following:
            """)
    ])
    
    # Combine prompt and LLM into a sequence
    chain = prompt | llm

    # Use `invoke` instead of `run`
    response = chain.invoke({"explanation": explanation, "user_query": user_query})
    print("response of query generation :",response)
    return response

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

def store_chat(chat_id: str, db_name: str, query: str = None, response: str = None):
    """
    Stores chat data into the 'chats' collection in MongoDB.
    """
    try:
        chats_collection = DB["chats"]
        
        # First ensure the document exists with empty arrays
        chats_collection.update_one(
            {"chat_id": chat_id},
            {
                "$setOnInsert": {
                    "chat_id": chat_id,
                    "db_name": db_name,
                    "queries": [],
                    "responses": []
                }
            },
            upsert=True
        )

        # Then perform the array updates
        update_data = {}
        if query:
            update_data["$push"] = {"queries": query}
        if response:
            update_data["$push"] = {"responses": response}

        if update_data:
            chats_collection.update_one(
                {"chat_id": chat_id},
                update_data
            )

        return {"message": "Chat stored successfully!"}

    except Exception as e:
        print(f"Error storing chat: {str(e)}")
        return {"error": str(e)}

def signup(name: str, email: str, password: str):
    """
    Signs up a new user by storing their details in the 'users' collection.

    Args:
        name (str): The name of the user.
        email (str): The email of the user.
        password (str): The plain-text password of the user.

    Returns:
        dict: A success message or error message.
    """
    try:
        users_collection = DB["users"]

        # Check if the email already exists
        if users_collection.find_one({"email": email}):
            return {"error": "Email already exists"}

        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Create the user document
        user = {
            "name": name,
            "email": email,
            "password": hashed_password.decode('utf-8')  # Store the hashed password as a string
        }

        # Insert the user into the database
        users_collection.insert_one(user)

        return {"message": "User signed up successfully!"}

    except Exception as e:
        print(f"Error during signup: {str(e)}")
        return {"error": str(e)}

def sign_in(email: str, password: str):
    """
    Signs in a user by verifying their email and password.

    Args:
        email (str): The email of the user.
        password (str): The plain-text password of the user.

    Returns:
        dict: A success message with a token or an error message.
    """
    try:
        users_collection = DB["users"]

        # Find the user by email
        user = users_collection.find_one({"email": email})
        if not user:
            return {"error": "Invalid email or password"}

        # Verify the password
        if not bcrypt.checkpw(password.encode('utf-8'), user["password"].encode('utf-8')):
            return {"error": "Invalid email or password"}

        # Generate a token
        payload = {
            "user_id": str(user["_id"]),
            "email": user["email"],
            "exp": datetime.utcnow() + timedelta(hours=24)  # Token expires in 24 hours
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")

        return {"message": "Sign-in successful", "token": token}

    except Exception as e:
        print(f"Error during sign-in: {str(e)}")
        return {"error": str(e)}
    
def get_chat(db_name: str):
    """
    Retrieves all chats for a given database name from the 'chats' collection.

    Args:
        db_name (str): The name of the database to filter chats.

    Returns:
        dict: A success message with the list of chats or an error message.
    """
    try:
        chats_collection = DB["chats"]

        # Find all chats with the given db_name
        chats = list(chats_collection.find({"db_name": db_name}))

        # Convert ObjectId to string for JSON serialization
        for chat in chats:
            if "_id" in chat and isinstance(chat["_id"], ObjectId):
                chat["_id"] = str(chat["_id"])

        return {"message": "Chats retrieved successfully!", "chats": chats}

    except Exception as e:
        print(f"Error retrieving chats: {str(e)}")
        return {"error": str(e)}

def extract_sql_schema(db_name):
    """
    Connects to PostgreSQL and extracts schema for all public tables.
    Returns table name, fields, and 2 sample rows.
    """
    conn = psycopg.connect(
        dbname=db_name,
        user="postgres",
        password="postgres@11",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()

    schema_data = []
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema='public' AND table_type='BASE TABLE';
    """)
    tables = cur.fetchall()

    for table in tables:
        table_name = table[0]
        cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}';")
        columns = [row[0] for row in cur.fetchall()]
        
        cur.execute(f"SELECT * FROM {table_name} LIMIT 2;")
        samples = cur.fetchall()

        schema_data.append({
            "schema": {
                "table": table_name,
                "fields": columns
            },
            "samples": samples
        })

    cur.close()
    conn.close()
    return schema_data

def generate_sql_query(user_query: str, explanation: str):
    """
    Generate a PostgreSQL query from natural language using LLM.
    
    Args:
        user_query (str): Natural language query
        explanation (str): Database schema explanation
        
    Returns:
        str: Generated SQL query
    """
    prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("""
            Using the following database explanation:
            {explanation}

            You have all the data, including database name, tables, and their full schemas. 
            Generate an accurate PostgreSQL query for the following user query written in English:
            {user_query}

            ### Instructions:
            1. Ensure the query strictly adheres to the schema provided in the explanation.
            2. Use proper SQL syntax for PostgreSQL.
            3. Include all necessary JOINs, WHERE clauses, and GROUP BY as needed.
            4. **Do not output any text other than the SQL query.**
            5. Only output a valid PostgreSQL query and nothing else!
            6. Do not include any markdown formatting or backticks.
            7. Do not include any comments or explanations.

            ### Example:
            User query: Find the names of all customers whose order total amount is greater than 30.

            Output:
            SELECT c.name 
            FROM customers c
            JOIN orders o ON c.id = o.customer_id
            WHERE o.total_amount > 30;

            Query: all those customers name whose have not placed any order
            SELECT c.name
            FROM customers c
            LEFT JOIN orders o ON c.id = o.customer_id
            WHERE o.id IS NULL;

            Now process the following:
            """)
    ])
    
    # Combine prompt and LLM into a sequence
    chain = prompt | llm

    # Use `invoke` instead of `run`
    response = chain.invoke({"explanation": explanation, "user_query": user_query})
    print("response of query generation:", response)
    return response

def execute_sql_query(db_name: str, query: Union[str, object]):
    """
    Execute a PostgreSQL query generated by an LLM or passed directly as a string.

    Args:
        db_name (str): Name of the PostgreSQL database.
        query (Union[str, object]): Query string or object containing SQL syntax.

    Returns:
        list or int: Query result.

    Raises:
        ValueError: If the query format is invalid or execution fails.
    """
    try:
        # Extract query content if needed
        if hasattr(query, "content"):
            query = query.content

        print("Query received:", query)

        if not isinstance(query, str):
            raise ValueError("Query must be a string.")

        # Basic validation to prevent obvious SQL injection
        forbidden_keywords = ['DROP', 'TRUNCATE', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'GRANT']
        if any(keyword in query.upper() for keyword in forbidden_keywords):
            raise ValueError("Query contains forbidden operations.")

        # Connect to PostgreSQL
        conn = psycopg.connect(
            dbname=db_name,
            user="postgres",
            password="postgres@11",
            host="localhost",
            port="5432"
        )
        
        with conn.cursor() as cur:
            try:
                cur.execute(query)
                
                # For SELECT queries, fetch results
                if query.strip().upper().startswith('SELECT'):
                    result = cur.fetchall()
                    # Get column names
                    colnames = [desc[0] for desc in cur.description]
                    result = [dict(zip(colnames, row)) for row in result]
                else:
                    # For other queries, return rowcount
                    result = cur.rowcount
                    conn.commit()
                
                print("Query Execution Result:", result)
                return result
                
            except Exception as e:
                conn.rollback()
                raise ValueError(f"Error executing query: {str(e)}")
            finally:
                conn.close()

    except psycopg.Error as e:
        raise ValueError(f"PostgreSQL error: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error: {str(e)}")

