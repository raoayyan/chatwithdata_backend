from django.shortcuts import render
from bson.objectid import ObjectId 
# Create your views here.
from django.http import JsonResponse
# from .utils import extract_mongo_schema, generate_explanations_with_llama, save_explanations_to_mongodb , get_database_explanation, generate_query, execute_query,refine_output, all_databases, check_database_exists,save_sql_explanations_to_mongodb ,store_chat, signup, sign_in
from .utils import (
    extract_sql_schema,
    generate_sql_query,
    execute_sql_query,
    EXPLANATIONS_COLLECTION,
    extract_mongo_schema,
    generate_explanations_with_llama,
    save_explanations_to_mongodb,
    get_database_explanation,
    generate_query,
    execute_query,
    refine_output,
    all_databases,
    check_database_exists,
    store_chat,
    signup,
    save_sql_explanations_to_mongodb,
    sign_in,
)

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json
import bcrypt
import jwt
from datetime import datetime, timedelta


SECRET_KEY = "432874u5872"



@csrf_exempt
def list_databases(request):
    try:
        db_list = all_databases()
        return JsonResponse({'databases': db_list})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
@csrf_exempt
def add_database(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Only POST method is allowed"}, status=405)

    try:
        data = json.loads(request.body)
        db_type = data.get('db_type')
        db_name = data.get('db_name')

        if db_type == 'mongodb':
            db_uri = data.get('db_uri')
            if check_database_exists(db_name):
                return JsonResponse({"error": f"Database '{db_name}' already exists."}, status=400)
            schema_data = extract_mongo_schema(db_uri, db_name)
        elif db_type == 'postgresql':
            if check_database_exists(db_name):
                return JsonResponse({"error": f"Database '{db_name}' already exists."}, status=400)
            schema_data = extract_sql_schema()
        else:
            return JsonResponse({"error": "Unsupported database type"}, status=400)

        explanations = generate_explanations_with_llama(schema_data)
        user_id = request.user.id

        if db_type == 'postgresql':
            save_sql_explanations_to_mongodb(user_id, db_name, explanations)
        else:
            save_explanations_to_mongodb(user_id, db_name, explanations)

        return JsonResponse({"message": "Explanations generated and saved!"})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# @csrf_exempt
# def add_database(request):
#     # print('hitting')
    
#     # Check if it's a POST request
#     if request.method != 'POST':
#         return JsonResponse({"error": "Only POST method is allowed"}, status=405)
    
#     # For JSON data
    
#     try:
#         data = json.loads(request.body)
#         db_uri = data.get('db_uri')
#         db_name = data.get('db_name')
#     except json.JSONDecodeError:
#         # Fallback to form data if JSON parsing fails
#         db_uri = request.POST.get('db_uri')
#         db_name = request.POST.get('db_name')

#     print(db_name)
#     # Validate inputs
#     if not db_uri or not db_name:
#         return JsonResponse({"error": "db_uri and db_name are required"}, status=400)
    
#     # Ensure db_name is a string
#     db_name = str(db_name)
#     if check_database_exists(db_name):
#         return JsonResponse({"error": f"Database '{db_name}' already exists."}, status=400)

#     try:
#         user_id = request.user.id
        
#         schema_data = extract_mongo_schema(db_uri, db_name)
        
     
#         explanations = generate_explanations_with_llama(schema_data)
        

#         #Save explanations to MongoDB
#         save_explanations_to_mongodb(user_id, db_name, explanations)

#         return JsonResponse({"message": "Explanations generated and saved!"})
    
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return JsonResponse({"error": str(e)}, status=500)
@csrf_exempt
def chat_with_database(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            database_name = data.get('database_name')
            user_query = data.get('user_query')
            db_type = data.get('db_type')

            explanation = get_database_explanation(database_name)

            if db_type == 'mongodb':
                generated_query = generate_query(user_query, explanation)
                query_result = execute_query(database_name, generated_query)
            elif db_type == 'postgresql':
                generated_query = generate_sql_query(user_query, explanation)
                query_result = execute_sql_query(generated_query)
            else:
                return JsonResponse({"error": "Unsupported database type"}, status=400)

            refined_response = refine_output(user_query, query_result)
            return JsonResponse({"success": True, "response": refined_response})
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})
    else:
        return JsonResponse({"success": False, "error": "Invalid request method"})

# @csrf_exempt
# def chat_with_database(request):
#     if request.method == "POST":
#         try:
#             data = json.loads(request.body)
#             database_name = data.get('database_name')
#             user_query = data.get('user_query')
#         except json.JSONDecodeError:
#             # Fallback to form data if JSON parsing fails
#             database_name = request.POST.get('database_name')
#             user_query = request.POST.get('user_query')

#         try:
#             print("db Name :", database_name)
#             print("user_query :", user_query)

#             # Step 1: Get database explanation
#             explanation = get_database_explanation(database_name)
#             print("This is explanation from mongod db: ", explanation)
#             # Step 2: Generate query
#             generated_query = generate_query(user_query, explanation)
#             print("generated Query is :",generated_query)
#             # # Step 3: Execute query
#             query_result = execute_query(database_name, generated_query)
#             print("resulted output after execution of query:",query_result)
#             # # Step 4: Refine output
            
#             refined_output = refine_output(user_query, query_result)
#             print("Clean response:",refined_output)
#             return JsonResponse({"success": True, "response": refined_output})
#         except Exception as e:
#             return JsonResponse({"success": False, "error": str(e)})
#     else:
#         return JsonResponse({"success": False, "error": "Invalid request method"})

@csrf_exempt
def fetch_explanations(request):
    """
    Fetch explanations for a given database name.

    Expects:
        - database_name (str): The name of the database to fetch explanations for.

    Returns:
        JsonResponse: The explanations for the given database or an error message.
    """
    if request.method == "POST":
        try:
            # Parse JSON data from the request body
            data = json.loads(request.body)
            database_name = data.get('database_name')
            print("This is database name from request body: ", database_name)
            # Validate input
            if not database_name:
                return JsonResponse({"error": "database_name is required"}, status=400)

            # Fetch explanations using the provided database name
            explanation = get_database_explanation(database_name)
            if "_id" in explanation and isinstance(explanation["_id"], ObjectId):
                explanation["_id"] = str(explanation["_id"])

            return JsonResponse({"explanations": explanation})
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON data"}, status=400)
        except ValueError as e:
            return JsonResponse({"error": str(e)}, status=404)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method. Only POST is allowed."}, status=405)

@csrf_exempt  
def update_explanation(request, explanation_id):
    if request.method == "POST":
        new_explanation = request.POST.get('explanation')

        EXPLANATIONS_COLLECTION.update_one(
            {"_id": ObjectId(explanation_id)},  # Convert to ObjectId
            {"$set": {"explanation": new_explanation}}
        )

        return JsonResponse({"message": "Explanation updated successfully!"})

    return JsonResponse({"error": "Invalid request method."}, status=400)

@csrf_exempt
def store_chat_view(request):
    """
    View to store chat data into the 'chats' collection in MongoDB.

    Expects:
        - chat_id (str): Unique identifier for the chat session.
        - query (str, optional): The user's query.
        - response (str, optional): The LLM's response.

    Returns:
        JsonResponse: Success or error message.
    """
    if request.method == "POST":
        try:
            # Parse JSON data from the request
            data = json.loads(request.body)
            chat_id = data.get('chat_id')
            query = data.get('query')
            response = data.get('response')
            print("chat_id:", chat_id)
            print("query:", query)
            print("response:", response)
            # Validate inputs
            if not chat_id:
                return JsonResponse({"error": "chat_id is required"}, status=400)

            # Call the store_chat function from utils.py
            result = store_chat(chat_id=chat_id, query=query, response=response)

            return JsonResponse(result)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON data"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method. Only POST is allowed."}, status=405)

@csrf_exempt
def signup_view(request):
    """
    Handles user registration/signup.
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            name = data.get('name')
            email = data.get('email')
            password = data.get('password')

            # Validate inputs
            if not all([name, email, password]):
                return JsonResponse({"error": "Name, email and password are required"}, status=400)

            # Call the signup function
            result = signup(name=name, email=email, password=password)

            if "error" in result:
                return JsonResponse(result, status=400)
            
            return JsonResponse(result, status=201)  # 201 Created

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON data"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Only POST method is allowed"}, status=405)

@csrf_exempt
def sign_in_view(request):
    """
    Handles user authentication/sign-in.
    Returns a JWT token if successful.
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            email = data.get('email')
            password = data.get('password')

            # Validate inputs
            if not all([email, password]):
                return JsonResponse({"error": "Email and password are required"}, status=400)

            # Call the sign_in function
            result = sign_in(email=email, password=password)

            if "error" in result:
                return JsonResponse(result, status=401)  # 401 Unauthorized
            
            return JsonResponse(result)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON data"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Only POST method is allowed"}, status=405)