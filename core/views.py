from django.shortcuts import render
from bson.objectid import ObjectId 
# Create your views here.
from django.http import JsonResponse
from .utils import extract_mongo_schema, generate_explanations_with_llama, save_explanations_to_mongodb , get_database_explanation, generate_query, execute_query,refine_output, all_databases, check_database_exists, store_chat
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json

@csrf_exempt
def list_databases(request):
    try:
        db_list = all_databases()
        return JsonResponse({'databases': db_list})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def add_database(request):
    # print('hitting')
    
    # Check if it's a POST request
    if request.method != 'POST':
        return JsonResponse({"error": "Only POST method is allowed"}, status=405)
    
    # For JSON data
    
    try:
        data = json.loads(request.body)
        db_uri = data.get('db_uri')
        db_name = data.get('db_name')
    except json.JSONDecodeError:
        # Fallback to form data if JSON parsing fails
        db_uri = request.POST.get('db_uri')
        db_name = request.POST.get('db_name')

    print(db_name)
    # Validate inputs
    if not db_uri or not db_name:
        return JsonResponse({"error": "db_uri and db_name are required"}, status=400)
    
    # Ensure db_name is a string
    db_name = str(db_name)
    if check_database_exists(db_name):
        return JsonResponse({"error": f"Database '{db_name}' already exists."}, status=400)

    try:
        user_id = request.user.id
        
        schema_data = extract_mongo_schema(db_uri, db_name)
        
     
        explanations = generate_explanations_with_llama(schema_data)
        

        #Save explanations to MongoDB
        save_explanations_to_mongodb(user_id, db_name, explanations)

        return JsonResponse({"message": "Explanations generated and saved!"})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def chat_with_database(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            database_name = data.get('database_name')
            user_query = data.get('user_query')
        except json.JSONDecodeError:
            # Fallback to form data if JSON parsing fails
            database_name = request.POST.get('database_name')
            user_query = request.POST.get('user_query')

        try:
            print("db Name :", database_name)
            print("user_query :", user_query)

            # Step 1: Get database explanation
            explanation = get_database_explanation(database_name)
            print("This is explanation from mongod db: ", explanation)
            # Step 2: Generate query
            generated_query = generate_query(user_query, explanation)
            print("generated Query is :",generated_query)
            # # Step 3: Execute query
            query_result = execute_query(database_name, generated_query)
            print("resulted output after execution of query:",query_result)
            # # Step 4: Refine output
            
            refined_output = refine_output(user_query, query_result)
            print("Clean response:",refined_output)
            return JsonResponse({"success": True, "response": refined_output})
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})
    else:
        return JsonResponse({"success": False, "error": "Invalid request method"})

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

    