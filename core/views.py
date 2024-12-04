from django.shortcuts import render
from bson.objectid import ObjectId 
# Create your views here.
from django.http import JsonResponse
from .utils import extract_mongo_schema, generate_explanations_with_llama, save_explanations_to_mongodb , get_database_explanation, generate_query, execute_query,refine_output
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json

def list_databases(request):
    # Logic to list user databases
    return JsonResponse({'databases': []})

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
    
    # Validate inputs
    if not db_uri or not db_name:
        return JsonResponse({"error": "db_uri and db_name are required"}, status=400)
    
    # Ensure db_name is a string
    db_name = str(db_name)

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


def fetch_explanations(request, db_name):
    user_id = request.user.id  # Assuming user authentication is set up

    explanations = list(EXPLANATIONS_COLLECTION.find(
        {"user_id": user_id, "db_name": db_name},
        {"_id": 1, "schema": 1, "explanation": 1, "is_finalized": 1}  # Return these fields
    ))
    
    # Convert ObjectId to string for JSON serialization
    for explanation in explanations:
        explanation["_id"] = str(explanation["_id"])

    return JsonResponse({"explanations": explanations})

   
def update_explanation(request, explanation_id):
    if request.method == "POST":
        new_explanation = request.POST.get('explanation')

        EXPLANATIONS_COLLECTION.update_one(
            {"_id": ObjectId(explanation_id)},  # Convert to ObjectId
            {"$set": {"explanation": new_explanation}}
        )

        return JsonResponse({"message": "Explanation updated successfully!"})

    return JsonResponse({"error": "Invalid request method."}, status=400)

