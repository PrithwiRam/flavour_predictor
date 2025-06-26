# salad_predictor_app/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .utils.food_ai import ProfessionalSaladPredictor
import json

# Initialize predictor globally (consider moving to a singleton or app config later)
predictor = ProfessionalSaladPredictor()

def index(request):
    """Render the main salad predictor page."""
    return render(request, 'salad_predictor_app/index.html')

@csrf_exempt  # Temporary for testing; use proper CSRF in production
def predict_salad_taste(request):
    """Handle POST requests to predict salad taste."""
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            ingredient_list = data.get("ingredients", [])
            if not ingredient_list:
                return JsonResponse({"error": "No ingredients provided"}, status=400)
            
            result = predictor.predict_full(ingredient_list)
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid request method"}, status=405)