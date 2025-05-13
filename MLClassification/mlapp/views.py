from django.http import HttpResponse,JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json
import pandas as pd
from mlapp.mlmodels.randomforest.model.RF_prediction import predict_and_annotate_df

def home(request):
    return render(request, 'mlapp/index.html')  # Make sure this template exists


@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            model_choice = body.get('model')
            records = body.get('records')

            if not model_choice or not records:
                return JsonResponse({'status': 'error', 'message': 'Missing model or records'}, status=400)

            df = pd.DataFrame(records)

            # Handle based on model choice
            if model_choice == 'random_forest':
                model_path = 'mlapp/mlmodels/randomforest/data/RF_model.joblib'
                result_df,avg_confidence = predict_and_annotate_df(model_path, df)

            else:
                return JsonResponse({'status': 'error', 'message': f'Model {model_choice} is not implemented yet'}, status=400)

            return JsonResponse({
                'status': 'success',
                'data': result_df.to_dict(orient='records'),
                'confidence':avg_confidence
            })

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)
