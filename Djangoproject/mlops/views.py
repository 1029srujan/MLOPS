from django.shortcuts import render
from joblib import load

# Load the sentiment analysis model
model = load('./ml/sentiment_analysis_model.joblib')

# View for rendering the input form
def predictor(request):
    return render(request, 'main.html')

# View for processing the form and sending the result
def formInfo(request):
    # Get the text input from the form
    text = request.GET['text']
    
    # Predict the emotion using the model
    y_pred = model.predict([text])
    
    # Extract the predicted emotion (ensure it's a string and not an array)
    emotion = y_pred[0] if y_pred else "Unknown"
    
    # Pass the emotion result to result.html
    return render(request, 'result.html', {'result': emotion})
