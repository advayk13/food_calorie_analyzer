# type: ignore
from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import os
import json
import uuid
import cv2  # OpenCV for cropping

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load trained model
model = tf.keras.models.load_model('best_cnn_model.h5')

# Load nutrition data
with open('nutrition_data.json') as f:
    nutrition_data = json.load(f)

# Classes (make sure order matches your model)
class_names = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food',
               'Meat', 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable-Fruit']

# 
# Function to crop center and resize
# 
def prepare_image(img_path):
    # Read image using OpenCV
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    min_dim = min(h, w)
    
    # Crop center square
    start_h = (h - min_dim) // 2
    start_w = (w - min_dim) // 2
    img_cropped = img[start_h:start_h+min_dim, start_w:start_w+min_dim]
    
    # Resize to 224x224
    img_resized = cv2.resize(img_cropped, (224, 224))
    
    # Convert BGR to RGB
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize and expand dims
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------
# Function for contextual recommendation
# -------------------------------
def get_recommendation(pred_class, nutrition, weight=None):
    """
    Returns a contextual recommendation based on nutrition and weight
    """
    calories = nutrition.get('calories', 0)
    protein = nutrition.get('protein', 0)
    fats = nutrition.get('fats', 0)
    carbs = nutrition.get('carbs', 0)

    # Scale calories and macros if weight is given
    if weight:
        factor = weight / 100
        calories = calories * factor
        protein = protein * factor
        fats = fats * factor
        carbs = carbs * factor

    # Define generally healthy foods
    healthy_foods = ['Meat', 'Seafood', 'Vegetable-Fruit', 'Rice', 'Egg', 'Noodles-Pasta', 'Dairy product', 'Bread']
    indulgent_foods = ['Dessert', 'Fried food', 'Soup']

    # Contextual recommendation
    if pred_class in healthy_foods:
        if calories > 300:
            return "Great food, but eat in a planned manner."
        else:
            return "Healthy choice!"
    elif pred_class in indulgent_foods:
        if calories > 200:
            return "Delicious but eat in moderation."
        else:
            return "Enjoy, but in moderation."
    else:
        return "Healthy choice!"

# -------------------------------
# Routes
# -------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    nutrition = None
    top_predictions = None
    filename = None
    weight = None
    recommendation = None

    if request.method == 'POST':
        # Check if weight form submitted
        if 'weight' in request.form and 'current_prediction' in request.form:
            # User entered weight, scale nutrition
            weight = float(request.form.get('weight', 100))
            pred_class = request.form.get('current_prediction')
            nutrition_base = nutrition_data.get(pred_class, {})
            # Scale nutrition
            factor = weight / 100
            nutrition = {
                'calories': round(nutrition_base.get('calories', 0) * factor, 2),
                'protein': round(nutrition_base.get('protein', 0) * factor, 2),
                'fats': round(nutrition_base.get('fats', 0) * factor, 2),
                'carbs': round(nutrition_base.get('carbs', 0) * factor, 2)
            }
            recommendation = get_recommendation(pred_class, nutrition, weight)
            filename = request.form.get('current_file')
            prediction = pred_class
            top_predictions = json.loads(request.form.get('top_predictions', '[]'))

        else:
            # Image upload form submitted
            if 'food_image' not in request.files:
                return redirect(request.url)
            file = request.files['food_image']
            if file.filename == '':
                return redirect(request.url)
            if file:
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                unique_name = str(uuid.uuid4()) + "_" + file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
                file.save(filepath)

                # Prepare image and predict
                img = prepare_image(filepath)
                preds = model.predict(img)

                # Get top 3 predictions
                top_indices = preds[0].argsort()[-3:][::-1]
                top_predictions = [(class_names[i], float(preds[0][i])) for i in top_indices]

                # Extract top class and confidence
                pred_class = top_predictions[0][0]
                confidence = top_predictions[0][1]

                # --- NEW: Non-food detection based on confidence ---
                CONFIDENCE_THRESHOLD = 0.5  # Adjust based on your model testing
                if confidence < CONFIDENCE_THRESHOLD:
                    prediction = None
                    filename = unique_name
                    recommendation = "⚠️ This doesn’t seem to be a food item. Please upload a food image."
                    nutrition = None
                else:
                    # Normal case - confident food prediction
                    nutrition_base = nutrition_data.get(pred_class, {})
                    nutrition = nutrition_base
                    recommendation = get_recommendation(pred_class, nutrition)
                    prediction = pred_class
                    filename = unique_name

    return render_template('index.html',
                           filename=filename,
                           prediction=prediction,
                           nutrition=nutrition,
                           recommendation=recommendation,
                           top_predictions=top_predictions,
                           weight=weight)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
