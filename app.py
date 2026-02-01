import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tf_keras.models import load_model
from tf_keras.preprocessing import image
from PIL import Image

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'brain_tumor_model.h5')
# IMPORTANT: Update these class names to match your training data's class_indices exactly!
# Typically found in: train_data.class_indices
# Mapped to friendly names: ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# Health and Dietary Suggestions
HEALTH_TIPS = {
    'Glioma Tumor': "Suggested Diet: Foods rich in antioxidants like berries, leafy greens, and cruciferous vegetables. \nLifestyle: Regular low-impact exercise and stress management techniques like meditation.",
    'Meningioma Tumor': "Suggested Diet: Maintain a balanced diet low in processed sugars. Include omega-3 fatty acids (fish, flaxseeds). \nLifestyle: Ensure adequate sleep and hydration.",
    'Pituitary Tumor': "Suggested Diet: Foods high in fiber and calcium. Avoid excessive caffeine and alcohol. \nLifestyle: Regular monitoring of hormone levels and vision check-ups.",
    'No Tumor': "Great news! To maintain brain health:\nDiet: Walnuts, blueberries, and fatty fish.\nLifestyle: Mental exercises (puzzles, reading) and regular physical activity."
}

# Global model variable
model = None

def load_brain_tumor_model():
    global model
    print(f"📂 Current Directory: {os.getcwd()}")
    print(f"📄 Files in root: {os.listdir(os.path.dirname(__file__))}")
    try:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            print(f"✅ Model loaded from {MODEL_PATH}")
        else:
            print(f"❌ Model file not found at {MODEL_PATH}. Prediction will fail.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

load_brain_tumor_model()

def prepare_image(img_path):
    # Load image with target size 224x224 as per training script (MobileNetV2)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale 1./255
    return img_array

def predict_image_file(filepath):
    """Shared logic to predict class for an image file."""
    if model is None:
        raise Exception('Model not loaded.')
        
    # Preprocess and Predict
    processed_image = prepare_image(filepath)
    predictions = model.predict(processed_image)
    
    # Get result
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))
    predicted_label = CLASS_NAMES[predicted_class_index]
    
    return predicted_label, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please ensure brain_tumor_model.h5 is in the directory.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Save file temporarily
            filepath = os.path.join('static', 'temp_upload.png')
            file.save(filepath)

            # Use shared prediction logic
            predicted_label, confidence = predict_image_file(filepath)

            return jsonify({
                'class': predicted_label,
                'confidence': f"{confidence * 100:.2f}%",
                'image_path': filepath
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

def chatbot_response(message):
    message = message.lower()
    
    if "hello" in message or "hi" in message:
        return "Hello! I am the NeuroScan AI assistant. You can ask me questions or upload an MRI scan for analysis."
    
    elif "tumor" in message:
        return "I can help you classify brain tumors from MRI scans. Just upload an image using the camera icon!"
    
    elif "accuracy" in message:
        return "Our model uses Transfer Learning (MobileNetV2) and typically achieves high accuracy. Upload an image to see the confidence score."
        
    elif "glioma" in message:
        return "Glioma is a type of tumor that occurs in the brain and spinal cord. " + HEALTH_TIPS['Glioma Tumor']
        
    elif "meningioma" in message:
        return "Meningioma is a tumor that arises from the meninges. " + HEALTH_TIPS['Meningioma Tumor']
        
    elif "pituitary" in message:
        return "Pituitary tumors are abnormal growths that develop in your pituitary gland. " + HEALTH_TIPS['Pituitary Tumor']
    
    elif "help" in message:
        return "You can upload an MRI image for classification and health tips, or ask about tumor types."
        
    else:
        return "I'm sorry, I didn't quite understand. Try asking about 'tumors', or upload an MRI image."

@app.route('/chat', methods=['POST'])
def chat():
    # Handle optional image upload
    file = request.files.get('file')
    message = request.form.get('message', '')
    
    response_text = ""
    
    try:
        if file and file.filename != '':
            # Save properly to avoid overwriting main upload if possible, or use unique name
            # For simplicity using a chat-specific temp file
            filepath = os.path.join('static', 'temp_chat_upload.png')
            file.save(filepath)
            
            label, confidence = predict_image_file(filepath)
            conf_str = f"{confidence * 100:.1f}%"
            
            tips = HEALTH_TIPS.get(label, "")
            
            response_text = f"I analyzed the image. Prediction: **{label}** ({conf_str}).\n\n{tips}"
            
            # If user added a message with the image, answer that too or just append
            if message:
                response_text = f"{chatbot_response(message)}\n\n" + response_text
                
        elif message:
            response_text = chatbot_response(message)
        else:
            response_text = "Please send a message or image."
            
        return jsonify({'response': response_text})
        
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({'response': "Sorry, I encountered an error processing your request."})

if __name__ == '__main__':
    app.run(debug=True)
