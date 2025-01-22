from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

app = Flask(__name__)

# Global variables
uploaded_data = None
model = None
scaler = StandardScaler()

@app.route('/')
def home():
    return '''
    <html>
        <head><title>Flask API</title></head>
        <body>
            <h1>Welcome to the Flask API</h1>
            <p>Use the following endpoints:</p>
            <ul>
                <li><strong>/upload</strong>: Upload a CSV file</li>
                <li><strong>/train</strong>: Train a model</li>
                <li><strong>/predict</strong>: Make predictions</li>
            </ul>
        </body>
    </html>
    '''


@app.route('/upload', methods=['POST'])
def upload_data():
    global uploaded_data
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_data = pd.read_csv(file)
    return jsonify({"message": "File uploaded successfully", "columns": list(uploaded_data.columns)})

@app.route('/train', methods=['POST'])
def train_model():
    global uploaded_data, model, scaler
    if uploaded_data is None:
        return jsonify({"error": "No data uploaded. Please upload data first."}), 400

    X = uploaded_data.drop(columns=["DefectStatus"])
    y = uploaded_data["DefectStatus"]
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    joblib.dump(model, 'model.pkl')

    return jsonify({"message": "Model trained successfully", "accuracy": accuracy, "f1_score": f1})

@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler
    if model is None:
        return jsonify({"error": "Model not trained. Please train the model first."}), 400

    try:
        input_data = request.json
        input_df = pd.DataFrame([input_data]) 
        input_scaled = scaler.transform(input_df) 
        
        # Make predictions
        prediction = int(model.predict(input_scaled)[0]) 
        confidence = float(max(model.predict_proba(input_scaled)[0])) 

        # Return JSON response
        return jsonify({"DefectStatus": prediction, "Confidence": round(confidence, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('127.0.0.1', 5000, app, use_reloader=True, threaded=True)
