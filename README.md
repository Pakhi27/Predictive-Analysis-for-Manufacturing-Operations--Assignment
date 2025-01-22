## Predictive Analysis for Manufacturing Operations

## Objective
This project aims to develop a RESTful API to predict machine downtime or production defects using a manufacturing dataset. The system supports operations such as data upload, model training, and making predictions.

---

## Features
1. Upload Endpoint: Upload a CSV file containing manufacturing data.
2. Train Endpoint: Train a machine learning model on the uploaded dataset and return performance metrics.
3. Predict Endpoint: Accept JSON input with manufacturing parameters and return predictions with confidence scores.

---

## Dataset
The dataset used in this project contains the following key columns:
- ProductionVolume
- ProductionCost
- SupplierQuality
- DeliveryDelay
- DefectRate
- QualityScore
- MaintenanceHours
- DefectStatus(Target Variable)

Synthetic data or publicly available datasets from Kaggle or UCI ML Repository can also be used if required.

---

## Machine Learning Models
The following models are implemented:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Extreme Gradient Boosting (XGBoost)

Best Accuracy Achieved: 95.06% using XGBoost.

---

## Technical Stack
- Programming Language: Python
- Web Framework: Flask
- Machine Learning Libraries: scikit-learn, XGBoost
- Others: pandas, numpy, matplotlib, seaborn

---

## Endpoints
1. Upload Endpoint (`POST /upload`):
   - Accepts a CSV file containing manufacturing data.
   - Example Request:
     ```
     curl -X POST -F "file=@manufacturing_data.csv" http://localhost:5000/upload
     ```
   - Example Response:
     ```json
     { "message": "File uploaded successfully." }
     ```

2. Train Endpoint (`POST /train`):
   - Trains the model on the uploaded dataset.
   - Returns performance metrics such as accuracy and F1-score.
   - Example Request:
     ```
     curl -X POST http://localhost:5000/train
     ```
   - Example Response:
     ```json
     {
         "accuracy": 0.9506,
         "f1_score": 0.9475
     }
     ```

3. Predict Endpoint (`POST /predict`):
   - Accepts JSON input for prediction.
   - Example Request:
     ```
     curl -X POST -H "Content-Type: application/json" -d '{"Temperature": 80, "Run_Time": 120}' http://localhost:5000/predict
     ```
   - Example Response:
     ```json
     {
         "Downtime": "Yes",
         "Confidence": 0.85
     }
     ```

---

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```
   cd predictive_analysis
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the Flask application:
   ```
   python app.py
   ```

5. Test the API locally using Postman or cURL.

---

## Folder Structure
```
.
├── app.py
├── requirements.txt
├── dataset
│   └── manufacturing_data.csv
├── models
│   └── model.pkl
├── static
│   └── plots
└── README.md
```

---

## Performance Evaluation
The models were evaluated using the following metrics:
- Accuracy
- F1-Score
- Confusion Matrix

Best performance was achieved with XGBoost.

---

## Visualization
The project includes visualizations for:
- Correlation heatmaps
- Boxplots comparing features against the target variable
- Scatterplots to explore relationships
- Pair plots for feature analysis

---

## Contributors
Developed by: **Pakhi Singhal**

