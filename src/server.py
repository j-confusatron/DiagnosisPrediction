from flask import Flask, request, jsonify
from flask_cors import CORS

from dual_model_pipeline.diagnosis_predicter import DiagnosisPredicter

app = Flask(__name__)
CORS(app)

predicter = DiagnosisPredicter()

@app.route('/')
def index():
    return "Hello world."

@app.route('/infer', methods=['POST'])
def inference():
    data = request.get_json()
    print("\nInference")
    print("Request")
    print("---------------------")
    print(data)
    results = predicter.inference(data)
    print("Response")
    print("---------------------")
    print(results)
    results = jsonify(results)
    return results

app.run()