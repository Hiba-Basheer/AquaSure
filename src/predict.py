from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model/xgboost_model.pkl")


# FRONTEND
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AquaSure | Water Potability Analyzer</title>
        <style>
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                background: linear-gradient(to right, #74ebd5, #ACB6E5);
                margin: 0;
                padding: 0;
            }
            .container {
                max-width: 500px;
                margin: 50px auto;
                background: white;
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.15);
            }
            h1 {
                text-align: center;
                color: #333;
            }
            p {
                text-align: center;
                color: #666;
                font-size: 14px;
            }
            table {
                width: 100%;
                margin-top: 15px;
            }
            td {
                padding: 6px;
            }
            input {
                width: 100%;
                padding: 7px;
                border-radius: 4px;
                border: 1px solid #ccc;
            }
            button {
                width: 100%;
                margin-top: 15px;
                padding: 10px;
                border: none;
                border-radius: 5px;
                background: #007bff;
                color: white;
                font-size: 16px;
                cursor: pointer;
            }
            button:hover {
                background: #0056b3;
            }
            footer {
                text-align: center;
                margin-top: 15px;
                font-size: 12px;
                color: #888;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AquaSure</h1>
            <p>AI-Powered Water Potability Analyzer</p>

            <form action="/predict" method="post">
                <table>
                    <tr><td>pH</td><td><input name="ph" required></td></tr>
                    <tr><td>Hardness</td><td><input name="hardness" required></td></tr>
                    <tr><td>Solids</td><td><input name="solids" required></td></tr>
                    <tr><td>Chloramines</td><td>
                        <input name="chloramines" required></td></tr>
                    <tr><td>Sulfate</td><td><input name="sulfate" required></td></tr>
                    <tr><td>Conductivity</td><td>
                        <input name="conductivity" required></td></tr>
                    <tr><td>Organic Carbon</td><td>
                        <input name="organic_carbon" required></td></tr>
                    <tr><td>Trihalomethanes</td><td>
                        <input name="trihalomethanes" required></td></tr>
                    <tr><td>Turbidity</td><td><input name="turbidity" required></td></tr>
                </table>

                <button type="submit">Analyze Water</button>
            </form>

            <footer>
                Powered by XGBoost • FastAPI • MLflow
            </footer>
        </div>
    </body>
    </html>
    """


# BACKEND
@app.post("/predict", response_class=HTMLResponse)
def predict(
    ph: float = Form(...),
    hardness: float = Form(...),
    solids: float = Form(...),
    chloramines: float = Form(...),
    sulfate: float = Form(...),
    conductivity: float = Form(...),
    organic_carbon: float = Form(...),
    trihalomethanes: float = Form(...),
    turbidity: float = Form(...),
):
    features = np.array(
        [[
            ph,
            hardness,
            solids,
            chloramines,
            sulfate,
            conductivity,
            organic_carbon,
            trihalomethanes,
            turbidity,
        ]]
    )

    prediction = model.predict(features)[0]

    result = (
        "Water is POTABLE"
        if prediction == 1
        else "Water is NOT POTABLE"
    )

    return f"""
    <html>
        <body style="
            font-family: Segoe UI, Arial;
            background: linear-gradient(to right, #74ebd5, #ACB6E5);
            text-align: center;
            padding-top: 60px;
        ">
            <div style="
                background: white;
                display: inline-block;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.15);
            ">
                <h2>{result}</h2>
                <br>
                <a href="/" style="text-decoration: none; color: #007bff;">
                    ← Analyze another sample
                </a>
            </div>
        </body>
    </html>
    """