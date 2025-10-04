from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained ML model
model = pickle.load(open("models/crop_model.pkl", "rb"))

# Crop data with ideal ranges and extra info
crop_info = {
    "rice": {"N": (90, 120), "P": (40, 60), "K": (40, 60), "ph": (5.5, 7), "temp": (20, 35), "humidity": (60, 90),
             "note":"High water requirement, staple food", "price":(2000,2500), "season":"Jun-Sep", "water":(1200,1500), "rotation":"Wheat"},
    "maize": {"N": (100, 140), "P": (40, 60), "K": (40, 60), "ph": (5.5, 7), "temp": (18, 32), "humidity": (50, 80),
              "note":"Versatile crop", "price":(1500,2000), "season":"Jul-Oct", "water":(500,800), "rotation":"Soybean"},
    "chickpea": {"N": (20, 50), "P": (20, 40), "K": (20, 40), "ph": (6, 8), "temp": (15, 30), "humidity": (30, 60),
                 "note":"Drought resistant, protein-rich", "price":(3500,4000), "season":"Oct-Mar", "water":(300,500), "rotation":"Lentil"},
    "kidneybeans": {"N": (20, 60), "P": (20, 50), "K": (20, 50), "ph": (6, 7), "temp": (15, 30), "humidity": (50, 70),
                    "note":"Protein-rich, moderate water", "price":(4000,5000), "season":"Jul-Oct", "water":(400,700), "rotation":"Maize"},
    "pigeonpeas": {"N": (15, 50), "P": (10, 40), "K": (10, 40), "ph": (5.5, 7), "temp": (20, 30), "humidity": (50, 70),
                   "note":"Resistant to drought", "price":(4000,4500), "season":"Jun-Nov", "water":(400,600), "rotation":"Mothbeans"},
    "mothbeans": {"N": (10, 30), "P": (10, 30), "K": (10, 30), "ph": (6, 7.5), "temp": (25, 35), "humidity": (30, 60),
                  "note":"Low water & fertilizer need", "price":(4500,5000), "season":"Jul-Oct", "water":(300,500), "rotation":"Pigeonpeas"},
    "mungbean": {"N": (20, 40), "P": (10, 30), "K": (10, 30), "ph": (6, 7), "temp": (25, 35), "humidity": (40, 60),
                 "note":"Short-duration, protein-rich", "price":(3000,4000), "season":"Jun-Sep", "water":(300,500), "rotation":"Maize"},
    "blackgram": {"N": (20, 50), "P": (10, 30), "K": (10, 30), "ph": (6, 7), "temp": (25, 35), "humidity": (30, 60),
                  "note":"Legume, nitrogen fixer", "price":(3000,4000), "season":"Jun-Sep", "water":(300,500), "rotation":"Pigeonpeas"},
    "lentil": {"N": (20, 50), "P": (10, 40), "K": (10, 40), "ph": (6, 8), "temp": (15, 30), "humidity": (30, 60),
               "note":"Protein-rich, drought resistant", "price":(3500,4000), "season":"Oct-Mar", "water":(300,500), "rotation":"Chickpea"},
    "pomegranate": {"N": (20, 60), "P": (20, 50), "K": (30, 70), "ph": (5.5, 7), "temp": (20, 35), "humidity": (40, 70),
                    "note":"Fruit crop, high market", "price":(6000,7000), "season":"Mar-Jul", "water":(500,800), "rotation":"Mango"},
    "banana": {"N": (150, 200), "P": (50, 100), "K": (200, 300), "ph": (5.5, 7), "temp": (25, 35), "humidity": (70, 90),
               "note":"High water requirement", "price":(3000,4000), "season":"Year-round", "water":(1000,1500), "rotation":"Papaya"},
    "mango": {"N": (80, 150), "P": (40, 80), "K": (40, 80), "ph": (5.5, 7.5), "temp": (24, 35), "humidity": (50, 85),
              "note":"Long-term crop, high market demand", "price":(5000,6000), "season":"Mar-Jun", "water":(800,1200), "rotation":"Banana"},
    "grapes": {"N": (80, 150), "P": (40, 80), "K": (50, 100), "ph": (5.5, 7), "temp": (18, 30), "humidity": (50, 70),
               "note":"Fruit crop, medium water", "price":(4000,6000), "season":"Feb-May", "water":(400,700), "rotation":"Pomegranate"},
    "watermelon": {"N": (50, 100), "P": (30, 50), "K": (30, 60), "ph": (6, 7), "temp": (25, 35), "humidity": (40, 60),
                   "note":"Seasonal fruit crop", "price":(1500,2500), "season":"Mar-Jun", "water":(300,500), "rotation":"Maize"},
    "apple": {"N": (100, 150), "P": (50, 80), "K": (50, 100), "ph": (6, 7.5), "temp": (15, 25), "humidity": (60, 80),
              "note":"Temperate fruit crop", "price":(7000,9000), "season":"Oct-Mar", "water":(600,900), "rotation":"None"},
    "orange": {"N": (100, 150), "P": (50, 80), "K": (50, 100), "ph": (5.5, 7), "temp": (18, 30), "humidity": (50, 80),
               "note":"Citrus fruit crop", "price":(5000,7000), "season":"Oct-Mar", "water":(500,800), "rotation":"None"},
    "papaya": {"N": (80, 150), "P": (40, 80), "K": (50, 100), "ph": (5.5, 7), "temp": (25, 35), "humidity": (70, 90),
               "note":"Quick-growing fruit", "price":(4000,5000), "season":"Year-round", "water":(800,1200), "rotation":"Banana"},
    "coconut": {"N": (150, 200), "P": (50, 100), "K": (200, 300), "ph": (5.5, 7), "temp": (25, 35), "humidity": (70, 90),
                "note":"Long-term tropical crop", "price":(7000,9000), "season":"Year-round", "water":(1200,1500), "rotation":"None"},
    "cotton": {"N": (80, 150), "P": (40, 80), "K": (60, 120), "ph": (5.5, 7), "temp": (20, 35), "humidity": (50, 80),
               "note":"Fiber crop, moderate water need", "price":(2500,3000), "season":"Jul-Oct", "water":(600,900), "rotation":"Maize"},
    "jute": {"N": (80, 150), "P": (40, 80), "K": (60, 120), "ph": (5.5, 7), "temp": (20, 35), "humidity": (60, 90),
             "note":"Fiber crop, high humidity", "price":(2000,3000), "season":"Jun-Sep", "water":(800,1200), "rotation":"Rice"},
    "coffee": {"N": (80, 120), "P": (40, 60), "K": (40, 80), "ph": (5.5, 6.5), "temp": (18, 28), "humidity": (60, 80),
               "note":"Shade-loving, long-term crop", "price":(5000,7000), "season":"Oct-Mar", "water":(1200,1500), "rotation":"Banana"}
}

weights = {"N":0.1,"P":0.1,"K":0.1,"temp":0.2,"humidity":0.2,"ph":0.2,"rainfall":0.1}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values from form
        N = int(request.form['N'])
        P = int(request.form['P'])
        K = int(request.form['K'])
        temp = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        input_data = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        ml_prediction = model.predict(input_data)[0]

        output = []
        factors = {"N":N,"P":P,"K":K,"temp":temp,"humidity":humidity,"ph":ph,"rainfall":rainfall}

        for crop, info in crop_info.items():
            score = 0
            for f, val in factors.items():
                if f in info:
                    low, high = info[f]
                    if val < low:
                        score += weights[f]*(val/low)
                    elif val > high:
                        score += weights[f]*(high/val)
                    else:
                        score += weights[f]
                else:
                    score += weights[f]

            # Adjust with rainfall realism
            ideal_rainfall = info["water"][1]/2
            score *= min(1, rainfall/ideal_rainfall)
            suitability = round(score*100,1)

            # Fertilizer advice
            fertilizer = []
            if N < info.get("N",(0,0))[0]: fertilizer.append("Nitrogen low → Apply Urea")
            if P < info.get("P",(0,0))[0]: fertilizer.append("Phosphorus low → Apply DAP")
            if K < info.get("K",(0,0))[0]: fertilizer.append("Potassium low → Apply MOP")
            if not fertilizer: fertilizer.append("Soil nutrients sufficient")

            # Price and water based on suitability
            price = int(info["price"][0] + (suitability/100)*(info["price"][1]-info["price"][0]))
            water = int(info["water"][0] + (suitability/100)*(info["water"][1]-info["water"][0]))

            output.append({
                "name": crop.title(),
                "suitability": suitability,
                "reason": f"Soil pH ideal: {info['ph']}, Temp: {info['temp']} °C, Humidity: {info['humidity']}%",
                "fertilizer": " | ".join(fertilizer),
                "note": info["note"],
                "price": price,
                "season": info["season"],
                "water": water,
                "rotation": info["rotation"]
            })

        # ✅ Only change: Top 3 crops sorted by suitability
        output = sorted(output, key=lambda x: x['suitability'], reverse=True)[:3]

        return render_template("index.html", prediction_text=output)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
