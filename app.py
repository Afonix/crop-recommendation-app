import streamlit as st
import joblib
import numpy as np

# Model loading with caching
@st.cache_resource
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Model selection
model_options = {
    "Random Forest": "models/random_forest_model.joblib",
    "Light_gbm": "models/light_gbm_model.joblib",
    "SVC": "models/svc_model.joblib"
}

# Crop advice dictionary
crop_advice = {
    'rice': {
        'icon': '🌾',
        'soil': 'Clay loam with good water retention',
        'ph_range': '5.0-6.5',
        'water': 'Maintain 2-5 cm water depth during growth',
        'pests': 'Stem borers, leaf folders → Use neem-based pesticides',
        'fertilizer': '120-150 kg N/ha split into 3 applications'
    },
    'maize': {
        'icon': '🌽',
        'soil': 'Well-drained loamy soil',
        'ph_range': '5.8-7.0',
        'spacing': '60 cm between rows, 20 cm between plants',
        'pests': 'Fall armyworm → Use spinosad insecticides',
        'harvest': 'Harvest when kernels harden (20-25% moisture)'
    },
    'kidneybeans': {
        'icon': '🫘',
        'soil': 'Sandy loam with good drainage',
        'ph_range': '6.0-7.5',
        'rotation': 'Rotate with cereals to break disease cycles',
        'pests': 'Bean beetles → Use pyrethrin sprays',
        'watering': 'Avoid overhead irrigation to prevent fungal issues'
    },
    'banana': {
        'icon': '🍌',
        'planting': 'Use tissue-cultured plants, 2m spacing',
        'fertilizer': '250g N, 300g K per plant annually',
        'pests': 'Sigatoka disease → Apply Bordeaux mixture',
        'harvest': 'Cut bunches when fingers are 75% rounded'
    },
    'mango': {
        'icon': '🥭',
        'pruning': 'Open-center pruning for better light penetration',
        'flowering': 'Apply 1% KNO3 spray to induce flowering',
        'pests': 'Fruit flies → Use pheromone traps',
        'storage': 'Harvest at mature green stage for long transport'
    },
    'grapes': {
        'icon': '🍇',
        'trellis': 'Install Y-shaped trellis system',
        'pruning': 'Spur pruning (2-3 buds per spur)',
        'pests': 'Downy mildew → Apply copper fungicides',
        'harvest': 'Pick when Brix reaches 18-22%'
    },
    'watermelon': {
        'icon': '🍉',
        'spacing': '2m between hills, 3m between rows',
        'pollination': 'Ensure 1:3 honeybee hives per hectare',
        'pests': 'Aphids → Release ladybird beetles',
        'ripeness': 'Look for yellow ground spot and dried tendril'
    },
    'apple': {
        'icon': '🍎',
        'chilling': 'Requires 800-1200 hours below 7°C',
        'pruning': 'Summer pruning for better fruit color',
        'pests': 'Codling moth → Use mating disruption',
        'storage': 'Keep at 0-2°C with 90-95% humidity'
    },
    'orange': {
        'icon': '🍊',
        'irrigation': 'Drip irrigation during fruit development',
        'nutrition': 'Foliar zinc and manganese sprays',
        'pests': 'Citrus psylla → Use imidacloprid',
        'harvest': 'Pick when juice content reaches 35%'
    },
    'papaya': {
        'icon': '🥔',
        'planting': '3 females per 100 plants for pollination',
        'fertilizer': '200g N + 250g K per plant every 3 months',
        'pests': 'Papaya ring spot virus → Remove infected plants',
        'harvest': 'Pick when 25% yellow coloration appears'
    },
    'coconut': {
        'icon': '🥥',
        'spacing': '7-8m triangular planting',
        'fertilizer': '1.3kg NaCl per palm annually',
        'pests': 'Rhinoceros beetle → Apply Metarhizium',
        'processing': 'Dehusk within 2 weeks of harvest'
    },
    'cotton': {
        'icon': '🧵',
        'irrigation': 'Critical at flowering and boll formation',
        'pests': 'Bollworm → Use Bt cotton varieties',
        'harvest': 'Pick when bolls open completely',
        'grading': 'Sort by staple length (28-34mm ideal)'
    },
    'coffee': {
        'icon': '☕',
        'shade': 'Maintain 40-50% shade with silver oak',
        'processing': 'Wet process for premium quality',
        'pests': 'Berry borer → Use Beauveria bassiana',
        'roasting': 'Medium roast at 210-220°C for best flavor'
    },
    'default': {
        'icon': '🌱',
        'tips': [
            'Test soil every 2-3 years',
            'Maintain proper drainage',
            'Use crop rotation practices',
            'Monitor weather forecasts regularly',
            'Keep field records for better planning'
        ]
    }
}

# Streamlit UI
st.title("Smart Crop Advisor")
st.markdown("### Multi-Model Crop Recommendation System")

# Model selection
selected_model = st.selectbox("Choose Prediction Model", list(model_options.keys()))
model = load_model(model_options[selected_model])

# Input columns
col1, col2 = st.columns(2)
with col1:
    N = st.slider("Nitrogen (ppm)", 0, 150, 50)
    P = st.slider("Phosphorus (ppm)", 0, 100, 30)
    K = st.slider("Potassium (ppm)", 0, 200, 40)
    temp = st.slider("Temperature (°C)", 10.0, 40.0, 25.0)

with col2:
    humidity = st.slider("Humidity (%)", 20.0, 100.0, 65.0)
    ph = st.slider("Soil pH", 4.0, 9.0, 6.5)
    rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 120.0)

# Prediction
if model and st.button("Get Recommendation"):
    input_data = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    
    try:
        prediction = model.predict(input_data)[0].lower()
        try:
            confidence = np.max(model.predict_proba(input_data))
        except AttributeError:
            confidence = 1.0  # For models without probability
        
        st.success(f"**Recommended Crop:** {prediction.capitalize()} {crop_advice.get(prediction, {}).get('icon', '🌱')}")
        st.metric("Confidence Score", f"{confidence:.1%}")
        
        # Display advice
        if prediction in crop_advice:
            advice = crop_advice[prediction]
            st.subheader("Cultivation Guidelines")
            cols = st.columns(2)
            for key, value in advice.items():
                if key != 'icon':
                    cols[0].write(f"**{key.replace('_', ' ').title()}:**")
                    cols[1].write(value)
        else:
            st.subheader("General Farming Best Practices")
            for tip in crop_advice['default']['tips']:
                st.write(f"🌱 {tip}")
                
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")