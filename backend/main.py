from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.layers import Dense as _Dense

class Dense(_Dense):
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)


import joblib
import numpy as np
import pandas as pd

app = FastAPI()

model = tf.keras.models.load_model(
    'student_health_monitor.h5', custom_objects={'Dense': Dense}
)
scaler = joblib.load('scaler.pickel')

class InputData(BaseModel):
    sleep_duration: float
    heart_rate: float
    bmi: float
    calorie_expenditure: float
    step_count: float
    exercise_duration: float
    water_intake: float
    diet_type: int
    stress_level: int
    sleep_quality: int
    physical_activity_level: int
    smoking_alcohol: int
    gender: int


@app.post('/predict')
def predict(data: InputData):
    input_array = np.array([[
        data.sleep_duration, data.heart_rate, data.bmi, data.calorie_expenditure, data.step_count, data.exercise_duration, data.water_intake, data.diet_type, data.stress_level, data.sleep_quality, data.physical_activity_level, data.smoking_alcohol, data.gender
    ]])

    columns = ['sleep_duration','heart_rate','bmi','calorie_expenditure','step_count',
           'exercise_duration','water_intake','diet_type','stress_level',
           'sleep_quality','physical_activity_level','smoking_alcohol','gender']

    input_df = pd.DataFrame(input_array, columns=columns)


    scaled_input = scaler.transform(input_df.values)
    prediction = model.predict(scaled_input)
    predicted_class = int(np.argmax(prediction))
    return {"prediction": predicted_class}



from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"]
)