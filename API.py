from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# สร้าง FastAPI instance
app = FastAPI()

# โหลด Random Forest model สำหรับ Optimal RPM
with open("random_forest_optimalRPM.pkl", "rb") as file:
    random_forest_optimalRPM = pickle.load(file)

# โหลด StandardScaler ที่ใช้ในตอนเทรน (สำหรับ Optimal RPM)
with open("scaler_optimalRPM.pkl", "rb") as file:
    scaler_optimalRPM = pickle.load(file)

# โหลด Neural Network model ที่บันทึกไว้
with open("nn_model_Efficiency_Percentage.pkl", "rb") as file:
    nn_model_Efficiency_Percentage = pickle.load(file)

# โหลด StandardScaler ที่ใช้ในตอนเทรน (ต้องบันทึก scaler ไว้ตอนเทรนด้วย)
with open("scale_Efficiency_Percentage.pkl", "rb") as file:
    scaler_Efficiency_Percentage = pickle.load(file)

# โหลด Neural Network model ที่บันทึกไว้
with open("nn_model_Remaining_Lifespan.pkl", "rb") as file:
    nn_model_Remaining_Lifespan = pickle.load(file)

# โหลด StandardScaler ที่ใช้ในตอนเทรน (ต้องบันทึก scaler ไว้ตอนเทรนด้วย)
with open("scale_Remaining_Lifespan.pkl", "rb") as file:
    scaler_Remaining_Lifespan = pickle.load(file)

# Define the input schema for Optimal RPM prediction
class Optimal_RPM_Features(BaseModel):
    flow_rate: float
    inlet_temperature: float
    outlet_temperature: float
    delta_temperature: float
    pressure: float
    delta_pressure: float
    power_consumption: float
    vibration: float
    ambient_temperature: float
    time_of_day: str  # 'Peak' or 'Off-Peak'
    cooling_load: float
    motor_speed: float
    energy_usage_regular_rpm: float


# Define the input schema for Efficiency Percentage
class Efficiency_Percentage_Features(BaseModel):
    flow_rate: float
    inlet_temperature: float
    outlet_temperature: float
    delta_temperature: float
    pressure: float
    delta_pressure: float
    power_consumption: float
    vibration: float
    ambient_temperature: float
    cooling_load: float
    motor_speed: float
    energy_usage_regular_rpm: float


# Define the input schema for Remaining Lifespan
class Remaining_Lifespan_Features(BaseModel):
    delta_temperature: float
    pressure: float
    delta_pressure: float
    power_consumption: float
    cooling_load: float
    motor_speed: float
    vibration: float
    ambient_temperature: float


# Endpoint สำหรับตรวจสอบสถานะ
@app.get("/")
def read_root():
    return {"message": "Neural Network API for Efficiency Percentage and Remaining Lifespan Prediction is running!"}

# Endpoint สำหรับพยากรณ์ Optimal RPM
@app.post("/predict_optimalRPM/")
def predict_optimalRPM(features: Optimal_RPM_Features):
    # แปลง 'Time of Day' ให้เป็นตัวเลข ('Peak' = 1, 'Off-Peak' = 0)
    time_of_day = 1 if features.time_of_day == 'Peak' else 0
    
    # เตรียมข้อมูล input
    input_data = np.array([[
        features.flow_rate,
        features.inlet_temperature,
        features.outlet_temperature,
        features.delta_temperature,
        features.pressure,
        features.delta_pressure,
        features.power_consumption,
        features.vibration,
        features.ambient_temperature,
        time_of_day,  # encoded 'Time of Day'
        features.cooling_load,
        features.motor_speed,
        features.energy_usage_regular_rpm
    ]])

    # ปรับ scale ของข้อมูล input ด้วย StandardScaler
    input_data_scaled = scaler_optimalRPM.transform(input_data)

    # ใช้โมเดล Random Forest ในการพยากรณ์ Optimal RPM
    prediction = random_forest_optimalRPM.predict(input_data_scaled)

    # Return ผลลัพธ์
    return {"Optimal RPM": float(prediction[0])}

# Endpoint สำหรับพยากรณ์ Remaining Lifespan
@app.post("/predict_lifespan/")
def predict_lifespan(features: Remaining_Lifespan_Features):
    # รับค่า input มาเป็น array
    input_data = np.array([[
        features.delta_temperature,
        features.pressure,
        features.delta_pressure,
        features.power_consumption,
        features.cooling_load,
        features.motor_speed,
        features.vibration,
        features.ambient_temperature
    ]])

    # ทำการปรับ scale ของ input ด้วย StandardScaler
    input_data_scaled = scaler_Remaining_Lifespan.transform(input_data)

    # ใช้โมเดล Neural Network ในการพยากรณ์
    prediction = nn_model_Remaining_Lifespan.predict(input_data_scaled)

    # Return ผลลัพธ์
    return {"Remaining Lifespan (years)": float(prediction[0])}


# Endpoint สำหรับพยากรณ์ Efficiency Percentage
@app.post("/predict_efficiency/")
def predict_efficiency(features: Efficiency_Percentage_Features):
    # รับค่า input มาเป็น array
    input_data = np.array([[
        features.flow_rate,
        features.inlet_temperature,
        features.outlet_temperature,
        features.delta_temperature,
        features.pressure,
        features.delta_pressure,
        features.power_consumption,
        features.vibration,
        features.ambient_temperature,
        features.cooling_load,
        features.motor_speed,
        features.energy_usage_regular_rpm
    ]])

    # ทำการปรับ scale ของ input ด้วย StandardScaler
    input_data_scaled = scaler_Efficiency_Percentage.transform(input_data)

    # ใช้โมเดล Neural Network ในการพยากรณ์
    prediction = nn_model_Efficiency_Percentage.predict(input_data_scaled)

    # Return ผลลัพธ์
    return {"Efficiency Percentage": float(prediction[0])}