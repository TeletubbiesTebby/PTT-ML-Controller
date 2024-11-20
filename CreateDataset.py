import numpy as np
import pandas as pd

# ตั้งค่าจำนวนตัวอย่าง
num_samples = 5000

# กำหนดขอบเขตค่าจำลอง (อิงตามสถานการณ์จริงในระบบหล่อเย็นในไทย)
np.random.seed(42)

# สร้างข้อมูลจำลอง
data = {
    "Flow Rate (L/min)": np.random.uniform(500, 2000, num_samples),  # อัตราการไหลในระบบ
    "Inlet Temperature (°C)": np.random.uniform(30, 50, num_samples),  # อุณหภูมิน้ำเข้า
    "Outlet Temperature (°C)": np.random.uniform(25, 45, num_samples),  # อุณหภูมิน้ำออก
    "Delta Temperature (°C)": np.abs(np.random.uniform(5, 15, num_samples)),  # ความต่างอุณหภูมิ
    "Pressure (Bar)": np.random.uniform(1, 5, num_samples),  # แรงดันในระบบ
    "Delta Pressure (Bar)": np.random.uniform(0.1, 0.5, num_samples),  # ความต่างแรงดัน
    "Power Consumption (kW)": np.random.uniform(10, 50, num_samples),  # พลังงานที่ใช้โดยปั๊ม
    "Vibration (m/s²)": np.random.uniform(0.1, 2.0, num_samples),  # การสั่นสะเทือนของปั๊ม
    "Ambient Temperature (°C)": np.random.uniform(25, 40, num_samples),  # อุณหภูมิภายนอก
    "Time of Day": np.random.choice(["Peak", "Off-Peak"], num_samples),  # ช่วงเวลาของวัน
    "Cooling Load (kW)": np.random.uniform(20, 100, num_samples),  # ภาระความเย็น
    "Motor Speed (RPM)": np.random.uniform(1000, 1800, num_samples),  # ความเร็วมอเตอร์ปัจจุบัน
    "Regular RPM (RPM)": np.full(num_samples, 1500),  # ความเร็วปกติที่ใช้
    "Energy Usage (Regular RPM) (kW)": np.random.uniform(20, 60, num_samples),  # พลังงานที่ใช้เมื่อใช้ความเร็วปกติ
}

# สร้าง Target: Optimal RPM
data["Optimal RPM"] = np.clip(data["Motor Speed (RPM)"] * 0.85, 800, 1500)  # Optimal RPM ประมาณ 85% ของความเร็วเดิม
data["Energy Usage (Optimal RPM) (kW)"] = data["Energy Usage (Regular RPM) (kW)"] * 0.7  # ลดพลังงานจาก Optimal RPM

# สร้าง DataFrame
dataset = pd.DataFrame(data)

# บันทึกข้อมูลลงไฟล์
file_path = "CSV/cooling_system_data.csv"
dataset.to_csv(file_path, index=False)

# แสดงตัวอย่างข้อมูล
import ace_tools as tools; tools.display_dataframe_to_user(name="Cooling System ML Dataset", dataframe=dataset.head())

file_path
