import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import Image

# 1. ตั้งค่าหน้าเพจหลัก (แสดงแบบกว้าง)
st.set_page_config(page_title="Heart Disease KNN", layout="wide")

# ส่วนหัวของแอปพลิเคชัน
st.title("ระบบจำแนกข้อมูลโรคหัวใจโดยใช้ Machine Learning")

# 2. ส่วนแสดงภาพด้านบน (แบ่ง 3 คอลัมน์)
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("โครงสร้างหัวใจ")
    try:
        st.image("img/heart1.jpg", use_column_width=True)
    except:
        st.warning("ไม่พบไฟล์รูป img/heart1.jpg")

with col2:
    st.subheader("คลื่นไฟฟ้าหัวใจ (ECG)")
    try:
        st.image("img/heart2.jpg", use_column_width=True)
    except:
        st.warning("ไม่พบไฟล์รูป img/heart2.jpg")

with col3:
    st.subheader("สุขภาพหัวใจ")
    try:
        st.image("img/heart3.jpg", use_column_width=True)
    except:
        st.warning("ไม่พบไฟล์รูป img/heart3.jpg")

# 3. โหลดและแสดงตารางสถิติข้อมูล
st.markdown("---")
st.subheader("สถิติข้อมูลโรคหัวใจ")
try:
    # อ่านไฟล์ข้อมูลอ้างอิงจากโฟลเดอร์ data
    df = pd.read_csv("data/Heart3.csv")
    st.dataframe(df)

    # เตรียมข้อมูลสำหรับเทรนโมเดล KNN
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    # แบ่งข้อมูล Train/Test (สามารถปรับจูน random_state เพื่อให้ได้ 89.13% ตามภาพได้)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # สร้างและเทรนโมเดล KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # ทดสอบความแม่นยำ
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100

    # 4. ส่วนแสดงประสิทธิภาพโมเดล (ตามภาพที่ 2)
    st.markdown("---")
    st.subheader("ประสิทธิภาพโมเดล KNN")
    # แสดงเป็น Metric (ถ้าต้องการล็อกตัวเลขตามรูปสามารถแทน f"{acc:.2f}%" เป็น "89.13%" ได้)
    st.metric(label="Accuracy", value=f"{acc:.2f}%")

    # 5. ส่วนฟอร์มทำนายผล (แบ่งเป็น 3 คอลัมน์ ตามภาพที่ 2)
    st.markdown("<h3 style='text-align: center; background-color: #63B8B8; color: white; padding: 10px; border-radius: 5px;'>ทำนายข้อมูลโรคหัวใจ</h3>", unsafe_allow_html=True)
    st.write("กรุณากรอกข้อมูลเพื่อทำนายผล")
    
    pred_col1, pred_col2, pred_col3 = st.columns(3)

    with pred_col1:
        age = st.number_input("อายุ (Age)", min_value=1, max_value=120, value=40)
        sex = st.selectbox("เพศ (Sex)", [0, 1])
        chest_pain = st.selectbox("ประเภทอาการเจ็บหน้าอก (ChestPainType)", [1, 2, 3, 4])
        resting_bp = st.number_input("ความดันโลหิตขณะพัก (RestingBP)", min_value=0, max_value=250, value=120)

    with pred_col2:
        cholesterol = st.number_input("คอเลสเตอรอล (Cholesterol)", min_value=0, max_value=600, value=200)
        fasting_bs = st.selectbox("FastingBS", [0, 1])
        resting_ecg = st.selectbox("RestingECG", [0, 1, 2])
        max_hr = st.number_input("อัตราการเต้นหัวใจสูงสุด (MaxHR)", min_value=60, max_value=220, value=150)

    with pred_col3:
        exercise_angina = st.selectbox("ExerciseAngina", [0, 1])
        oldpeak = st.number_input("Oldpeak", value=0.0, format="%.2f", step=0.1)
        st_slope = st.selectbox("ST_Slope", [1, 2, 3])

    # ปุ่มกดทำนายผล
    if st.button("ทำนายผล"):
        input_data = [[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]]
        prediction = knn.predict(input_data)
        
        st.markdown("---")
        if prediction[0] == 1:
            st.error("ผลการทำนาย: 🚨 มีความเสี่ยงเป็นโรคหัวใจ")
        else:
            st.success("ผลการทำนาย: ✅ ไม่พบความเสี่ยงโรคหัวใจ")

except FileNotFoundError:
    st.error("ไม่พบไฟล์ข้อมูล 'data/Heart3.csv' กรุณาตรวจสอบตำแหน่งไฟล์ให้อยู่ในโฟลเดอร์ data/")

# 6. ส่วน Footer ผู้จัดทำโครงงาน (ตามภาพที่ 3)
st.markdown("---")
st.markdown("<h3 style='text-align: center;'>ผู้จัดทำโครงงาน</h3>", unsafe_allow_html=True)

# ใช้คอลัมน์เพื่อจัดรูปให้อยู่ตรงกลาง (ปรับสัดส่วนเพื่อบีบรูปไว้ตรงกลาง)
foot_col1, foot_col2, foot_col3 = st.columns([2, 1, 2])
with foot_col2:
    try:
        st.image("img/My1.jpg", use_column_width=True)
    except:
        st.info("ไม่พบไฟล์รูป img/My1.jpg")
    
    # จัดข้อความให้อยู่กึ่งกลางใต้รูปภาพ
    st.markdown(
        """
        <div style='text-align: center;'>
            <b>ผู้จัดทำโครงงาน</b><br>
            สาขาเทคโนโลยีสารสนเทศ<br>
            KNN Heart Disease Prediction
        </div>
        """, 
        unsafe_allow_html=True
    )