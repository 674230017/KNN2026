import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    # ---------------------------------------------------------
    # เพิ่มส่วน: การแสดงผลกราฟ (Data Visualization)
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("กราฟวิเคราะห์ข้อมูลเบื้องต้น")
    
    # ใช้ Tab เพื่อให้กราฟดูเป็นระเบียบ ไม่กินพื้นที่หน้าเว็บ
    tab1, tab2, tab3 = st.tabs(["สัดส่วนการเป็นโรคหัวใจ", "อายุ vs โรคหัวใจ", "คอเลสเตอรอล vs โรคหัวใจ"])
    
    with tab1:
        st.markdown("**สัดส่วนของผู้ป่วยและผู้ที่ไม่เป็นโรคหัวใจในชุดข้อมูล**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='HeartDisease', palette='Set2', ax=ax1)
        ax1.set_xticklabels(['ปกติ (0)', 'มีความเสี่ยง (1)'])
        ax1.set_xlabel("สถานะโรคหัวใจ")
        ax1.set_ylabel("จำนวนคน")
        st.pyplot(fig1)
        
    with tab2:
        st.markdown("**การกระจายตัวของอายุ เปรียบเทียบกับการเป็นโรคหัวใจ**")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.histplot(data=df, x='Age', hue='HeartDisease', multiple='stack', palette='coolwarm', ax=ax2, bins=15)
        ax2.set_xlabel("อายุ (ปี)")
        ax2.set_ylabel("จำนวนคน")
        st.pyplot(fig2)
        
    with tab3:
        st.markdown("**ความสัมพันธ์ระหว่างคอเลสเตอรอลและการเป็นโรคหัวใจ**")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df, x='HeartDisease', y='Cholesterol', palette='pastel', ax=ax3)
        ax3.set_xticklabels(['ปกติ (0)', 'มีความเสี่ยง (1)'])
        ax3.set_xlabel("สถานะโรคหัวใจ")
        ax3.set_ylabel("คอเลสเตอรอล")
        st.pyplot(fig3)
    # ---------------------------------------------------------

    # เตรียมข้อมูลสำหรับเทรนโมเดล KNN
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    # แบ่งข้อมูล Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # สร้างและเทรนโมเดล KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # ทดสอบความแม่นยำ
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100

    # 4. ส่วนแสดงประสิทธิภาพโมเดล
    st.markdown("---")
    st.subheader("ประสิทธิภาพโมเดล KNN")
    st.metric(label="Accuracy", value=f"{acc:.2f}%")

    # 5. ส่วนฟอร์มทำนายผล
    st.markdown("<h3 style='text-align: center; background-color: #63B8B8; color: white; padding: 10px; border-radius: 5px;'>ทำนายข้อมูลโรคหัวใจ</h3>", unsafe_allow_html=True)
    st.write("กรุณากรอกข้อมูลเพื่อทำนายผล")