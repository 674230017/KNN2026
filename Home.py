import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. ตั้งค่าหน้าเพจหลัก
st.set_page_config(page_title="Heart Disease KNN", layout="wide")

st.write("ระบบจำแนกข้อมูลโรคหัวใจโดยใช้ Machine Learning")

# 2. ส่วนแสดงภาพด้านบน (แบ่ง 3 คอลัมน์)
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("โครงสร้างหัวใจ")
    try:
        st.image("img/heart1.jpg", use_container_width=True)
    except:
        st.warning("ไม่พบไฟล์รูป img/heart1.jpg")

with col2:
    st.subheader("คลื่นไฟฟ้าหัวใจ (ECG)")
    try:
        st.image("img/heart2.jpg", use_container_width=True)
    except:
        st.warning("ไม่พบไฟล์รูป img/heart2.jpg")

with col3:
    st.subheader("สุขภาพหัวใจ")
    try:
        st.image("img/heart3.jpg", use_container_width=True)
    except:
        st.warning("ไม่พบไฟล์รูป img/heart3.jpg")

# 3. โหลดและแสดงตารางสถิติข้อมูล
st.markdown("---")
st.subheader("สถิติข้อมูลโรคหัวใจ")

try:
    df = pd.read_csv("data/Heart3.csv")
    st.dataframe(df, height=250)

    # แถบสีแดงใต้ตารางตามแบบในรูป
    st.markdown(
        """
        <div style="background-color:#EC7063; padding:10px; border-radius:10px; text-align:center; color:black; font-weight:bold;">
            สถิติข้อมูลโรคหัวใจ
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    # ---------------------------------------------------------
    # ส่วนกราฟ: ย่อขนาดและวางคู่กัน 2 คอลัมน์ (วงกลม + แท่ง)
    # ---------------------------------------------------------
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**สัดส่วนผู้มีความเสี่ยงโรคหัวใจ (Pie Chart)**")
        fig1, ax1 = plt.subplots(figsize=(4.5, 3.2))
        counts = df['HeartDisease'].value_counts().sort_index()
        ax1.pie(
            counts,
            labels=['Normal (0)', 'Heart Disease (1)'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['#66B2FF', '#FF9999']
        )
        ax1.axis('equal')
        plt.tight_layout()
        st.pyplot(fig1)

    with chart_col2:
        st.markdown("**จำนวนข้อมูลแยกตามอาการเจ็บหน้าอก (Bar Chart)**")
        fig2, ax2 = plt.subplots(figsize=(4.5, 3.2))
        sns.countplot(
            data=df,
            x='ChestPainType',
            hue='HeartDisease',
            palette=['#66B2FF', '#FF9999'],
            ax=ax2
        )
        ax2.set_xlabel("ChestPainType (1-4)")
        ax2.set_ylabel("Count")
        ax2.legend(title="HeartDisease", labels=["Normal (0)", "Disease (1)"], fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2)
    # ---------------------------------------------------------

    # 4. เตรียมข้อมูลและเทรนโมเดล KNN