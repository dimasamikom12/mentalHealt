import streamlit as st
import pickle
import numpy as np

# Load model dan encoder
model = pickle.load(open('model.pkl', 'rb'))
encoders = pickle.load(open('encoders.pkl', 'rb'))
target_encoder = pickle.load(open('target_encoder.pkl', 'rb'))

# Mapping Bahasa Indonesia ke English
mapping_gender = {
    "Laki-laki": "Male",
    "Perempuan": "Female",
    "Lainnya": "Other"
}

mapping_country = {
    "Indonesia": "Indonesia",
    "Negara lain": "Other"
}

mapping_occupation = {
    "Mahasiswa": "Student",
    "Karyawan": "Corporate employee",
    "Freelancer": "Freelancer",
    "Wirausaha": "Entrepreneur",
    "Lainnya": "Other"
}

mapping_yesno = {
    "Ya": "Yes",
    "Tidak": "No"
}

mapping_daysindoors = {
    "1-14 hari": "1-14 days",
    "15-30 hari": "15-30 days",
    "Lebih dari 30 hari": "More than 30 days"
}

mapping_moodswings = {
    "Rendah (Jarang)": "Low",
    "Sedang (Kadang-kadang)": "Medium",
    "Tinggi (Sering)": "High"
}

# Daftar fitur yang digunakan
features = ['Gender', 'Country', 'Occupation', 'self_employed', 'family_history',
            'Days_Indoors', 'Growing_Stress', 'Changes_Habits', 'Mental_Health_History',
            'Mood_Swings', 'Coping_Struggles', 'Work_Interest', 'Social_Weakness']

# Streamlit UI
st.set_page_config(page_title="Prediksi Kesehatan Mental", layout="centered")
st.title("Prediksi Kebutuhan Bantuan Ahli Kesehatan Mental")

with st.form("mental_health_form"):
    gender = st.selectbox("Jenis Kelamin", list(mapping_gender.keys()))
    country = st.selectbox("Negara", list(mapping_country.keys()))
    occupation = st.selectbox("Pekerjaan", list(mapping_occupation.keys()))
    self_employed = st.radio("Apakah bekerja sendiri?", list(mapping_yesno.keys()))
    family_history = st.radio("Ada riwayat kesehatan mental dalam keluarga?", list(mapping_yesno.keys()))
    days_indoors = st.selectbox("Berapa lama kamu di dalam rumah?", list(mapping_daysindoors.keys()))
    growing_stress = st.radio("Apakah kamu merasa stres meningkat?", list(mapping_yesno.keys()))
    changes_habits = st.radio("Ada perubahan kebiasaan?", list(mapping_yesno.keys()))
    mental_health_history = st.radio("Ada riwayat kesehatan mental?", list(mapping_yesno.keys()))
    mood_swings = st.selectbox("Perubahan suasana hati", list(mapping_moodswings.keys()))
    coping_struggles = st.radio("Sulit mengatasi tekanan?", list(mapping_yesno.keys()))
    work_interest = st.radio("Kehilangan minat terhadap pekerjaan?", list(mapping_yesno.keys()))
    social_weakness = st.radio("Merasa lemah dalam hubungan sosial?", list(mapping_yesno.keys()))
    
    submitted = st.form_submit_button("Prediksi")

if submitted:
    # Siapkan data input
    input_data = []

    form_values = {
        'Gender': mapping_gender[gender],
        'Country': mapping_country[country],
        'Occupation': mapping_occupation[occupation],
        'self_employed': mapping_yesno[self_employed],
        'family_history': mapping_yesno[family_history],
        'Days_Indoors': mapping_daysindoors[days_indoors],
        'Growing_Stress': mapping_yesno[growing_stress],
        'Changes_Habits': mapping_yesno[changes_habits],
        'Mental_Health_History': mapping_yesno[mental_health_history],
        'Mood_Swings': mapping_moodswings[mood_swings],
        'Coping_Struggles': mapping_yesno[coping_struggles],
        'Work_Interest': mapping_yesno[work_interest],
        'Social_Weakness': mapping_yesno[social_weakness],
    }

    for feature in features:
        value = form_values[feature]
        encoder = encoders.get(feature)
        if value in encoder.classes_:
            encoded = encoder.transform([value])[0]
        else:
            encoded = 0  # fallback jika label tidak dikenal
        input_data.append(encoded)

    input_array = np.array([input_data])

    # Prediksi
    pred_encoded = model.predict(input_array)[0]
    proba = model.predict_proba(input_array)[0][pred_encoded] * 100
    prediction = target_encoder.inverse_transform([pred_encoded])[0]

    # Output
    st.markdown("---")
    if prediction == "No":
        st.success(f"Tidak perlu bantuan ahli saat ini. ✅\n\n**Keyakinan model:** {proba:.2f}%")
    else:
        st.error(f"Disarankan untuk mendapatkan bantuan ahli. 🧠\n\n**Keyakinan model:** {proba:.2f}%")

