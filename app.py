from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

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

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    teks_bawah = None  # Tambahkan variabel teks_bawah

    features = ['Gender', 'Country', 'Occupation', 'self_employed', 'family_history',
                'Days_Indoors', 'Growing_Stress', 'Changes_Habits', 'Mental_Health_History',
                'Mood_Swings', 'Coping_Struggles', 'Work_Interest', 'Social_Weakness']

    if request.method == 'POST':
        input_data = []

        for feature in features:
            value = request.form.get(feature)

            # Mapping bahasa Indonesia ke English
            if feature == "Gender":
                value = mapping_gender.get(value, value)
            elif feature == "Country":
                value = mapping_country.get(value, value)
            elif feature == "Occupation":
                value = mapping_occupation.get(value, value)
            elif feature in ["self_employed", "family_history", "Growing_Stress", "Changes_Habits",
                             "Mental_Health_History", "Coping_Struggles", "Work_Interest", "Social_Weakness"]:
                value = mapping_yesno.get(value, value)
            elif feature == "Days_Indoors":
                value = mapping_daysindoors.get(value, value)
            elif feature == "Mood_Swings":
                value = mapping_moodswings.get(value, value)

            # Encode fitur dengan LabelEncoder
            encoder = encoders.get(feature)
            if value in encoder.classes_:
                encoded = encoder.transform([value])[0]
            else:
                encoded = 0  # fallback jika label tidak dikenal

            input_data.append(encoded)

        input_array = np.array([input_data])

        # Prediksi dan probabilitas
        pred_encoded = model.predict(input_array)[0]
        proba = model.predict_proba(input_array)[0][pred_encoded] * 100

        # Decode target label
        prediction = target_encoder.inverse_transform([pred_encoded])[0]

        # Buat teks_bawah sesuai prediksi
        if prediction == "No":
            teks_bawah = f"Tidak perlu bantuan ahli saat ini. Keyakinan model : {proba:.2f}%"
        else:
            teks_bawah = f"Disarankan untuk mendapatkan bantuan ahli. Keyakinan model : {proba:.2f}%"

        # Bulatkan probabilitas untuk ditampilkan
        probability = round(proba, 2)

    return render_template(
        'index.html',
        prediction=prediction,
        probability=probability,
        teks_bawah=teks_bawah
    )

if __name__ == '__main__':
    app.run(debug=True)
