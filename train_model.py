import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
df = pd.read_csv('Mental Health Dataset.csv')

# Drop kolom yang tidak digunakan
df = df.drop(['Timestamp', 'mental_health_interview', 'care_options'], axis=1)

# Drop NA
df = df.dropna()

# Tambah 1 dummy row dengan semua kemungkinan label dari form
dummy_row = {
    'Gender': 'Laki-laki',
    'Country': 'Indonesia',
    'Occupation': 'Mahasiswa',
    'self_employed': 'Ya',
    'family_history': 'Tidak',
    'Days_Indoors': '1-14 hari',
    'Growing_Stress': 'Ya',
    'Changes_Habits': 'Tidak',
    'Mental_Health_History': 'Tidak',
    'Mood_Swings': 'Sedang (Kadang-kadang)',
    'Coping_Struggles': 'Ya',
    'Work_Interest': 'Ya',
    'Social_Weakness': 'Tidak',
    'treatment': 'Yes'
}
df = pd.concat([df, pd.DataFrame([dummy_row])], ignore_index=True)

# Pisahkan fitur dan target
X = df.drop('treatment', axis=1)
y = df['treatment']

# Encode fitur
encoders = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Evaluasi
acc_lr = accuracy_score(y_test, lr.predict(X_test))
acc_rf = accuracy_score(y_test, rf.predict(X_test))

print("=== Logistic Regression ===")
print(f"Akurasi: {acc_lr:.4f}")
print(classification_report(y_test, lr.predict(X_test)))

print("\n=== Random Forest ===")
print(f"Akurasi: {acc_rf:.4f}")
print(classification_report(y_test, rf.predict(X_test)))

# Pilih model terbaik
if acc_rf >= acc_lr:
    best_model = rf
    print("\n✅ Model terbaik: Random Forest")
else:
    best_model = lr
    print("\n✅ Model terbaik: Logistic Regression")

# Simpan model & encoder
pickle.dump(best_model, open('model.pkl', 'wb'))
pickle.dump(encoders, open('encoders.pkl', 'wb'))
pickle.dump(target_encoder, open('target_encoder.pkl', 'wb'))

print("📦 Model & encoder berhasil disimpan!")
