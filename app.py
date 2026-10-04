import os
import pickle
from simple_model import BernoulliNB, SimpleLabelEncoder
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from collections import Counter
from models import db, User, Prediction, Doctor, Appointment

app = Flask(__name__)

# Database: online MySQL via DATABASE_URL (e.g. Aiven); locally we use XAMPP MySQL
db_url = os.environ.get('DATABASE_URL', 'mysql+mysqlconnector://root:@localhost/healthcare_db')
if db_url.startswith('mysql://'):
    db_url = db_url.replace('mysql://', 'mysql+mysqlconnector://', 1)
db_url = db_url.split('?')[0]  # drop ?ssl-mode=REQUIRED; the connector uses SSL automatically

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-change-this')
app.config['SQLALCHEMY_DATABASE_URI'] = db_url
db.init_app(app)

# Create tables on startup (gunicorn does not run the __main__ block below)
with app.app_context():
    db.create_all()

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered!')
            return redirect(url_for('register'))

        hashed_pw = generate_password_hash(password)
        new_user = User(name=name, email=email, password=hashed_pw, role=role)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            elif user.role == 'doctor':
                return redirect(url_for('doctor_dashboard'))
            else:
                return redirect(url_for('patient_dashboard'))
        else:
            flash('Invalid email or password')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/patient-dashboard')
@login_required
def patient_dashboard():
    return render_template('dashboard.html', user=current_user)

@app.route('/doctor-dashboard')
@login_required
def doctor_dashboard():
    return render_template('admin_dashboard.html', user=current_user)

@app.route('/admin-dashboard')
@login_required
def admin_dashboard():
    return render_template('admin_dashboard.html', user=current_user)

# Load ML model
with open('ml_model/disease_model.pkl', 'rb') as f:
    disease_model = pickle.load(f)

with open('ml_model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    result = None
    if request.method == 'POST':
        fever = int(request.form.get('fever', 0))
        cough = int(request.form.get('cough', 0))
        headache = int(request.form.get('headache', 0))
        fatigue = int(request.form.get('fatigue', 0))
        body_pain = int(request.form.get('body_pain', 0))
        sore_throat = int(request.form.get('sore_throat', 0))
        nausea = int(request.form.get('nausea', 0))
        rash = int(request.form.get('rash', 0))

        features = np.array([[fever, cough, headache, fatigue, body_pain, sore_throat, nausea, rash]])
        prediction = disease_model.predict(features)
        result = label_encoder.inverse_transform(prediction)[0]

        symptoms_str = f"fever:{fever},cough:{cough},headache:{headache},fatigue:{fatigue},body_pain:{body_pain},sore_throat:{sore_throat},nausea:{nausea},rash:{rash}"
        new_prediction = Prediction(patient_id=current_user.id, symptoms=symptoms_str, predicted_disease=result)
        db.session.add(new_prediction)
        db.session.commit()

    return render_template('predict.html', result=result)

@app.route('/book-appointment', methods=['GET', 'POST'])
@login_required
def book_appointment():
    doctors = User.query.filter_by(role='doctor').all()

    if request.method == 'POST':
        doctor_id = request.form['doctor_id']
        date = request.form['date']
        time = request.form['time']

        new_appointment = Appointment(
            patient_id=current_user.id,
            doctor_id=doctor_id,
            date=date,
            time=time,
            status='Pending'
        )
        db.session.add(new_appointment)
        db.session.commit()
        flash('Appointment booked successfully!')
        return redirect(url_for('my_appointments'))

    return render_template('book_appointment.html', doctors=doctors)

@app.route('/my-appointments')
@login_required
def my_appointments():
    appointments = Appointment.query.filter_by(patient_id=current_user.id).all()
    doctor_names = {}
    for appt in appointments:
        doc = User.query.get(appt.doctor_id)
        doctor_names[appt.doctor_id] = doc.name if doc else "Unknown"
    return render_template('my_appointments.html', appointments=appointments, doctor_names=doctor_names)

@app.route('/api/appointments')
@login_required
def api_appointments():
    if current_user.role == 'doctor':
        appointments = Appointment.query.filter_by(doctor_id=current_user.id).all()
    else:
        appointments = Appointment.query.all()

    result = []
    for a in appointments:
        patient = User.query.get(a.patient_id)
        doctor = User.query.get(a.doctor_id)
        result.append({
            'id': a.id,
            'patient_name': patient.name if patient else 'Unknown',
            'doctor_name': doctor.name if doctor else 'Unknown',
            'date': a.date,
            'time': a.time,
            'status': a.status
        })
    return jsonify(result)

@app.route('/update-appointment/<int:appt_id>/<string:new_status>')
@login_required
def update_appointment(appt_id, new_status):
    appt = Appointment.query.get(appt_id)
    if appt and current_user.role == 'doctor' and appt.doctor_id == current_user.id:
        appt.status = new_status
        db.session.commit()
    return redirect(url_for('doctor_dashboard'))

@app.route('/api/stats')
@login_required
def api_stats():
    total_patients = User.query.filter_by(role='patient').count()
    total_doctors = User.query.filter_by(role='doctor').count()
    total_predictions = Prediction.query.count()

    all_predictions = Prediction.query.all()
    disease_list = [p.predicted_disease for p in all_predictions]
    disease_counts = dict(Counter(disease_list))

    return jsonify({
        'total_patients': total_patients,
        'total_doctors': total_doctors,
        'total_predictions': total_predictions,
        'disease_counts': disease_counts
    })

if __name__ == '__main__':
    app.run(debug=True)