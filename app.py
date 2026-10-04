import os
import pickle
from collections import Counter

import numpy as np
from dotenv import load_dotenv

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
)

from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user,
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash,
)

from models import db, User, Prediction, Doctor, Appointment


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# CREATE FLASK APP
# =========================================================

app = Flask(__name__)

# FIX: Secret key for sessions, login and flash messages
app.config["SECRET_KEY"] = os.getenv(
    "SECRET_KEY",
    "medicare-secret-key-2026",
)


# =========================================================
# DATABASE CONFIGURATION
# =========================================================

db_url = os.getenv("DATABASE_URL", "sqlite:///healthcare.db")

# Convert MySQL URL to MySQL Connector URL
if db_url.startswith("mysql://"):
    db_url = db_url.replace(
        "mysql://",
        "mysql+mysqlconnector://",
        1
    )

# Remove Aiven's ssl-mode parameter from the URL
if "?" in db_url:
    base_url, query_string = db_url.split("?", 1)

    query_parts = [
        part
        for part in query_string.split("&")
        if not part.startswith("ssl-mode=")
        and not part.startswith("ssl_disabled=")
    ]

    db_url = base_url

    if query_parts:
        db_url += "?" + "&".join(query_parts)

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Aiven MySQL SSL
if db_url.startswith("mysql+mysqlconnector://"):
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "connect_args": {
            "ssl_ca": os.path.join(
                os.path.dirname(__file__),
                "aiven-ca.pem"
            ),
            "ssl_verify_cert": False,
            "ssl_verify_identity": False
        }
    }

db.init_app(app)

# =========================================================
# CREATE DATABASE TABLES
# =========================================================

with app.app_context():
    db.create_all()


# =========================================================
# LOGIN CONFIGURATION
# =========================================================

login_manager = LoginManager()
login_manager.init_app(app)

login_manager.login_view = "login"


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    return redirect(url_for("login"))


# =========================================================
# REGISTER
# =========================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        role = request.form.get("role", "patient").strip().lower()

        # Validation
        if not name or not email or not password:
            flash("Please fill all fields.")
            return redirect(url_for("register"))

        # Only patient and doctor can register publicly
        if role not in ["patient", "doctor"]:
            role = "patient"

        # Check existing email
        existing_user = User.query.filter_by(
            email=email
        ).first()

        if existing_user:
            flash("Email already registered!")
            return redirect(url_for("register"))

        # Hash password
        hashed_password = generate_password_hash(
            password
        )

        # Create user
        new_user = User(
            name=name,
            email=email,
            password=hashed_password,
            role=role
        )

        db.session.add(new_user)
        db.session.commit()

        # Create doctor profile
        if role == "doctor":

            doctor = Doctor(
                user_id=new_user.id,
                specialization="General Physician"
            )

            db.session.add(doctor)
            db.session.commit()

        flash(
            "Registration successful! Please login."
        )

        return redirect(url_for("login"))

    return render_template("register.html")


# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        user = User.query.filter_by(
            email=email
        ).first()

        if user and check_password_hash(
            user.password,
            password
        ):

            login_user(user)

            if user.role == "admin":

                return redirect(
                    url_for("admin_dashboard")
                )

            elif user.role == "doctor":

                return redirect(
                    url_for("doctor_dashboard")
                )

            else:

                return redirect(
                    url_for("patient_dashboard")
                )

        flash(
            "Invalid email or password."
        )

    return render_template("login.html")


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
@login_required
def logout():

    logout_user()

    return redirect(
        url_for("login")
    )


# =========================================================
# PATIENT DASHBOARD
# =========================================================

@app.route("/patient-dashboard")
@login_required
def patient_dashboard():

    if current_user.role != "patient":
        return redirect(
            url_for("login")
        )

    return render_template(
        "dashboard.html",
        user=current_user
    )


# =========================================================
# DOCTOR DASHBOARD
# =========================================================

@app.route("/doctor-dashboard")
@login_required
def doctor_dashboard():

    if current_user.role != "doctor":
        return redirect(
            url_for("login")
        )

    return render_template(
        "admin_dashboard.html",
        user=current_user
    )


# =========================================================
# ADMIN DASHBOARD
# =========================================================

@app.route("/admin-dashboard")
@login_required
def admin_dashboard():

    if current_user.role != "admin":
        return redirect(
            url_for("login")
        )

    return render_template(
        "admin_dashboard.html",
        user=current_user
    )


# =========================================================
# LOAD AI MODEL
# =========================================================

disease_model = None
label_encoder = None

model_path = os.path.join(
    "ml_model",
    "disease_model.pkl"
)

encoder_path = os.path.join(
    "ml_model",
    "label_encoder.pkl"
)


if os.path.exists(model_path) and os.path.exists(
    encoder_path
):

    try:

        with open(
            model_path,
            "rb"
        ) as f:

            disease_model = pickle.load(f)

        with open(
            encoder_path,
            "rb"
        ) as f:

            label_encoder = pickle.load(f)

        print(
            "AI model loaded successfully."
        )

    except Exception as e:

        print(
            "AI model loading error:",
            e
        )

else:

    print(
        "AI model files not found."
    )

    print(
        "Run: python train_simple.py"
    )


# =========================================================
# AI DISEASE PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
@login_required
def predict():

    if current_user.role != "patient":

        return redirect(
            url_for("login")
        )

    result = None

    if request.method == "POST":

        # Check model
        if disease_model is None or label_encoder is None:

            flash(
                "AI model is not ready. "
                "Run: python train_simple.py"
            )

            return render_template(
                "predict.html",
                result=None
            )

        # Get symptoms
        fever = int(
            request.form.get(
                "fever",
                0
            )
        )

        cough = int(
            request.form.get(
                "cough",
                0
            )
        )

        headache = int(
            request.form.get(
                "headache",
                0
            )
        )

        fatigue = int(
            request.form.get(
                "fatigue",
                0
            )
        )

        body_pain = int(
            request.form.get(
                "body_pain",
                0
            )
        )

        sore_throat = int(
            request.form.get(
                "sore_throat",
                0
            )
        )

        nausea = int(
            request.form.get(
                "nausea",
                0
            )
        )

        rash = int(
            request.form.get(
                "rash",
                0
            )
        )

        # Create feature array
        features = np.array([[
            fever,
            cough,
            headache,
            fatigue,
            body_pain,
            sore_throat,
            nausea,
            rash
        ]])

        # Prediction
        prediction = disease_model.predict(
            features
        )

        result = label_encoder.inverse_transform(
            prediction
        )[0]

        # Store symptoms
        symptoms_str = (
            f"fever:{fever},"
            f"cough:{cough},"
            f"headache:{headache},"
            f"fatigue:{fatigue},"
            f"body_pain:{body_pain},"
            f"sore_throat:{sore_throat},"
            f"nausea:{nausea},"
            f"rash:{rash}"
        )

        # Save prediction
        new_prediction = Prediction(
            patient_id=current_user.id,
            symptoms=symptoms_str,
            predicted_disease=result
        )

        db.session.add(
            new_prediction
        )

        db.session.commit()

    return render_template(
        "predict.html",
        result=result
    )


# =========================================================
# BOOK APPOINTMENT
# =========================================================

@app.route(
    "/book-appointment",
    methods=["GET", "POST"]
)
@login_required
def book_appointment():

    if current_user.role != "patient":

        return redirect(
            url_for("login")
        )

    doctors = User.query.filter_by(
        role="doctor"
    ).all()

    if request.method == "POST":

        doctor_id = request.form.get(
            "doctor_id"
        )

        date = request.form.get(
            "date"
        )

        time = request.form.get(
            "time"
        )

        if not doctor_id or not date or not time:

            flash(
                "Please select doctor, date and time."
            )

            return redirect(
                url_for("book_appointment")
            )

        doctor = User.query.filter_by(
            id=doctor_id,
            role="doctor"
        ).first()

        if not doctor:

            flash(
                "Doctor not found."
            )

            return redirect(
                url_for("book_appointment")
            )

        appointment = Appointment(
            patient_id=current_user.id,
            doctor_id=doctor.id,
            date=date,
            time=time,
            status="Pending"
        )

        db.session.add(
            appointment
        )

        db.session.commit()

        flash(
            "Appointment booked successfully!"
        )

        return redirect(
            url_for("my_appointments")
        )

    return render_template(
        "book_appointment.html",
        doctors=doctors
    )


# =========================================================
# MY APPOINTMENTS
# =========================================================

@app.route("/my-appointments")
@login_required
def my_appointments():

    if current_user.role != "patient":

        return redirect(
            url_for("login")
        )

    appointments = Appointment.query.filter_by(
        patient_id=current_user.id
    ).all()

    doctor_names = {}

    for appointment in appointments:

        doctor = db.session.get(
            User,
            appointment.doctor_id
        )

        doctor_names[
            appointment.doctor_id
        ] = (
            doctor.name
            if doctor
            else "Unknown"
        )

    return render_template(
        "my_appointments.html",
        appointments=appointments,
        doctor_names=doctor_names
    )


# =========================================================
# APPOINTMENTS API
# =========================================================

@app.route("/api/appointments")
@login_required
def api_appointments():

    if current_user.role == "doctor":

        appointments = Appointment.query.filter_by(
            doctor_id=current_user.id
        ).all()

    elif current_user.role == "admin":

        appointments = Appointment.query.all()

    else:

        appointments = Appointment.query.filter_by(
            patient_id=current_user.id
        ).all()

    result = []

    for appointment in appointments:

        patient = db.session.get(
            User,
            appointment.patient_id
        )

        doctor = db.session.get(
            User,
            appointment.doctor_id
        )

        result.append({

            "id": appointment.id,

            "patient_name":
                patient.name
                if patient
                else "Unknown",

            "doctor_name":
                doctor.name
                if doctor
                else "Unknown",

            "date": appointment.date,

            "time": appointment.time,

            "status": appointment.status

        })

    return jsonify(result)


# =========================================================
# UPDATE APPOINTMENT STATUS
# =========================================================

@app.route(
    "/update-appointment/<int:appt_id>/<string:new_status>"
)
@login_required
def update_appointment(
    appt_id,
    new_status
):

    if current_user.role != "doctor":

        return redirect(
            url_for("login")
        )

    # Only allowed statuses
    if new_status not in [
        "Confirmed",
        "Rejected"
    ]:

        flash(
            "Invalid appointment status."
        )

        return redirect(
            url_for("doctor_dashboard")
        )

    appointment = db.session.get(
        Appointment,
        appt_id
    )

    if not appointment:

        flash(
            "Appointment not found."
        )

        return redirect(
            url_for("doctor_dashboard")
        )

    # Doctor can update only own appointment
    if appointment.doctor_id != current_user.id:

        flash(
            "You are not allowed to update this appointment."
        )

        return redirect(
            url_for("doctor_dashboard")
        )

    appointment.status = new_status

    db.session.commit()

    flash(
        f"Appointment {new_status.lower()}."
    )

    return redirect(
        url_for("doctor_dashboard")
    )


# =========================================================
# ADMIN STATISTICS API
# =========================================================

@app.route("/api/stats")
@login_required
def api_stats():

    if current_user.role != "admin":

        return jsonify({
            "error": "Unauthorized"
        }), 403

    total_patients = User.query.filter_by(
        role="patient"
    ).count()

    total_doctors = User.query.filter_by(
        role="doctor"
    ).count()

    total_predictions = Prediction.query.count()

    all_predictions = Prediction.query.all()

    disease_list = [
        prediction.predicted_disease
        for prediction in all_predictions
    ]

    disease_counts = dict(
        Counter(disease_list)
    )

    return jsonify({

        "total_patients":
            total_patients,

        "total_doctors":
            total_doctors,

        "total_predictions":
            total_predictions,

        "disease_counts":
            disease_counts

    })


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )