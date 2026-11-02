import os
import pickle
import re

import numpy as np
import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)
base_path = os.path.dirname(__file__)
dataset_path = os.path.join(base_path, "dataset")
MINIMUM_SYMPTOMS = 2
MAX_FOLLOW_UP_QUESTIONS = 2
NON_SYMPTOM_FOLLOW_UPS = {
    "extra_marital_contacts",
    "family_history",
    "history_of_alcohol_consumption",
    "receiving_blood_transfusion",
    "receiving_unsterile_injections",
}


class SymptomInputError(ValueError):
    pass


try:
    training_data = pd.read_csv(os.path.join(dataset_path, "Training.csv"))
    sym_des = pd.read_csv(os.path.join(dataset_path, "symtoms_df.csv"))
    precautions_df = pd.read_csv(os.path.join(dataset_path, "precautions_df.csv"))
    workout_df = pd.read_csv(os.path.join(dataset_path, "workout_df.csv"))
    description_df = pd.read_csv(os.path.join(dataset_path, "description.csv"))
    medications_df = pd.read_csv(os.path.join(dataset_path, "medications.csv"))
    diets_df = pd.read_csv(os.path.join(dataset_path, "diets.csv"))
except FileNotFoundError as e:
    raise FileNotFoundError(
        f"Dataset file not found: {e.filename}. "
        "Please make sure all required CSV files exist in the dataset folder."
    ) from e

if "prognosis" not in training_data.columns:
    raise ValueError("Training.csv must contain a 'prognosis' column.")

symptom_columns = list(training_data.columns.drop("prognosis"))
if not symptom_columns:
    raise ValueError("Training.csv must contain symptom feature columns.")

try:
    with open(os.path.join(base_path, "model", "svc.pkl"), "rb") as model_file:
        svc_model = pickle.load(model_file)
except FileNotFoundError as e:
    raise FileNotFoundError(
        "Model file 'svc.pkl' not found in the 'model' directory."
    ) from e

if svc_model.n_features_in_ != len(symptom_columns):
    raise ValueError(
        "The saved model feature count does not match the Training.csv columns."
    )

disease_labels = sorted(training_data["prognosis"].dropna().unique())
model_classes = list(svc_model.classes_)
if len(model_classes) != len(disease_labels):
    raise ValueError(
        "The saved model classes do not match the disease labels in Training.csv."
    )
if all(label in disease_labels for label in model_classes):
    diseases_list = {model_class: model_class for model_class in model_classes}
elif np.array_equal(np.asarray(model_classes), np.arange(len(disease_labels))):
    diseases_list = dict(zip(model_classes, disease_labels))
else:
    raise ValueError(
        "The saved model classes cannot be aligned with Training.csv disease labels."
    )


def normalize_symptom_name(symptom):
    normalized = re.sub(r"[\s-]+", "_", str(symptom).strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")


symptoms_dict = {}
for symptom in symptom_columns:
    normalized = normalize_symptom_name(symptom)
    if normalized in symptoms_dict:
        raise ValueError(
            f"Ambiguous symptom name in Training.csv: {symptom!r}."
        )
    symptoms_dict[normalized] = symptom

follow_up_data = training_data.drop_duplicates(
    subset=symptom_columns + ["prognosis"]
)


def get_disease_info(disease):
    description = " ".join(
        description_df[description_df["Disease"] == disease]["Description"]
        .dropna()
        .astype(str)
    )
    precautions = precautions_df[precautions_df["Disease"] == disease]
    precautions_list = (
        precautions.values.flatten().tolist()[1:] if not precautions.empty else []
    )
    medications = medications_df[
        medications_df["Disease"] == disease
    ]["Medication"].dropna().tolist()
    diets = diets_df[diets_df["Disease"] == disease]["Diet"].dropna().tolist()
    workout = workout_df[workout_df["disease"] == disease]["workout"].dropna().tolist()
    return description, precautions_list, medications, diets, workout


def resolve_symptoms(symptom_list):
    selected_symptoms = []
    unrecognized_symptoms = []

    for symptom in symptom_list:
        normalized = normalize_symptom_name(symptom)
        if not normalized:
            continue
        matching_symptom = symptoms_dict.get(normalized)
        if matching_symptom is None:
            if symptom.strip() not in unrecognized_symptoms:
                unrecognized_symptoms.append(symptom.strip())
        elif matching_symptom not in selected_symptoms:
            selected_symptoms.append(matching_symptom)

    if unrecognized_symptoms:
        message = (
            f"Unrecognized symptom(s): {', '.join(unrecognized_symptoms)}. "
            "Use symptom names from the dataset."
        )
        if any(normalize_symptom_name(item) == "fever" for item in unrecognized_symptoms):
            message += " Specify 'high fever' or 'mild fever' instead of 'fever'."
        raise SymptomInputError(message)

    return selected_symptoms


def get_next_follow_up(selected_symptoms, negative_symptoms, asked_symptoms):
    candidates = follow_up_data
    if selected_symptoms:
        candidates = candidates[
            candidates[selected_symptoms].eq(1).all(axis=1)
        ]
    if negative_symptoms:
        candidates = candidates[
            candidates[negative_symptoms].eq(0).all(axis=1)
        ]
    if candidates["prognosis"].nunique() < 2:
        return None

    labels = candidates["prognosis"]
    label_probabilities = labels.value_counts(normalize=True).to_numpy()
    disease_entropy = -np.sum(
        label_probabilities * np.log2(label_probabilities)
    )

    best_symptom = None
    best_information_gain = 0.0
    unavailable_symptoms = set(selected_symptoms + negative_symptoms + asked_symptoms)
    for symptom in symptom_columns:
        if symptom in unavailable_symptoms or symptom in NON_SYMPTOM_FOLLOW_UPS:
            continue
        answer_groups = candidates.groupby(symptom, sort=False)
        if answer_groups.ngroups < 2:
            continue
        remaining_entropy = sum(
            len(answer_group) / len(candidates)
            * -np.sum(
                answer_group["prognosis"].value_counts(normalize=True).to_numpy()
                * np.log2(
                    answer_group["prognosis"]
                    .value_counts(normalize=True)
                    .to_numpy()
                )
            )
            for _, answer_group in answer_groups
        )
        information_gain = disease_entropy - remaining_entropy
        if information_gain > best_information_gain:
            best_information_gain = information_gain
            best_symptom = symptom

    return best_symptom


def predict_disease(selected_symptoms):
    if len(selected_symptoms) < MINIMUM_SYMPTOMS:
        raise SymptomInputError(
            f"Please enter at least {MINIMUM_SYMPTOMS} recognized symptoms "
            "so the app does not force a prediction from too little information."
        )

    input_vector = [
        int(symptom in selected_symptoms) for symptom in symptom_columns
    ]
    input_frame = pd.DataFrame([input_vector], columns=symptom_columns)
    prediction_index = svc_model.predict(input_frame)[0]
    return diseases_list.get(prediction_index, "Unknown")


def render_follow_up(selected_symptoms, negative_symptoms, asked_symptoms):
    symptom = get_next_follow_up(
        selected_symptoms,
        negative_symptoms,
        asked_symptoms,
    )
    if symptom is None or len(asked_symptoms) >= MAX_FOLLOW_UP_QUESTIONS:
        return render_prediction_result(selected_symptoms)

    return render_template(
        "index.html",
        follow_up_symptom=symptom,
        follow_up_question=f"Do you also have {symptom.replace('_', ' ')}?",
        follow_up_number=len(asked_symptoms) + 1,
        selected_symptoms=selected_symptoms,
        negative_symptoms=negative_symptoms,
        asked_symptoms=asked_symptoms + [symptom],
    )


def render_prediction_result(selected_symptoms):
    predicted_disease = predict_disease(selected_symptoms)
    if predicted_disease == "Unknown":
        return render_template(
            "index.html",
            message="Disease could not be predicted. Check symptoms.",
        )

    desc, precautions, meds, diets, workouts = get_disease_info(predicted_disease)
    return render_template(
        "index.html",
        predicted_disease=predicted_disease,
        dis_des=desc,
        my_precautions=precautions,
        medications=meds,
        my_diet=diets,
        workout=workouts,
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    symptoms_input = request.form.get("symptoms", "")
    if not symptoms_input or symptoms_input.strip().lower() == "symptoms":
        return render_template(
            "index.html",
            message="Please enter valid symptoms, separated by commas.",
        )

    user_symptoms = [symptom.strip() for symptom in symptoms_input.split(",")]
    try:
        selected_symptoms = resolve_symptoms(user_symptoms)
        if len(selected_symptoms) < MINIMUM_SYMPTOMS:
            raise SymptomInputError(
                f"Please enter at least {MINIMUM_SYMPTOMS} recognized symptoms "
                "before continuing."
            )
    except SymptomInputError as e:
        return render_template("index.html", message=str(e))

    return render_follow_up(selected_symptoms, [], [])


@app.route("/follow-up", methods=["POST"])
def follow_up():
    try:
        selected_symptoms = resolve_symptoms(
            request.form.getlist("positive_symptoms")
        )
        negative_symptoms = resolve_symptoms(
            request.form.getlist("negative_symptoms")
        )
        asked_symptoms = resolve_symptoms(request.form.getlist("asked_symptoms"))
    except SymptomInputError as e:
        return render_template("index.html", message=str(e)), 400

    current_question = request.form.get("follow_up_symptom", "")
    if (
        len(selected_symptoms) < MINIMUM_SYMPTOMS
        or not asked_symptoms
        or asked_symptoms[-1] != current_question
        or len(asked_symptoms) > MAX_FOLLOW_UP_QUESTIONS
        or set(selected_symptoms) & set(negative_symptoms)
    ):
        return render_template(
            "index.html",
            message="That follow-up question is no longer valid. Please start again.",
        ), 400

    expected_question = get_next_follow_up(
        selected_symptoms,
        negative_symptoms,
        asked_symptoms[:-1],
    )
    if expected_question != current_question:
        return render_template(
            "index.html",
            message="That follow-up question is no longer valid. Please start again.",
        ), 400

    answer = request.form.get("answer")
    if answer not in {"yes", "no", "unsure"}:
        return render_template(
            "index.html",
            message="Please choose yes, no, or not sure.",
        ), 400

    if answer == "yes":
        selected_symptoms.append(current_question)
    elif answer == "no":
        negative_symptoms.append(current_question)

    return render_follow_up(
        selected_symptoms,
        negative_symptoms,
        asked_symptoms,
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/blog")
def blog():
    return render_template("blog.html")


if __name__ == "__main__":
    app.run(debug=True)
