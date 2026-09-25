import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import json
import seaborn as sns
import os


st.set_page_config(page_title="ARP Detection System", layout="wide")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = load_model(os.path.join(BASE_DIR, "arp_detection_model.keras"))
scaler = joblib.load("scaler.save")


with open("metrics.json") as f:
    metrics = json.load(f)

cm = np.load("confusion_matrix.npy")
selected_indices = np.load("selected_indices.npy")

st.title("🔐 ARP Spoofing Detection System")
st.markdown("XGBoost-Based ARP Spoofing Detection with Feature Selection")


tab1, tab2 = st.tabs(["🧪 Model Evaluation", "🚨 Real-Time Detection"])

with tab1:

    st.subheader("📊 Model Evaluation Summary")

    total_test_samples = int(np.sum(cm))

    TN = int(cm[0][0])
    FP = int(cm[0][1])
    FN = int(cm[1][0])
    TP = int(cm[1][1])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Test Samples", total_test_samples)
    col2.metric("True Positives (TP)", TP)
    col3.metric("True Negatives (TN)", TN)
    col4.metric("False Negatives (FN)", FN)

    col5, col6 = st.columns(2)
    col5.metric("False Positives (FP)", FP)
    col6.metric("Attack Detection Rate (%)", f"{(TP/(TP+FN))*100:.2f}")

    st.markdown("---")

    st.subheader("📈 Performance Metrics")

    colA, colB, colC, colD = st.columns(4)

    colA.metric("Accuracy", f"{metrics['accuracy']:.2f}")
    colB.metric("Precision (Attack)", f"{metrics['1']['precision']:.2f}")
    colC.metric("Recall (Attack)", f"{metrics['1']['recall']:.2f}")
    colD.metric("F1 Score (Attack)", f"{metrics['1']['f1-score']:.2f}")

    st.markdown("---")

    st.subheader("Confusion Matrix (Visual)")

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=["Pred Normal", "Pred Attack"],
        yticklabels=["True Normal", "True Attack"]
    )
    st.pyplot(fig)

    st.markdown("""
    ### 📘 Explanation:

    - **True Positive (TP):** Correctly detected ARP attacks  
    - **True Negative (TN):** Correctly identified normal traffic  
    - **False Positive (FP):** Normal traffic wrongly flagged as attack  
    - **False Negative (FN):** Attack missed by the model (critical case)

    In cybersecurity systems, minimizing **False Negatives** is very important.
    """)

with tab2:

    st.subheader("📂 Upload Dataset for Detection Simulation")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        data["Label"] = data["Label"].map({
            "normal": 0,
            "arp_spoofing": 1
        })

        data = data.select_dtypes(include=[np.number])

        X = data.drop("Label", axis=1)
        X_scaled = scaler.transform(X)

        # Same 30 features used during training
     # Use the exact 30 features selected during training
        X_selected = X_scaled[:, selected_indices]
        

        total_packets = len(X_selected)

        st.write(f"Total Packets in Dataset: {total_packets}")
        start_packet = st.slider(
            "Select Start Packet",
            0,
            total_packets - 1,
            0
            )

        end_packet = st.slider(
            "Select End Packet",
            start_packet + 1,
            total_packets,
            start_packet + 2000
            )

        if st.button("Start Detection"):

            normal_count = 0
            attack_count = 0
            attack_rows = []

            status_box = st.empty()
            progress_bar = st.progress(0)

            end_packet =min(end_packet, len(X_selected))

            for i in range(start_packet, end_packet):

                prob = float(model.predict(
                    X_selected[i].reshape(1, -1),
                    verbose=0
                )[0][0])

                prediction = 1 if prob >= 0.5 else 0

                if prediction == 1:
                    attack_count += 1
                    attack_rows.append(i)
                    status_box.error(
                        f"⚠ ARP Spoofing Detected | Row: {i} | Confidence: {prob*100:.2f}%"
                         )
                else:
                    normal_count += 1
                    status_box.success(
                        f"Normal Traffic | Row: {i} | Confidence: {(1-prob)*100:.2f}%"
                        )

                progress_bar.progress((i - start_packet + 1) / (end_packet - start_packet))

            st.subheader("📈 Detection Summary")

            colA, colB, colC = st.columns(3)

            colA.metric("Packets Analyzed", end_packet - start_packet)
            colB.metric("Normal Packets", normal_count)
            colC.metric("Attack Packets", attack_count)

            attack_percentage = (attack_count / (end_packet - start_packet)) * 100
            st.write(f"Attack Percentage: {attack_percentage:.2f}%")
            
            st.subheader("🚨 Attack Packet Numbers")
            if attack_rows:
                st.write("Packets where ARP spoofing detected:")
                st.write(attack_rows)
            else:
                st.write("No attacks detected in selected packet range.")
            fig2, ax2 = plt.subplots()
            ax2.bar(["Normal", "Attack"], [normal_count, attack_count])
            ax2.set_title("Traffic Classification Summary")
            st.pyplot(fig2)
