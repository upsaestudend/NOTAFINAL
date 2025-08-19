import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ------------------------------
# 🎯 Configuración inicial
# ------------------------------
st.set_page_config(page_title="Predicción de Notas", layout="centered")
st.title("📚 Predicción y Cálculo de la Nota Final del Estudiante")

# ------------------------------
# 📂 Cargar modelo y dataset
# ------------------------------
modelo_path = "modelo_entrenado.pkl"
data_path = "calificaciones_1000_estudiantes_con_id.csv"

if not os.path.exists(modelo_path) or not os.path.exists(data_path):
    st.error("❌ No se encontró el modelo entrenado o el dataset.")
    st.stop()

modelo = joblib.load(modelo_path)
df = pd.read_csv(data_path)

# ------------------------------
# 🧮 Reglas de negocio sobre dataset
# ------------------------------
df["Bono"] = np.where(df["Asistencia"] > 95, df["TP"] * 0.20, 0)
df["TP_Modificado"] = df["TP"] + df["Bono"]
df["Final_Usado"] = np.where(df["Asistencia"] < 80, 0, df["Examen_Final"])
df["Nota_Final_Calculada"] = (
    0.1333 * df["Parcial_1"] +
    0.1333 * df["Parcial_2"] +
    0.1333 * df["Parcial_3"] +
    0.20 * df["TP_Modificado"] +
    0.40 * df["Final_Usado"]
).round(1)

# ------------------------------
# 🏅 Clasificación de notas
# ------------------------------
def clasificar(nota):
    if nota >= 91: return "Excelente"
    elif nota >= 81: return "Óptimo"
    elif nota >= 71: return "Satisfactorio"
    elif nota >= 61: return "Bueno"
    elif nota >= 51: return "Regular"
    else: return "Insuficiente"

df["Clasificacion"] = df["Nota_Final_Calculada"].apply(clasificar)

# ------------------------------
# ✍️ Formulario de entrada
# ------------------------------
st.sidebar.header("✍️ Datos del estudiante")
p1 = st.sidebar.slider("Parcial 1", 0.0, 100.0, 70.0)
p2 = st.sidebar.slider("Parcial 2", 0.0, 100.0, 70.0)
p3 = st.sidebar.slider("Parcial 3", 0.0, 100.0, 70.0)
tp = st.sidebar.slider("Trabajos Prácticos", 0.0, 100.0, 75.0)
final = st.sidebar.slider("Examen Final", 0.0, 100.0, 70.0)
asistencia = st.sidebar.slider("Asistencia (%)", 0.0, 100.0, 85.0)

# ------------------------------
# 🔮 Predicción del modelo
# ------------------------------
X_nuevo = pd.DataFrame({
    "Parcial_1": [p1],
    "Parcial_2": [p2],
    "Parcial_3": [p3],
    "Asistencia": [asistencia]
})

nota_pred_modelo = modelo.predict(X_nuevo)[0]
nota_pred_modelo = max(0, min(100, nota_pred_modelo))
clas_pred_modelo = clasificar(nota_pred_modelo)

# ------------------------------
# 📏 Cálculo con reglas exactas
# ------------------------------
bono = tp * 0.20 if asistencia > 95 else 0
tp_modificado = tp + bono
final_usable = 0 if asistencia < 80 else final

nota_reglas = (
    0.1333 * p1 +
    0.1333 * p2 +
    0.1333 * p3 +
    0.20 * tp_modificado +
    0.40 * final_usable
)
clas_reglas = clasificar(nota_reglas)

# ------------------------------
# 📊 Resultados comparativos
# ------------------------------
st.subheader("📈 Resultados del estudiante")
col1, col2 = st.columns(2)

with col1:
    st.write("🔮 **Predicción del modelo**")
    st.metric("Nota estimada", f"{nota_pred_modelo:.1f}")
    st.metric("Clasificación", clas_pred_modelo)

with col2:
    st.write("📏 **Cálculo con reglas**")
    st.metric("Nota calculada", f"{nota_reglas:.1f}")
    st.metric("Clasificación", clas_reglas)

# Comparación gráfica
st.subheader("📊 Comparación gráfica")
fig, ax = plt.subplots()
ax.bar(["Modelo", "Reglas"], [nota_pred_modelo, nota_reglas], color=["blue", "orange"])
ax.set_ylabel("Nota final")
st.pyplot(fig)

# ------------------------------
# 📊 Estadísticas del dataset
# ------------------------------
st.subheader("📊 Estadísticas del dataset")

col1, col2 = st.columns(2)
with col1:
    clas_counts = df["Clasificacion"].value_counts().reindex(
        ["Excelente","Óptimo","Satisfactorio","Bueno","Regular","Insuficiente"]
    ).fillna(0)
    fig1, ax1 = plt.subplots()
    clas_counts.plot(kind="bar", ax=ax1, color="skyblue")
    ax1.set_title("Distribución de Clasificaciones")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sns.histplot(df["Nota_Final_Calculada"], bins=20, kde=True, ax=ax2, color="orange")
    ax2.set_title("Histograma de Notas Finales")
    st.pyplot(fig2)

# ------------------------------
# 📌 Matriz de confusión
# ------------------------------
st.subheader("📌 Matriz de Confusión del Modelo")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

X = df[["Parcial_1","Parcial_2","Parcial_3","Asistencia"]]
y_real = df["Nota_Final_Calculada"]
y_pred = modelo.predict(X)

y_real_clas = y_real.apply(clasificar)
y_pred_clas = pd.Series(y_pred).apply(clasificar)

labels = ["Excelente","Óptimo","Satisfactorio","Bueno","Regular","Insuficiente"]
cm = confusion_matrix(y_real_clas, y_pred_clas, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

fig3, ax3 = plt.subplots(figsize=(8,6))
disp.plot(ax=ax3, cmap="Blues", xticks_rotation=45)
st.pyplot(fig3)
