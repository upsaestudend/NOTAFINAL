import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------
# 🎯 Configuración inicial
# ------------------------------
st.set_page_config(page_title="Cálculo de Notas", layout="centered")
st.title("📚 Cálculo de la Nota Final del Estudiante (Reglas de Negocio)")

# ------------------------------
# 📂 Cargar dataset histórico
# ------------------------------
data_path = "calificaciones_1000_estudiantes_con_id.csv"

if not os.path.exists(data_path):
    st.error("❌ No se encontró el dataset.")
    st.stop()

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
asistencia = st.sidebar.slider("Asistencia (%)", 0.0, 100.0, 85.0)

# ------------------------------
# 📏 Cálculo con reglas exactas
# ------------------------------
bono = tp * 0.20 if asistencia > 95 else 0
tp_modificado = tp + bono
final_usable = np.mean(df["Examen_Final"]) if asistencia >= 80 else 0  # promedio histórico

nota_final = (
    0.1333 * p1 +
    0.1333 * p2 +
    0.1333 * p3 +
    0.20 * tp_modificado +
    0.40 * final_usable
)
clas_final = clasificar(nota_final)

# ------------------------------
# 📊 Resultado
# ------------------------------
st.subheader("📈 Resultado del estudiante")
st.metric("Nota final calculada", f"{nota_final:.1f}")
st.metric("Clasificación", clas_final)

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
# 📦 Boxplot de notas por clasificación
# ------------------------------
st.subheader("📦 Distribución de notas por clasificación")
fig3, ax3 = plt.subplots()
sns.boxplot(x="Clasificacion", y="Nota_Final_Calculada", data=df, palette="Set2", ax=ax3)
ax3.set_title("Boxplot de Notas por Clasificación")
st.pyplot(fig3)

# ------------------------------
# 📌 Relación Asistencia vs Nota Final
# ------------------------------
st.subheader("📌 Relación Asistencia vs Nota Final")
fig4, ax4 = plt.subplots()
sns.scatterplot(x="Asistencia", y="Nota_Final_Calculada", hue="Clasificacion", data=df, palette="Set1", ax=ax4)
ax4.set_title("Asistencia vs Nota Final")
st.pyplot(fig4)

# ------------------------------
# 📊 Promedio de notas por parcial
# ------------------------------
st.subheader("📊 Promedio por Parcial")
parciales_mean = df[["Parcial_1", "Parcial_2", "Parcial_3"]].mean()
fig5, ax5 = plt.subplots()
parciales_mean.plot(kind="bar", ax=ax5, color=["#1f77b4","#ff7f0e","#2ca02c"])
ax5.set_title("Promedio de cada Parcial")
st.pyplot(fig5)