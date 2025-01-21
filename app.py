import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle

# ---- Configuración General ----
st.set_page_config(
    page_title="Análisis de Finanzas Personales en India",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Introducción ----
st.markdown("""
<h1 style="text-align: center; color: #4CAF50;">Análisis de Finanzas Personales en India</h1>
<p style="text-align: center;">
Este análisis interactivo explora los ingresos, gastos y patrones financieros de la población en India, 
y utiliza Machine Learning para predecir el ingreso disponible de las personas.
</p>
""", unsafe_allow_html=True)

#st.image("cover.png", use_column_width=True)  # Imagen opcional como portada

# ---- Cargar Datos ----
@st.cache
def load_data():
    return pd.read_csv('datos_limpios.csv')

data = load_data()

# ---- Navegación ----
menu = st.sidebar.selectbox(
    "Navegación",
    ["🏠 Introducción", "📊 Análisis Visual", "🤖 Predicción Financiera", "🔍 Conclusiones y Recomendaciones"]
)

# ---- Introducción (Opcional) ----
if menu == "🏠 Introducción":
    st.markdown("""
    ### ¿Qué encontrarás en este análisis?
    """)
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📊 Análisis de Ingresos")
        st.write("""
        - Distribución de ingresos según edad, ocupación y ubicación.
        - Visualización de desigualdades en ingresos.
        """)
        st.progress(75)

    with col2:
        st.subheader("💸 Análisis de Gastos")
        st.write("""
        - Identificación de los principales tipos de gastos.
        - Comparativa entre ingresos y gastos para patrones de ahorro.
        """)

    with col3:
        st.subheader("🤖 Predicción Financiera")
        st.write("""
        - Predicción del ingreso disponible basado en características clave.
        - Uso de un modelo de Machine Learning para generar insights.
        """)

# ---- Análisis Visual ----
elif menu == "📊 Análisis Visual":
    st.markdown("## Análisis Visual de los Datos")
    st.write("Explora las relaciones entre ingresos, gastos y otras variables clave.")

    # Opciones de visualización
    st.sidebar.subheader("Opciones de Visualización")
    x_axis = st.sidebar.selectbox("Eje X:", data.columns)
    y_axis = st.sidebar.selectbox("Eje Y:", data.columns)
    chart_type = st.sidebar.radio("Tipo de gráfico:", ["Scatterplot", "Boxplot", "Histogram"])

    # Crear gráficos
    if chart_type == "Scatterplot":
        st.write(f"### Gráfico de dispersión: {x_axis} vs {y_axis}")
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x=x_axis, y=y_axis, ax=ax)
        st.pyplot(fig)
    elif chart_type == "Boxplot":
        st.write(f"### Diagrama de caja: {x_axis} vs {y_axis}")
        fig, ax = plt.subplots()
        sns.boxplot(data=data, x=x_axis, y=y_axis, ax=ax)
        st.pyplot(fig)
    elif chart_type == "Histogram":
        st.write(f"### Histograma de {x_axis}")
        fig, ax = plt.subplots()
        sns.histplot(data[x_axis], kde=True, ax=ax)
        st.pyplot(fig)

# ---- Predicción Financiera ----
elif menu == "🤖 Predicción Financiera":
    st.markdown("## Predicción del Ingreso Disponible")
    st.write("Introduce las características para predecir el ingreso disponible.")

    # Inputs del usuario
    age = st.number_input("Edad:", min_value=18, max_value=70, value=30)
    occupation = st.selectbox("Ocupación:", data['Occupation'].unique())
    city_tier = st.selectbox("City Tier:", data['City_Tier'].unique())
    income = st.number_input("Ingreso Mensual (₹):", min_value=0, max_value=500000, value=50000)

    # Crear DataFrame con datos del usuario
    user_data = pd.DataFrame({
        "Age": [age],
        "Occupation": [occupation],
        "City_Tier": [city_tier],
        "Income": [income]
    })

    st.write("Datos del Usuario:")
    st.dataframe(user_data)

    # Cargar modelo de predicción
    @st.cache
    def load_model():
        with open('models/disposable_income_model.pkl', 'rb') as file:
            return pickle.load(file)

    model = load_model()

    # Predicción
    if st.button("Predecir"):
        prediction = model.predict(user_data)
        st.success(f"El ingreso disponible estimado es: ₹{prediction[0]:,.2f}")

# ---- Conclusiones y Recomendaciones ----
elif menu == "🔍 Conclusiones y Recomendaciones":
    st.markdown("## Conclusiones y Recomendaciones")
    st.write("""
    ### Conclusiones:
    1. Los ingresos varían significativamente según la ocupación y el City Tier.
    2. Los gastos en transporte y entretenimiento representan las principales áreas de mejora.
    3. Una planificación financiera adecuada puede aumentar los ahorros en un 20%.

    ### Recomendaciones:
    - Fomentar la educación financiera para maximizar el ingreso disponible.
    - Implementar herramientas de ahorro automatizado para la población urbana.
    - Promover alternativas de transporte más económicas y sostenibles.
    """)

    # Descarga de resultados
    st.markdown("### Descarga de Datos:")
    @st.cache
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(data)
    st.download_button("Descargar Datos", data=csv, file_name="finanzas_india.csv", mime="text/csv")
