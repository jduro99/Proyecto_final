import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import joblib
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from prediccion import predecir_finanzas

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType
import warnings

warnings.filterwarnings('ignore')

#CONFIGURACION DE LA PAGINA
st.set_page_config(
    #page_title="Análisis de Finanzas Personales en India",
    page_icon="🇮🇳",
    layout="wide", 
    initial_sidebar_state="expanded",
)

if "current_page" not in st.session_state:
    st.session_state.current_page = 0

pages = ["🏠 Introducción", "📊 Análisis Visual","📈 Dashboard Power BI", "🤖 Predicción Financiera", "🔍 Conclusiones y Recomendaciones"]

# Funciones para manejar los botones de navegación
def next_page():
    if st.session_state.current_page < len(pages) - 1:
        st.session_state.current_page += 1

def previous_page():
    if st.session_state.current_page > 0:
        st.session_state.current_page -= 1

st.markdown(
    """
    <style>
    /* Fondo oscuro para toda la aplicación */
    .stApp {
        background-color: #1e1e2f !important;  /* Fondo oscuro para toda la página */
        color: #ffffff !important;  /* Texto blanco */
    }

    /* Fondo y texto de la barra lateral */
    .css-1d391kg, .css-1y4p8pa, .css-qbe2hs, .stSidebar {
        background-color: #1e1e2f !important;  /* Fondo oscuro igual al resto */
        color: #ffffff !important;  /* Texto blanco */
    }

    /* Texto de títulos y subtítulos */
    .css-h3b2pw, h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;  /* Encabezados en blanco */
    }

    /* Fondo oscuro y texto blanco en selectores desplegables */
    .css-1l02zno, .stSelectbox, .stRadio, .stMultiselect {
        background-color: #29293d !important;  /* Fondo oscuro */
        color: #ffffff !important;  /* Texto blanco */
    }

    /* Bordes en los filtros desplegables */
    .css-1d391kg select {
        background-color: #29293d !important; /* Fondo oscuro */
        color: white !important; /* Texto blanco */
        border-color: #ffffff !important; /* Bordes blancos */
    }

    /* Fondo oscuro en los campos de texto */
    .stTextInput>div>input, .stTextArea>div>textarea {
        background-color: #29293d !important; /* Fondo oscuro */
        color: #ffffff !important;  /* Texto blanco */
    }

    /* Fondo y texto de los botones */
    .stButton>button {
        background-color: #ff4b4b !important; /* Fondo rojo */
        color: #ffffff !important; /* Texto blanco */
        border: 1px solid #ffffff !important; /* Borde blanco */
    }
    </style>
    """,
    unsafe_allow_html=True
)

#FUNCION PARA CARGAR EL dfSET
df= pd.read_csv('datos_limpios.csv')
data=pd.read_csv('data/data.csv')
datos=pd.read_csv('datos_modelo.csv') 

st.sidebar.image('data/cover.png')
st.sidebar.title("Navegación")
selected_page = st.sidebar.selectbox("Selecciona una página:", pages, index=st.session_state.current_page)

# Sincronizar el selector con la página actual
st.session_state.current_page = pages.index(selected_page)

menu = pages[st.session_state.current_page]

# ---- Pestaña de Introducción Mejorada ----
if menu == "🏠 Introducción":
    st.image("data/india2.png",width=1000)
    st.markdown("""
    <h1 style="text-align: center; color: #4CAF50;">Análisis de Finanzas Personales en India</h1>
    <p style="text-align: justify; font-size: 18px;">
    Bienvenidos a esta exploración interactiva de las <b>finanzas personales en India</b>. Este proyecto está diseñado para analizar los ingresos, 
    los gastos y los patrones de ahorro de las personas, destacando insights clave que pueden ser utilizados para tomar decisiones financieras más 
    informadas. Además, presentamos un modelo predictivo para estimar el ingreso disponible basado en variables específicas.
    </p>
    """, unsafe_allow_html=True)

    # Puntos Destacados
    st.markdown("""
    ### ¿Qué encontrarás en este análisis?
    """)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📊 Análisis de Ingresos")
        st.write("""
        - Exploración de cómo varían los ingresos según la edad, ocupación y ubicación.
        - Identificación de desigualdades y patrones en los ingresos.
        """)
        st.progress(75)

    with col2:
        st.subheader("💸 Análisis de Gastos")
        st.write("""
        - Evaluación de las principales categorías de gasto.
        - Comparativa entre ingresos y gastos para detectar oportunidades de ahorro.
        """)
        st.progress(85)

    with col3:
        st.subheader("🤖 Predicción Financiera")
        st.write("""
        - Modelo predictivo para calcular el ingreso disponible.
        - Insights clave generados a partir de Machine Learning.
        """)
        st.progress(90)

    st.markdown("""
                ---
                """)

    # 🔹 Contexto en Sección Colapsable (Para no saturar la introducción)
    with st.expander("📌 Información sobre India y su economía"):
        st.markdown("""
        ---
        India es el segundo país más poblado del mundo, con más de **1,400 millones de habitantes**. Su economía es una de las economías con más rápido crecimiento, 
        aunque presenta desigualdades significativas entre zonas rurales y urbanas.  
        
        **📊 Datos Clave:**  
        - **PIB (2023):** $3.7 billones USD (5° lugar mundial).  
        - **Ahorro promedio:** 30% del ingreso anual.  
        - **Ciudades principales:** Mumbai, Delhi, Bangalore, Chennai.  
        """)

    # Información adicional y contexto
    st.markdown("""
        ---
    ### Estructura de la Aplicación:
    - **📊 Análisis Visual:** Representación interactiva de los datos financieros clave.
    - **📈 Dashboard Power BI:** Visualizaciones avanzadas creadas en Power BI.
    - **🤖 Predicción Financiera:** Modelo predictivo para estimar el ingreso disponible.
    - **🔍 Conclusiones y Recomendaciones:** Resumen de insights clave y propuestas de acción.
    """, unsafe_allow_html=True)

# ---- Análisis Visual ----
elif menu == "📊 Análisis Visual":
    st.title("Análisis Visual de los Datos")
    st.write("Explora las relaciones entre ingresos, gastos y otras variables clave.")

# ----------------------- Promedio de Ingresos por Edad -----------------------
    income_by_age = data.groupby('Age')['Income'].mean()
    overall_mean_income = data['Income'].mean()

    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(
        x=income_by_age.index,
        y=income_by_age.values,
        mode='lines+markers',
        name='Ingreso promedio por edad',
        line=dict(color='green'),
        marker=dict(size=8)
    ))

    fig1.add_trace(go.Scatter(
        x=income_by_age.index,
        y=[overall_mean_income] * len(income_by_age),
        mode='lines',
        name='Ingreso promedio general',
        line=dict(color='orange', dash='dash')
    ))

    fig1.update_layout(
        title='Promedio de Ingresos por Edad',
        xaxis_title='Edad',
        yaxis_title='Ingreso Promedio',
        template='plotly_white',
        width=800,
        height=500,
        font=dict(size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

# ----------------------- Promedio de Ahorro Potencial por Categoría -----------------------
    categories = ['Groceries', 'Transport', 'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education', 'Miscellaneous']
    for category in categories:
        data[f'{category}_savings'] = data['Disposable_Income'] - data[category]

    category_savings = {category: data[f'{category}_savings'].mean() for category in categories}
    category_savings_datos = pd.DataFrame(list(category_savings.items()), columns=['Categoría', 'Ahorro Potencial Promedio'])

    fig2 = px.bar(
        category_savings_datos.sort_values('Ahorro Potencial Promedio', ascending=False),
        x='Categoría',
        y='Ahorro Potencial Promedio',
        color='Categoría',
        text='Ahorro Potencial Promedio',
        color_discrete_sequence=px.colors.qualitative.Dark2,
        title='Promedio de Ahorro Potencial por Categoría'
    )

    fig2.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside'
    )
    fig2.update_layout(
        xaxis_title='Categoría de Gasto',
        yaxis_title='Ahorro Potencial Promedio',
        template='plotly_dark',
        width=800,
        height=500,
        font=dict(size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

# ----------------------- Porcentaje Promedio de Gasto por Categoría -----------------------
    for category in categories:
        data[f'{category}_percentage'] = ((data[category] / data['Income']) * 100).round(2)

    mean_percentages = data[[f'{category}_percentage' for category in categories]].mean().round(2)
    df_plot = pd.DataFrame({
        'Categoría de Gasto': categories,
        'Porcentaje Promedio': mean_percentages.values
    })


    fig3 = px.bar(
        df_plot.sort_values('Porcentaje Promedio', ascending=False),
        x='Categoría de Gasto',
        y='Porcentaje Promedio',
        color='Categoría de Gasto',
        color_discrete_sequence=px.colors.qualitative.Dark2,
        title='Porcentaje Promedio de Gasto por Categoría sobre el Ingreso Total',
    )

    fig3.update_layout(
        xaxis_title='Categoría de Gasto',
        yaxis_title='Porcentaje sobre el Ingreso',
        template='plotly_dark',
        width=800,
        height=500,
        font=dict(size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

# ----------------------- Gastos Fijos y Variables por City_Tier y Occupation --------------------------------

    gastos_por_city_tier_occupation = datos.groupby(['City_Tier', 'Occupation'])[['Gasto_Fijo', 'Gastos_variables']].sum().reset_index().sort_values('Gasto_Fijo', ascending=False) 
    labels = gastos_por_city_tier_occupation['City_Tier'] + " - " + gastos_por_city_tier_occupation['Occupation']

    fig4 = go.Figure()

    fig4.add_trace(go.Bar(
        x=labels,
        y=gastos_por_city_tier_occupation['Gasto_Fijo'],
        name='Gasto Fijo',
        marker_color='#6BA368'
    ))

    #fig4.add_trace(go.Bar(
    fig4.add_trace(go.Bar(
        x=labels,
        y=gastos_por_city_tier_occupation['Gastos_variables'],
        name='Gasto Variable',
        marker_color='#A3D9A5'  # Verde aún más claro
    ))

    fig4.update_layout(
        barmode='stack',
        title='Gastos Fijos y Variables por City_Tier y Occupation',
        xaxis_title='City_Tier - Occupation',
        yaxis_title='Total Gastos',
        template='plotly_dark',
        width=800,
        height=500,
        font=dict(size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

# ----------------------- Gastos Fijos y Variables por City_Tier y Occupation --------------------------------

# Crear el gráfico de dispersión
    fig5 = go.Figure()

    # Agregar puntos para Gastos Fijos
    fig5.add_trace(go.Scatter(
        x=datos['Income'],
        y=datos['Gasto_Fijo'],
        mode='markers',
        name='Gasto Fijo',
        marker=dict(
            color='#87CEEB',  # Azul claro
            size=8,
            line=dict(width=1, color='#4682B4')  # Bordes azul más oscuro
        )
    ))

    # Agregar puntos para Gastos Variables
    fig5.add_trace(go.Scatter(
        x=datos['Income'],
        y=datos['Gastos_variables'],
        mode='markers',
        name='Gasto Variable',
        marker=dict(
            color='#FFDAB9',  # Melocotón claro
            size=8,
            line=dict(width=1, color='#FF8C00')  # Bordes naranja más oscuro
        )
    ))

    fig5.update_layout(
    title='Relación entre Income y Gastos (Fijos vs Variables)',
    xaxis_title='Income',
    yaxis_title='Gastos',
    template='plotly_dark',  # Tema claro
    width=1000,  # Ancho del gráfico
    height=600,
    font=dict(size=12),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
    )

# ----------------------- Streamlit Layout -----------------------

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Ingresos por Edad", "Ahorros por Categoría", "Porcentaje de Gasto", "Gastos por Ciudad",'Relación entre ingresos y gastos'])

    with tab1:
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("""
                ### **Conclusiones:**
                - El ingreso promedio no sigue una tendencia clara y muestra fluctuaciones a lo largo de las edades.
                - Sin embargo, existe un umbral general de ingresos alrededor de los **41,000** (línea naranja).
                - Esto sugiere que factores externos, como la industria laboral o nivel educativo, pueden jugar un papel más relevante que la edad en sí.""",unsafe_allow_html=True)

    with tab2:
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("""
                ### **Conclusiones:** 
                - Las categorías con mayor ahorro potencial son **Misceláneos (9,816)**, **Entretenimiento (9,198)** y **Eating Out (9,185)**.
                - Esto indica que los gastos no esenciales representan una gran oportunidad para aumentar el ahorro.""")

    with tab3:
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("""
                ### **Conclusiones:** 
                - La mayor parte del gasto se destina a **alimentos (12%)**, seguido de transporte y servicios públicos.
                - Esto sugiere que los costos fijos consumen gran parte del ingreso, limitando la capacidad de ahorro.""")

    with tab4:
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("""
                ### **Conclusiones:** 
                - Mayor gasto en City_Tier 2: Los trabajadores independientes y estudiantes en estas ciudades tienen los gastos más altos.
                - Diferencias en ocupaciones: Los trabajadores independientes tienden a gastar más, mientras que los jubilados y estudiantes gastan menos.
                - Gastos fijos dominan: En la mayoría de los casos, los gastos fijos representan la mayor parte del gasto total.
                - Menor gasto en City_Tier 3: Sugiere un costo de vida más bajo o menores ingresos disponibles.""",unsafe_allow_html=True)


    with tab5:
        st.plotly_chart(fig5, use_container_width=True)
        st.write("**Conclusión:** ")


elif menu == "📈 Dashboard Power BI":
    st.markdown("## Dashboard Interactivo de Power BI")
    st.write("""
    Este dashboard interactivo explora los patrones financieros en India, integrando visualizaciones dinámicas de Power BI.
    Puedes interactuar directamente con los gráficos para explorar tendencias e insights clave.
    """)


        # Inserción del informe de Power BI
    st.components.v1.iframe(
        src="https://app.powerbi.com/view?r=eyJrIjoiNDY4ZTI4OTYtMjc1YS00YjlhLWEwZDItYjk3MWFmZjY5MzhkIiwidCI6IjhhZWJkZGI2LTM0MTgtNDNhMS1hMjU1LWI5NjQxODZlY2M2NCIsImMiOjl9",
        width=800,  # Ajusta el ancho según tus necesidades
        height=450,  # Ajusta la altura según tus necesidades
        scrolling=True
    )

# ---- Predicción Financiera ----
#elif menu == "🤖 Predicción Financiera":
    #st.markdown("## Predicción del Ingreso Disponible")
    #st.write("Introduce las características para predecir el ingreso disponible.")

    import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

data_sample = PandasParameterType(pd.DataFrame({"Income": pd.Series([44637.25], dtype="float64"), "Age": pd.Series([49], dtype="int8"), "Dependents": pd.Series([0], dtype="int8"), "Occupation": pd.Series(["Self_Employed"], dtype="object"), "City_Tier": pd.Series(["Tier_1"], dtype="object"), "Rent": pd.Series([13391.17], dtype="float64"), "Loan_Repayment": pd.Series([0], dtype="float64"), "Disposable_Income": pd.Series([11265.63], dtype="float64"), "Gasto_Fijo": pd.Series([24893.4], dtype="float64"), "Gastos_variables": pd.Series([8478.21], dtype="float64")}))
input_sample = StandardPythonParameterType({'data': data_sample})

result_sample = NumpyParameterType(np.array([0.0]))
output_sample = StandardPythonParameterType({'Results':result_sample})
sample_global_parameters = StandardPythonParameterType(1.0)

try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script_v2')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = "modelo\model.pkl"
    path = os.path.normpath(model_path)
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise

@input_schema('Inputs', input_sample)
@input_schema('GlobalParameters', sample_global_parameters, convert_to_provided_type=False)
@output_schema(output_sample)
def run(Inputs, GlobalParameters=1.0):
    data = Inputs['data']
    result = model.predict(data)
    return {'Results':result.tolist()}

def predecir_finanzas(datos):
    """ Realiza una predicción con el modelo cargado """
    income = datos["Income"]
    age = datos["Age"]
    dependents = datos["Dependents"]
    occupation = datos["Occupation"]
    city_tier = datos["City_Tier"]
    rent = datos["Rent"]
    loan_repayment = datos["Loan_Repayment"]
    disposable_income = datos["Disposable_Income"]
    gasto_fijo = datos["Gasto_Fijo"]
    gastos_variables = datos["Gastos_variables"]

    # Realizar la predicción
    data = [[income, age, dependents, occupation, city_tier, rent, loan_repayment, disposable_income, gasto_fijo, gastos_variables]]
    prediccion = model.predict(data)

    return {"Results": [prediccion[0]]}  # Devuelve la predicción en formato dict

# Función de predicción que se invoca en el Streamlit app
if menu == "🤖 Predicción Financiera":
    st.markdown("## Predicción del Ingreso Disponible")
    st.write("Introduce las características para predecir el ingreso disponible.")

    # Inputs del usuario
    Income = st.number_input("Ingreso Mensual (₹):", min_value=0, max_value=500000, value=50000)
    Age = st.number_input("Edad:", min_value=18, max_value=64, value=30)
    Dependents = st.selectbox("Dependientes:", df['Dependents'].unique())
    Occupation = st.selectbox("Ocupación:", df['Occupation'].unique())
    City_Tier = st.selectbox("City Tier:", df['City_Tier'].unique())
    Rent = st.number_input("Renta", min_value=0, max_value=1000000)
    Loan_Repayment = st.number_input("Préstamos", min_value=0, max_value=1000000)
    Disposable_Income = st.number_input("Ingreso Disponible", min_value=0, max_value=1000000)
    Gasto_Fijo = st.number_input("Gastos Fijos (₹):", min_value=0, max_value=500000, value=50000)
    Gastos_variables = st.number_input("Gastos variables (₹):", min_value=0, max_value=500000, value=50000)

    # Crear el diccionario con los datos ingresados por el usuario
    datos_usuario = {
        "Income": Income,
        "Age": Age,
        "Dependents": Dependents,
        "Occupation": Occupation,
        "City_Tier": City_Tier,
        "Rent": Rent,
        "Loan_Repayment": Loan_Repayment,
        "Disposable_Income": Disposable_Income,
        "Gasto_Fijo": Gasto_Fijo,
        "Gastos_variables": Gastos_variables
    }

    # Mostrar los datos del usuario
    st.write("Datos del Usuario:")
    st.write(pd.DataFrame([datos_usuario]))

    # Predicción
    if st.button("Predecir"):
        prediccion = predecir_finanzas(datos_usuario)
        st.success(f"El ingreso disponible estimado es: ₹{round(prediccion['Results'][0], 2)}")
# ---- Conclusiones y Recomendaciones ----
elif menu == "🔍 Conclusiones y Recomendaciones":
    st.title("📊 Conclusiones y Recomendaciones")

    # Sección de Conclusiones Generales
    st.header("🔎 Conclusiones")

    st.markdown("""
    El análisis de los métodos de ahorro en la India muestra que la capacidad de ahorro está influenciada por múltiples factores, incluyendo los ingresos, los patrones de gasto y las categorías en las que se distribuyen los recursos. 

    Se observa que una parte importante de los ingresos se destina a gastos fijos, lo que puede limitar la capacidad de ahorro si no se gestiona adecuadamente el presupuesto. Además, los gastos discrecionales, como el entretenimiento y comer fuera de casa, representan una oportunidad significativa para optimizar el ahorro.

    A través de una mejor planificación financiera y la adopción de hábitos más eficientes de consumo, es posible incrementar el porcentaje de ingresos destinados al ahorro sin afectar considerablemente la calidad de vida.
    """)

    # Sección de Recomendaciones
    st.header("💡 Recomendaciones")

    st.markdown("""
    ### **📌 Estrategias para Mejorar el Ahorro**
    ✅ **Establecer un presupuesto mensual**, asignando un porcentaje fijo al ahorro antes de destinarlo a otros gastos.  
    ✅ **Reducir gastos en categorías no esenciales**, priorizando necesidades sobre deseos.  
    ✅ **Automatizar el ahorro**, utilizando herramientas bancarias que transfieran automáticamente una parte del ingreso a cuentas de inversión o ahorro.  
    ✅ **Optimizar los gastos fijos**, buscando opciones más económicas en transporte, servicios públicos y alimentación.  
    ✅ **Fomentar la educación financiera**, para mejorar la toma de decisiones económicas a largo plazo.  

    ---
    📢 **Conclusión Final:** Pequeños ajustes en la administración de ingresos y gastos pueden marcar una gran diferencia en la capacidad de ahorro, permitiendo una mayor estabilidad financiera y un mejor futuro económico.
    """)

# ---- Botones de Navegación ----
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.session_state.current_page > 0:
        st.button("⬅️ Previous", on_click=previous_page)

with col3:
    if st.session_state.current_page < len(pages) - 1:
        st.button("Next ➡️", on_click=next_page)
