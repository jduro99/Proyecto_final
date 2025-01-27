import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import plotly.graph_objects as go
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
    st.markdown("""
    <h1 style="text-align: center; color: #4CAF50;">Análisis de Finanzas Personales en India</h1>
    <p style="text-align: justify; font-size: 18px;">
    Bienvenidos a esta exploración interactiva de las <b>finanzas personales en India</b>. Este proyecto está diseñado para analizar los ingresos, 
    los gastos y los patrones de ahorro de las personas, destacando insights clave que pueden ser utilizados para tomar decisiones financieras más 
    informadas. Además, presentamos un modelo predictivo para estimar el ingreso disponible basado en variables específicas.
    </p>
    """, unsafe_allow_html=True)

    # Nueva sección: Sobre India
    st.markdown("""
    ---
    ### Sobre India: Contexto Económico y Cultural
    India es el segundo país más poblado del mundo, con más de **1,400 millones de habitantes**. 
    Su economía es una de las de más rápido crecimiento a nivel global, impulsada por sectores clave como la tecnología, 
    la agricultura y los servicios. Sin embargo, la diversidad cultural y socioeconómica crea un escenario único 
    donde las finanzas personales varían ampliamente según factores como la región, ocupación y nivel educativo.
    
    #### Datos Clave:
    - **PIB (2023):** Más de $3.7 billones USD, ocupando el quinto lugar en el mundo.
    - **Distribución de ingresos:** Desigualdades significativas entre las zonas rurales y urbanas.
    - **Cultura del ahorro:** Tradicionalmente, las familias indias priorizan el ahorro, con una tasa media de ahorro del **30%** del ingreso anual.
    - **Ciudades principales:** Mumbai, Delhi, Bangalore, y Chennai son centros económicos clave.

    ---
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

    # Información adicional y contexto
    st.markdown("""
    ---
    ### Contexto del Proyecto
    India, con una población de más de 1,400 millones de personas, es uno de los mercados más diversos y dinámicos en términos financieros. 
    Este análisis tiene como objetivo comprender los patrones de gasto y ahorro de diferentes segmentos de la población, 
    utilizando datos reales y herramientas avanzadas de análisis.

    ### Objetivos Clave:
    - Identificar las principales áreas de gasto y ahorro.
    - Explorar cómo las características demográficas afectan las finanzas personales.
    - Utilizar Machine Learning para predecir el ingreso disponible y ofrecer recomendaciones prácticas.

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

    #fig3.update_traces(
        #marker_color=colors,
        #texttemplate='%{text:.2f}%',
        #textposition='outside'
    #)
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
        st.write("**Conclusión:** El ingreso promedio aumenta consistentemente con la edad, lo que refleja una relación positiva entre experiencia laboral y remuneración.")

    with tab2:
        st.plotly_chart(fig2, use_container_width=True)
        st.write("**Conclusión:** Los mayores ahorros potenciales se observan en las categorías de educación y entretenimiento, indicando oportunidades para ajustar gastos en estas áreas.")

    with tab3:
        st.plotly_chart(fig3, use_container_width=True)
        st.write("**Conclusión:** Las categorías de alimentos y transporte representan los mayores porcentajes de gasto sobre el ingreso total.")

    with tab4:
        st.plotly_chart(fig4, use_container_width=True)
        st.write("**Conclusión:** Los gastos varían significativamente entre ocupaciones y niveles de ciudad, con diferencias notables en gastos fijos y variables.")

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
elif menu == "🤖 Predicción Financiera":
    st.markdown("## Predicción del Ingreso Disponible")
    st.write("Introduce las características para predecir el ingreso disponible.")

    # Inputs del usuario
    age = st.number_input("Edad:", min_value=18, max_value=64, value=30)
    occupation = st.selectbox("Ocupación:", df['Occupation'].unique())
    city_tier = st.selectbox("City Tier:", df['City_Tier'].unique())
    income = st.number_input("Ingreso Mensual (₹):", min_value=0, max_value=500000, value=50000)
    gasto_fijo = st.number_input("Gastos Fijos (₹):", min_value=0, max_value=500000, value=50000)
    gasto_variable = st.number_input("Gastos variables (₹):", min_value=0, max_value=500000, value=50000)

    # Crear DataFrame con datos del usuario
    user_data = pd.DataFrame({
        "Age": [age],
        "Occupation": [occupation],
        "City_Tier": [city_tier],
        "Income": [income],
        "Gasto_fijo": [gasto_fijo],
        "Gasto_variable": [gasto_variable],
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

    csv = convert_df(df)
    st.download_button("Descargar Datos", data=csv, file_name="finanzas_india.csv", mime="text/csv")

# ---- Botones de Navegación ----
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.session_state.current_page > 0:
        st.button("⬅️ Previous", on_click=previous_page)

with col3:
    if st.session_state.current_page < len(pages) - 1:
        st.button("Next ➡️", on_click=next_page)
