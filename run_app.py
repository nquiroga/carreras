import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Carreras",
    page_icon="🏃",
    layout="wide"
)

# Estilo para toda la página
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
    }
    .subtitle {
        font-size: 1.5rem;
        color: #2563EB;
        margin-top: 1rem;
    }
    .card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Título de la aplicación
st.markdown('<div class="title">📊 Análisis de Carreras</div>', unsafe_allow_html=True)

# Función para cargar datos
@st.cache_data
def cargar_datos():
    try:
        # Cargar el archivo CSV
        df = pd.read_csv('base_arg.csv')
        
        # Convertir columna 'fecha' a datetime si está en formato string
        if 'fecha' in df.columns and df['fecha'].dtype == 'object':
            df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce', format='%d.%m.%Y')
            
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

# Función para extraer km de la columna distancia
def extraer_distancia_km(texto):
    if pd.isna(texto):
        return np.nan
    
    texto = str(texto).lower()
    # Buscar patrón de número seguido de "km"
    km_match = re.search(r'(\d+(\.\d+)?)\s*km', texto)
    
    if km_match:
        return float(km_match.group(1))
    
    # Si no encuentra km pero hay un número, intentar extraerlo
    num_match = re.search(r'^(\d+(\.\d+)?)', texto)
    if num_match:
        return float(num_match.group(1))
    
    return np.nan

# Función para convertir tiempo (string) a segundos (int) con soporte para días
def tiempo_a_segundos(tiempo_str):
    if pd.isna(tiempo_str):
        return np.nan
    
    # Manejar el formato específico "6:34:10 h" o "MM:SS min"
    tiempo_str = str(tiempo_str).strip()
    
    # Eliminar la unidad al final (h, min)
    if " h" in tiempo_str:
        tiempo_str = tiempo_str.replace(" h", "")
    elif " min" in tiempo_str:
        tiempo_str = tiempo_str.replace(" min", "")
    
    partes = tiempo_str.split(':')
    
    if len(partes) == 3:  # formato HH:MM:SS
        horas, minutos, segundos = partes
        if 'd' in horas:
            dias = int(horas.split()[0][:-1]) * 24
            horas = dias + int(horas.split()[1])
        return int(horas) * 3600 + int(minutos) * 60 + int(segundos)
    elif len(partes) == 2:  # formato MM:SS
        minutos, segundos = partes
        return int(minutos) * 60 + int(segundos)
    else:
        return np.nan

# Función para convertir segundos a formato HH:MM:SS
def segundos_a_tiempo(segundos):
    if pd.isna(segundos):
        return "N/A"
    
    segundos = int(segundos)
    horas = segundos // 3600
    minutos = (segundos % 3600) // 60
    segundos = segundos % 60
    
    return f"{horas:02d}:{minutos:02d}:{segundos:02d}"

# Función para convertir segundos a formato HH:MM
def segundos_a_horas_minutos(segundos):
    if pd.isna(segundos):
        return np.nan
    
    horas = segundos / 3600
    return horas

# Función para formatear el eje y en formato HH:MM
def formato_horas_minutos(x):
    horas = int(x)
    minutos = int((x - horas) * 60)
    return f"{horas}:{minutos:02d}"

# Función para calcular el decil de edad
def calcular_decil_edad(edad):
    if pd.isna(edad):
        return np.nan
    return (edad // 10) * 10

# Función para extraer números de finishers
def extraer_finishers(texto):
    if pd.isna(texto):
        return {"total": "N/A", "masculino": "N/A", "femenino": "N/A"}
    
    # Formato esperado: "1105 (801 M, 304 F)"
    try:
        partes = texto.split(" ")
        total = partes[0]
        
        # Extraer los números entre paréntesis
        if "(" in texto and ")" in texto:
            dentro_parentesis = texto.split("(")[1].split(")")[0]
            masculino = dentro_parentesis.split(" M,")[0].strip()
            femenino = dentro_parentesis.split(",")[1].replace("F", "").strip()
            
            return {
                "total": total,
                "masculino": masculino,
                "femenino": femenino
            }
    except:
        pass
    
    return {"total": texto, "masculino": "N/A", "femenino": "N/A"}

# Cargar los datos
df = cargar_datos()

if df is not None:
    # Preprocesamiento de datos
    if 'Performance' in df.columns:
        df['tiempo_segundos'] = df['Performance'].apply(tiempo_a_segundos)
        df['tiempo_horas'] = df['tiempo_segundos'].apply(segundos_a_horas_minutos)
    
    # Calcular edad en el año de la carrera
    if 'YOB' in df.columns and 'fecha' in df.columns:
        df['edad'] = df['fecha'].dt.year - df['YOB']
        df['decil_edad'] = df['edad'].apply(calcular_decil_edad)
    
    # Extraer distancias en kilómetros
    if 'distancia' in df.columns:
        df['distancia_km'] = df['distancia'].apply(extraer_distancia_km)
    
    # Sidebar para selección de evento
    st.sidebar.markdown('### Selección de Carrera')
    
    eventos_disponibles = sorted(df['evento'].unique())
    evento_seleccionado = st.sidebar.selectbox('Selecciona un evento:', eventos_disponibles)
    
    # Filtrar datos por evento seleccionado
    df_evento = df[df['evento'] == evento_seleccionado].copy()
    
    # Verificar que hay datos para el evento seleccionado
    if len(df_evento) > 0:
        # Información básica de la carrera
        st.markdown('<div class="subtitle">📝 Información Básica de la Carrera</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fecha_evento = df_evento['fecha'].iloc[0].strftime('%d/%m/%Y') if pd.notna(df_evento['fecha'].iloc[0]) else "No disponible"
            st.metric("Fecha", fecha_evento)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("Evento", evento_seleccionado)
            st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            finishers_info = extraer_finishers(df_evento['finishers'].iloc[0]) if 'finishers' in df_evento.columns else {"total": "N/A", "masculino": "N/A", "femenino": "N/A"}
            st.metric("Total Finishers", finishers_info["total"])
            st.markdown(f"Masculinos: {finishers_info['masculino']} | Femeninos: {finishers_info['femenino']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            distancia = df_evento['distancia'].iloc[0] if 'distancia' in df_evento.columns else "No disponible"
            st.metric("Distancia/Tipo", distancia)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Información detallada y gráfico comparativo de distancias (más compacto)
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Preparar datos para el gráfico de distancias compacto
            if 'distancia_km' in df.columns:
                # Colocar en la barra lateral
                st.sidebar.markdown('---')
                st.sidebar.markdown('<p style="font-size: 14px;"><b>📏 Comparativa de Distancias</b></p>', unsafe_allow_html=True)
                
                eventos_con_distancia = df[['evento', 'distancia', 'distancia_km']].drop_duplicates().dropna(subset=['distancia_km'])
                
                if len(eventos_con_distancia) > 0:
                    # Ordenar por distancia
                    eventos_con_distancia = eventos_con_distancia.sort_values('distancia_km')
                    
                    # Crear una columna que indique si es el evento seleccionado
                    eventos_con_distancia['es_seleccionado'] = eventos_con_distancia['evento'] == evento_seleccionado
                    
                    # Obtener la distancia del evento seleccionado
                    distancia_seleccionada = eventos_con_distancia[eventos_con_distancia['es_seleccionado']]['distancia_km'].values[0] if any(eventos_con_distancia['es_seleccionado']) else None
                    
                    # Crear gráfico de puntos minimalista
                    fig_distancias = px.scatter(
                        eventos_con_distancia,
                        x='distancia_km',
                        y=None,  # Sin eje Y para que todos los puntos estén en una línea
                        color='es_seleccionado',
                        size_max=12,
                        size=[8 if not sel else 15 for sel in eventos_con_distancia['es_seleccionado']],  # Tamaño fijo
                        hover_data=['distancia'],
                        labels={
                            'distancia_km': 'Distancia (km)',
                            'es_seleccionado': ''
                        },
                        color_discrete_map={True: "#E11D48", False: "#1E3A8A"}
                    )
                    
                    # Configurar el gráfico para que sea muy compacto
                    fig_distancias.update_layout(
                        height=120,
                        margin=dict(l=10, r=10, t=30, b=20),
                        showlegend=False,
                        title=dict(
                            text=f"Esta carrera: {round(distancia_seleccionada, 1)} km",
                            font=dict(size=12),
                            x=0.5
                        ),
                        yaxis=dict(
                            showticklabels=False,
                            showgrid=False,
                            zeroline=False
                        ),
                        xaxis=dict(
                            title=dict(
                                text="Distancia (km)",
                                font=dict(size=10)
                            )
                        ),
                        plot_bgcolor='rgba(240,240,240,0.5)'
                    )
                    
                    # Quitar líneas de cuadrícula y otros elementos
                    fig_distancias.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(200,200,200,0.4)')
                    fig_distancias.update_yaxes(showgrid=False)
                    
                    # Mostrar el gráfico
                    st.sidebar.plotly_chart(fig_distancias, use_container_width=True)
                    
                    # Añadir una nota explicativa debajo del gráfico
                    st.sidebar.markdown('<p style="font-size: 11px; color: #666;">Los puntos representan la distancia de cada evento. El punto rojo corresponde al evento seleccionado.</p>', unsafe_allow_html=True)
                else:
                    st.sidebar.markdown('<p style="font-size: 12px; color: #666;">No hay información de distancias disponible.</p>', unsafe_allow_html=True)
        
        # Análisis de tiempos por género
        st.markdown('<div class="subtitle">⏱️ Análisis de Tiempos por Género</div>', unsafe_allow_html=True)
        
        # Separar por género
        df_masculino = df_evento[df_evento['M/F'] == 'M'].copy()
        df_femenino = df_evento[df_evento['M/F'] == 'F'].copy()
        
        # Calcular métricas para ambos géneros
        metricas = []
        
        # Top 10 masculino
        if len(df_masculino) >= 10:
            top10_m = df_masculino.nsmallest(10, 'tiempo_segundos')
            tiempo_prom_top10_m = top10_m['tiempo_segundos'].mean()
            metricas.append({"Categoría": "Top 10 Masculino", "Tiempo promedio (s)": tiempo_prom_top10_m, 
                             "Tiempo promedio": segundos_a_tiempo(tiempo_prom_top10_m)})
        else:
            metricas.append({"Categoría": "Top 10 Masculino", "Tiempo promedio (s)": np.nan, "Tiempo promedio": "Datos insuficientes"})
        
        # Top 10 femenino
        if len(df_femenino) >= 10:
            top10_f = df_femenino.nsmallest(10, 'tiempo_segundos')
            tiempo_prom_top10_f = top10_f['tiempo_segundos'].mean()
            metricas.append({"Categoría": "Top 10 Femenino", "Tiempo promedio (s)": tiempo_prom_top10_f, 
                             "Tiempo promedio": segundos_a_tiempo(tiempo_prom_top10_f)})
        else:
            metricas.append({"Categoría": "Top 10 Femenino", "Tiempo promedio (s)": np.nan, "Tiempo promedio": "Datos insuficientes"})
        
        # Promedio general
        tiempo_prom_total = df_evento['tiempo_segundos'].mean()
        metricas.append({"Categoría": "Promedio General", "Tiempo promedio (s)": tiempo_prom_total, 
                         "Tiempo promedio": segundos_a_tiempo(tiempo_prom_total)})
        
        # Últimos 15 masculino
        if len(df_masculino) >= 15:
            ultimos15_m = df_masculino.nlargest(15, 'tiempo_segundos')
            tiempo_prom_ultimos15_m = ultimos15_m['tiempo_segundos'].mean()
            metricas.append({"Categoría": "Últimos 15 Masculino", "Tiempo promedio (s)": tiempo_prom_ultimos15_m, 
                             "Tiempo promedio": segundos_a_tiempo(tiempo_prom_ultimos15_m)})
        else:
            metricas.append({"Categoría": "Últimos 15 Masculino", "Tiempo promedio (s)": np.nan, "Tiempo promedio": "Datos insuficientes"})
        
        # Últimos 15 femenino
        if len(df_femenino) >= 15:
            ultimos15_f = df_femenino.nlargest(15, 'tiempo_segundos')
            tiempo_prom_ultimos15_f = ultimos15_f['tiempo_segundos'].mean()
            metricas.append({"Categoría": "Últimos 15 Femenino", "Tiempo promedio (s)": tiempo_prom_ultimos15_f, 
                             "Tiempo promedio": segundos_a_tiempo(tiempo_prom_ultimos15_f)})
        else:
            metricas.append({"Categoría": "Últimos 15 Femenino", "Tiempo promedio (s)": np.nan, "Tiempo promedio": "Datos insuficientes"})
        
        # Mostrar tabla de métricas
        df_metricas = pd.DataFrame(metricas)
        st.table(df_metricas[['Categoría', 'Tiempo promedio']])
        
        # Gráfico violin para los tiempos
        st.markdown('<div class="subtitle">🎻 Distribución de Tiempos por Género</div>', unsafe_allow_html=True)
        
        # Verificar que tenemos datos de tiempo suficientes
        if df_evento['tiempo_segundos'].notna().sum() > 10:
            # Preparar datos para gráfico
            df_plot = df_evento[['M/F', 'tiempo_horas']].dropna().copy()
            df_plot = df_plot.rename(columns={'M/F': 'Género', 'tiempo_horas': 'Tiempo (horas)'})
            df_plot['Género'] = df_plot['Género'].map({'M': 'Masculino', 'F': 'Femenino'})
            
            # Crear gráfico violin con plotly
            fig_violin = px.violin(df_plot, 
                                 x="Género", 
                                 y="Tiempo (horas)", 
                                 color="Género",
                                 box=True,
                                 points="all",
                                 title="Distribución de Tiempos por Género",
                                 color_discrete_map={"Masculino": "#1E40AF", "Femenino": "#DB2777"})
            
            # Configurar el formato del eje y para mostrar horas:minutos
            fig_violin.update_layout(
                height=600,
                title_font_size=20,
                xaxis_title_font_size=16,
                yaxis_title_font_size=16,
                yaxis=dict(
                    tickmode='array',
                    tickvals=[int(i) + 0.5 for i in range(int(df_plot['Tiempo (horas)'].min()), int(df_plot['Tiempo (horas)'].max()) + 1)],
                    ticktext=[f"{i}:30" for i in range(int(df_plot['Tiempo (horas)'].min()), int(df_plot['Tiempo (horas)'].max()) + 1)]
                )
            )
            
            # Añadir medidas estadísticas como anotaciones
            for genero, color in zip(['Masculino', 'Femenino'], ['#1E40AF', '#DB2777']):
                subset = df_plot[df_plot['Género'] == genero]
                if len(subset) > 0:
                    median_time = subset['Tiempo (horas)'].median()
                    horas = int(median_time)
                    minutos = int((median_time - horas) * 60)
                    
                    fig_violin.add_annotation(
                        x=genero,
                        y=median_time,
                        text=f"Mediana: {horas}:{minutos:02d}",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=color,
                        bgcolor="white",
                        bordercolor=color
                    )
            
            st.plotly_chart(fig_violin, use_container_width=True)
            
            # Gráfico de distribución de llegadas a lo largo del tiempo
            st.markdown('<div class="subtitle">🏁 Distribución de Llegadas a Meta</div>', unsafe_allow_html=True)
            
            # Preparar datos para el gráfico de distribución
            df_dist = df_evento.sort_values('Rank').copy()
            df_dist = df_dist[['Rank', 'M/F', 'tiempo_horas']].dropna()
            df_dist['Género'] = df_dist['M/F'].map({'M': 'Masculino', 'F': 'Femenino'})
            
            # Crear gráfico de dispersión con plotly
            fig_dist = px.scatter(
                df_dist,
                x='tiempo_horas',
                y='Rank',
                color='Género',
                title='Distribución de Llegadas a Meta por Tiempo y Género',
                labels={
                    'tiempo_horas': 'Tiempo (horas)',
                    'Rank': 'Posición de Llegada'
                },
                color_discrete_map={"Masculino": "#1E40AF", "Femenino": "#DB2777"},
                opacity=0.7,
                size_max=10,
                height=600
            )
            
            # Configurar el formato del eje x para mostrar horas:minutos
            fig_dist.update_layout(
                height=600,
                title_font_size=20,
                xaxis_title_font_size=16,
                yaxis_title_font_size=16,
                xaxis=dict(
                    tickmode='array',
                    tickvals=[int(i) + 0.5 for i in range(int(df_dist['tiempo_horas'].min()), int(df_dist['tiempo_horas'].max()) + 1)],
                    ticktext=[f"{i}:30" for i in range(int(df_dist['tiempo_horas'].min()), int(df_dist['tiempo_horas'].max()) + 1)]
                )
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Histograma de distribución de tiempos por género
            fig_hist = px.histogram(
                df_plot,
                x='Tiempo (horas)',
                color='Género',
                barmode='overlay',
                opacity=0.7,
                nbins=30,
                title='Histograma de Tiempos por Género',
                color_discrete_map={"Masculino": "#1E40AF", "Femenino": "#DB2777"}
            )
            
            # Configurar el formato del eje x para mostrar horas:minutos
            fig_hist.update_layout(
                height=500,
                title_font_size=20,
                xaxis_title_font_size=16,
                yaxis_title_font_size=16,
                xaxis=dict(
                    tickmode='array',
                    tickvals=[int(i) + 0.5 for i in range(int(df_plot['Tiempo (horas)'].min()), int(df_plot['Tiempo (horas)'].max()) + 1)],
                    ticktext=[f"{i}:30" for i in range(int(df_plot['Tiempo (horas)'].min()), int(df_plot['Tiempo (horas)'].max()) + 1)]
                )
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
            
        else:
            st.warning("No hay suficientes datos de tiempo para generar las visualizaciones.")
        
        # Análisis por deciles de edad
        st.markdown('<div class="subtitle">👴 Análisis por Deciles de Edad</div>', unsafe_allow_html=True)
        
        # Asegurarse de que tenemos datos de edad suficientes
        if 'decil_edad' in df_evento.columns and df_evento['decil_edad'].notna().sum() > 5:
            # Eliminar valores faltantes
            df_deciles = df_evento.dropna(subset=['decil_edad', 'M/F', 'tiempo_segundos']).copy()
            
            # Agrupar por decil de edad y género
            deciles = df_deciles.groupby(['decil_edad', 'M/F']).agg({
                'tiempo_segundos': 'mean',
                'Rank': 'count'
            }).reset_index()
            
            deciles = deciles.rename(columns={
                'tiempo_segundos': 'Tiempo Promedio (s)',
                'Rank': 'Cantidad'
            })
            
            # Convertir tiempo a formato HH:MM:SS
            deciles['Tiempo Promedio'] = deciles['Tiempo Promedio (s)'].apply(segundos_a_tiempo)
            
            # Calcular tiempo en horas para la visualización
            deciles['Tiempo Promedio (h)'] = deciles['Tiempo Promedio (s)'] / 3600
            
            # Mapear valores de género
            deciles['Género'] = deciles['M/F'].map({'M': 'Masculino', 'F': 'Femenino'})
            
            # Crear gráfico de barras para tiempos promedio por decil de edad y género
            fig_deciles = px.bar(
                deciles, 
                x='decil_edad', 
                y='Tiempo Promedio (h)',
                color='Género',
                barmode='group',
                title='Tiempo Promedio por Decil de Edad y Género',
                labels={
                    'decil_edad': 'Decil de Edad',
                    'Tiempo Promedio (h)': 'Tiempo Promedio (horas)'
                },
                color_discrete_map={"Masculino": "#1E40AF", "Femenino": "#DB2777"}
            )
            
            # Configurar formato del eje y
            fig_deciles.update_layout(
                height=500,
                title_font_size=20,
                xaxis_title_font_size=16,
                yaxis_title_font_size=16,
                yaxis=dict(
                    tickmode='array',
                    tickvals=[int(i) + 0.5 for i in range(int(deciles['Tiempo Promedio (h)'].min()), int(deciles['Tiempo Promedio (h)'].max()) + 1)],
                    ticktext=[f"{i}:30" for i in range(int(deciles['Tiempo Promedio (h)'].min()), int(deciles['Tiempo Promedio (h)'].max()) + 1)]
                )
            )
            
            # Añadir etiquetas de tiempo en formato HH:MM
            for i, row in deciles.iterrows():
                tiempo_h = row['Tiempo Promedio (s)'] / 3600
                horas = int(tiempo_h)
                minutos = int((tiempo_h - horas) * 60)
                
                fig_deciles.add_annotation(
                    x=row['decil_edad'],
                    y=tiempo_h,
                    text=f"{horas}:{minutos:02d}",
                    showarrow=False,
                    yshift=10,
                    font=dict(size=10, color='black')
                )
            
            st.plotly_chart(fig_deciles, use_container_width=True)
            
            # Tabla de deciles
            st.markdown("#### Datos por Decil de Edad")
            
            # Preparar tabla para mostrar
            tabla_deciles = deciles[['decil_edad', 'Género', 'Tiempo Promedio', 'Cantidad']]
            tabla_deciles = tabla_deciles.rename(columns={'decil_edad': 'Decil de Edad'})
            tabla_deciles['Decil de Edad'] = tabla_deciles['Decil de Edad'].astype(int).astype(str) + "s"
            
            # Mostrar tabla
            st.dataframe(tabla_deciles, use_container_width=True)
            
            # Gráfico adicional: Cantidad de participantes por decil de edad y género
            fig_cantidad = px.bar(
                deciles, 
                x='decil_edad', 
                y='Cantidad',
                color='Género',
                barmode='group',
                title='Cantidad de Participantes por Decil de Edad y Género',
                labels={
                    'decil_edad': 'Decil de Edad',
                    'Cantidad': 'Número de Participantes'
                },
                color_discrete_map={"Masculino": "#1E40AF", "Femenino": "#DB2777"}
            )
            
            fig_cantidad.update_layout(
                height=500,
                title_font_size=20,
                xaxis_title_font_size=16,
                yaxis_title_font_size=16
            )
            
            st.plotly_chart(fig_cantidad, use_container_width=True)
            
        else:
            st.warning("No hay suficientes datos de edad para generar análisis por deciles.")
        
    else:
        st.warning(f"No hay datos disponibles para el evento seleccionado: {evento_seleccionado}")
else:
    st.error("No se pudieron cargar los datos. Por favor verifica que el archivo 'base_arg_pickle.csv' existe en el directorio.")

# Pie de página
st.markdown("---")
st.markdown("Desarrollado con 🏃 por 🐢 y ClaudeAI")
