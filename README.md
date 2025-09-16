# 🧭 Análisis Económico de Eventos y Turismo en Cartagena

![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Estado](https://img.shields.io/badge/Estado-En%20Desarrollo-yellow)

Este proyecto analiza el impacto económico de eventos de turismo en Cartagena utilizando una aplicación desarrollada en **Streamlit**. Estima el número de visitantes no residentes (PNL) y evalúa los **efectos directos, indirectos e inducidos** del turismo sobre los sectores económicos locales especificamente sobre el sector de alimentos, alojamiento y transporte interno.

---

## 📂 Estructura del Proyecto

```plaintext
calculos-turismo-cartagena/
├── app.py                ← Interfaz principal de Streamlit.
├── backend.py            ← Funciones de cálculo: PNL, efecto indirecto e inducido.
├── data/                 ← Carpeta sugerida para tus archivos .xlsx.
│   ├── EED.xlsx
│   ├── Encuesta.xlsx
│   └── Potencial_de_aforo.xlsx
│── README.md             ← Este archivo.
```

🧠 ¿Qué hace la App?
🔹 Paso 1: Carga de Archivos

    EED.xlsx: Contiene el efecto económico directo estimado por sector.

    Encuesta.xlsx: Respuestas de turistas sobre gasto, estadía y residencia.

    Potencial_de_aforo.xlsx: Capacidad esperada de los eventos religiosos.

🔹 Paso 2: Cálculo del PNL

    Estima la Población No Local (PNL) con base en el aforo total de eventos, la proporción de visitantes no residentes y el motivo del viaje.

🔹 Paso 3: Efecto Económico Indirecto

    Se multiplica el PNL estimado por:

        El gasto promedio del evento o sector seleccionado.

        El número promedio o mediano de días de estadía.

🔹 Paso 4: Efecto Inducido Neto

    Se calcula por sector a partir del efecto indirecto estimado y el multiplicador económico, restando el efecto base:

Efecto Inducido Neto = (Efecto Indirecto × Multiplicador) - Efecto Indirecto

🚀 Cómo Ejecutar la App

# Clona el repositorio
git clone https://github.com/Pnzvyva/turismo-religioso-analisis.git
cd turismo-religioso-analisis

# Instala las dependencias
pip install -r requirements.txt
# o si no tienes ese archivo
pip install streamlit pandas openpyxl scipy io

# Ejecuta la aplicación
streamlit run app.py

🛠 Tecnologías Utilizadas

    Python

    Streamlit

    Pandas

    SciPy

    Openpyxl

    io
