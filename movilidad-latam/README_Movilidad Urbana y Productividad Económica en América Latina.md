# 🚦 Movilidad Urbana y Productividad Económica en América Latina

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-brightgreen.svg)](https://pandas.pydata.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Descripción del Proyecto

Análisis de la relación entre la congestión vehicular y la productividad económica en las principales ciudades de América Latina. Este proyecto integra datos de tráfico en tiempo real de TomTom con indicadores socioeconómicos de la OECD para identificar patrones, outliers y oportunidades de inversión en infraestructura de transporte.

### 🎯 Objetivos

- Evaluar cómo la movilidad urbana se relaciona con el PIB per cápita
- Identificar ciudades con mayor desequilibrio entre desarrollo económico y calidad de movilidad
- Generar recomendaciones basadas en datos para inversión en infraestructura
- Crear un índice de eficiencia económica-movilidad

## 📁 Dataset

### Fuentes de datos:
- **TomTom Traffic Index**: Datos de congestión en tiempo real (2024)
- **OECD City Economy**: Indicadores económicos por ciudad (2023-2024)

### Variables principales:
- `jams_delay`: Retraso promedio por congestión (minutos)
- `gdp_per_capita`: PIB per cápita (USD)
- `unemployment_rate`: Tasa de desempleo (%)
- `population`: Población total
- `pm25`: Nivel de contaminación (μg/m³)

## 🛠️ Metodología

1. **Limpieza y preparación**:
   - Normalización de formatos numéricos (europeo → decimal)
   - Conversión de tipos de datos (fechas, enteros, flotantes)
   - Estandarización de nombres de columnas

2. **Análisis exploratorio**:
   - Agregación de datos de tráfico por ciudad-año
   - Identificación de outliers mediante boxplots
   - Análisis de correlaciones

3. **Métricas de eficiencia**:
   - Creación de índice PIB/congestión
   - Ranking de ciudades por eficiencia

## 📈 Resultados Clave

### 🏆 Top 5 Ciudades más Eficientes
| Ciudad | País | PIB per cápita | Congestión (min) |
|--------|------|----------------|------------------|
| Montevideo | Uruguay | $2,617,600 | 50.2 |
| Santiago | Chile | $2,150,000 | 120.5 |
| Brasilia | Brasil | $1,625,100 | 101.6 |
| Buenos Aires | Argentina | $1,811,700 | 571.1 |
| Mexico City | México | $2,111,100 | 2,833.1 |

### ⚠️ Ciudades con Mayor Desequilibrio
- **Bogotá, Colombia**: PIB medio-alto con congestión crítica
- **Lima, Perú**: Productividad limitada por movilidad ineficiente
- **São Paulo, Brasil**: Motor económico con alto costo de transporte

## 💡 Recomendaciones

1. **Inversión prioritaria**: Bogotá y Lima requieren intervención urgente en transporte masivo
2. **Modelos a seguir**: Montevideo y Santiago como casos de éxito en planificación urbana
3. **Impacto regional**: São Paulo ofrece el mayor retorno de inversión potencial

## 🚀 Tecnologías Utilizadas

- Python 3.9+
- Pandas (manipulación de datos)
- Matplotlib/Seaborn (visualizaciones)
- Jupyter Notebook

## 📦 Instalación y Uso

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/movilidad-economia-latam.git

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar notebook
jupyter notebook analisis_movilidad_latam.ipynb
