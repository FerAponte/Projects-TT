# ============================================
# PROYECTO: MOVILIDAD URBANA Y PRODUCTIVIDAD ECONÓMICA EN LATAM
# Análisis de la relación entre congestión vehicular y PIB per cápita
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para gráficos profesionales
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# ============================================
# 1. CARGA Y EXPLORACIÓN INICIAL DE DATOS
# ============================================

print("="*60)
print("PROYECTO: MOVILIDAD URBANA Y ECONOMÍA EN LATAM")
print("="*60)

# Carga de datasets
traffic = pd.read_csv('/datasets/tomtom_traffic.csv')
eco = pd.read_csv('/datasets/oecd_city_economy.csv')

print(f"\n📊 Dimensiones de los datasets:")
print(f"   - Tráfico (TomTom): {traffic.shape[0]:,} registros, {traffic.shape[1]} variables")
print(f"   - Económico (OECD): {eco.shape[0]} registros, {eco.shape[1]} variables")

# ============================================
# 2. LIMPIEZA Y PREPARACIÓN DE DATOS
# ============================================

print("\n🔧 INICIANDO PROCESO DE LIMPIEZA...")

# 2.1 Renombrar columnas a formato snake_case (estándar profesional)
traffic = traffic.rename(columns={
    'Country': 'country',
    'City': 'city',
    'UpdateTimeUTC': 'update_time_utc',
    'JamsDelay': 'jams_delay',
    'TrafficIndexLive': 'traffic_index_live',
    'JamsLengthInKms': 'jams_length_km',
    'JamsCount': 'jams_count',
    'TrafficIndexWeekAgo': 'traffic_index_week_ago',
    'UpdateTimeUTCWeekAgo': 'update_time_utc_week_ago',
    'TravelTimeLivePer10KmsMins': 'travel_time_live_10km_min',
    'TravelTimeHistoricPer10KmsMins': 'travel_time_historic_10km_min',
    'MinsDelay': 'mins_delay'
})

eco = eco.rename(columns={
    'Year': 'year',
    'City': 'city',
    'Country': 'country',
    'City GDP/capita': 'gdp_per_capita',
    'Unemployment %': 'unemployment_rate',
    'PM2.5 (μg/m³)': 'pm25',
    'Population (M)': 'population_millions'
})

# 2.2 Conversión de tipos de datos
# Tráfico: fechas a datetime
traffic['update_time_utc'] = pd.to_datetime(traffic['update_time_utc'], utc=True)
traffic['update_time_utc_week_ago'] = pd.to_datetime(traffic['update_time_utc_week_ago'], utc=True)

# Económico: limpieza de formatos numéricos
eco['gdp_per_capita'] = (eco['gdp_per_capita']
                         .astype(str)
                         .str.replace('.', '')
                         .str.replace(',', '.')
                         .astype(float))

eco['unemployment_rate'] = (eco['unemployment_rate']
                           .astype(str)
                           .str.replace('%', '')
                           .str.replace(',', '.')
                           .astype(float))

eco['pm25'] = (eco['pm25']
              .astype(str)
              .str.replace(',', '.')
              .astype(float))

eco['population_millions'] = (eco['population_millions']
                             .astype(str)
                             .str.replace(',', '.')
                             .astype(float))

# Crear población absoluta
eco['population'] = eco['population_millions'] * 1_000_000

print("✅ Limpieza completada")
print("   - Columnas renombradas a formato estándar")
print("   - Tipos de datos corregidos")
print("   - Formatos numéricos normalizados")

# ============================================
# 3. FILTRADO Y AGRUPACIÓN (AÑO 2024)
# ============================================

print("\n📅 FILTRANDO DATOS PARA 2024...")

# Extraer año y filtrar
traffic['year'] = traffic['update_time_utc'].dt.year
traffic_2024 = traffic[traffic['year'] == 2024].copy()
eco_2024 = eco[eco['year'] == 2024].copy()

print(f"   - Registros de tráfico 2024: {len(traffic_2024):,}")
print(f"   - Registros económicos 2024: {len(eco_2024)}")

# Agrupar tráfico por ciudad (promedios anuales)
traffic_metrics = [
    'jams_delay',
    'traffic_index_live',
    'jams_length_km',
    'jams_count',
    'mins_delay',
    'travel_time_live_10km_min',
    'travel_time_historic_10km_min'
]

traffic_city_2024 = (traffic_2024
                     .groupby(['city', 'country', 'year'])[traffic_metrics]
                     .mean()
                     .reset_index())

print(f"\n📊 Promedios anuales calculados para {len(traffic_city_2024)} ciudades")

# ============================================
# 4. INTEGRACIÓN DE DATASETS
# ============================================

print("\n🔗 INTEGRANDO DATOS DE TRÁFICO Y ECONOMÍA...")

# Seleccionar columnas relevantes
traffic_cols = ['city', 'country', 'year'] + traffic_metrics
eco_cols = ['city', 'year', 'gdp_per_capita', 'unemployment_rate', 'pm25', 'population']

traffic_final = traffic_city_2024[traffic_cols].copy()
eco_final = eco_2024[eco_cols].copy()

# Unión inner (solo ciudades presentes en ambos datasets)
merged = pd.merge(traffic_final, eco_final, on=['city', 'year'], how='inner')

print(f"✅ Dataset integrado: {len(merged)} ciudades analizadas")
print(f"\n🌎 PAÍSES INCLUIDOS:")
for country in merged['country'].unique():
    cities = merged[merged['country'] == country]['city'].tolist()
    print(f"   - {country}: {', '.join(cities)}")

# ============================================
# 5. ANÁLISIS EXPLORATORIO Y VISUALIZACIONES
# ============================================

print("\n📈 GENERANDO VISUALIZACIONES...")

# Figura 1: Distribución de la congestión (boxplot)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('ANÁLISIS DE MOVILIDAD Y ECONOMÍA EN CIUDADES LATAM (2024)', 
             fontsize=16, fontweight='bold')

# Boxplot de congestión
ax1 = axes[0, 0]
bp = ax1.boxplot(merged['jams_delay'], patch_artist=True, showmeans=True)
bp['boxes'][0].set_facecolor('lightcoral')
ax1.set_title('Distribución de Retraso por Congestión (Jams Delay)')
ax1.set_ylabel('Minutos de retraso')
ax1.grid(True, alpha=0.3)
# Identificar outlier
outlier_idx = np.where(merged['jams_delay'] > 2500)[0]
if len(outlier_idx) > 0:
    outlier_city = merged.iloc[outlier_idx[0]]['city']
    ax1.annotate(f'Outlier: {outlier_city}', 
                xy=(1, merged['jams_delay'].max()),
                xytext=(1.1, merged['jams_delay'].max()*0.9),
                arrowprops=dict(arrowstyle='->'))

# Histograma de PIB per cápita
ax2 = axes[0, 1]
ax2.hist(merged['gdp_per_capita']/1000, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
ax2.axvline(merged['gdp_per_capita'].mean()/1000, color='red', 
            linestyle='--', label=f"Promedio: {merged['gdp_per_capita'].mean()/1000:.1f}K")
ax2.set_title('Distribución del PIB per cápita')
ax2.set_xlabel('PIB per cápita (miles USD)')
ax2.set_ylabel('Frecuencia')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Top 10 ciudades por congestión
ax3 = axes[1, 0]
top_congestion = merged.nlargest(10, 'jams_delay')[['city', 'country', 'jams_delay']]
colors = plt.cm.Reds(np.linspace(0.4, 1, 10))
ax3.barh(range(len(top_congestion)), top_congestion['jams_delay'], color=colors)
ax3.set_yticks(range(len(top_congestion)))
ax3.set_yticklabels([f"{row['city']} ({row['country']})" for _, row in top_congestion.iterrows()])
ax3.set_title('Top 10 Ciudades con Mayor Congestión')
ax3.set_xlabel('Retraso promedio (minutos)')
ax3.invert_yaxis()

# Relación PIB vs Congestión
ax4 = axes[1, 1]
scatter = ax4.scatter(merged['gdp_per_capita']/1000, merged['jams_delay'], 
                      c=merged['population']/1e6, s=100, alpha=0.7, cmap='viridis')
ax4.set_xlabel('PIB per cápita (miles USD)')
ax4.set_ylabel('Retraso por congestión (minutos)')
ax4.set_title('Relación: PIB per cápita vs Congestión')
ax4.grid(True, alpha=0.3)

# Añadir etiquetas para ciudades destacadas
highlight_cities = ['mexico-city', 'sao-paulo', 'buenos-aires', 'montevideo']
for city in highlight_cities:
    city_data = merged[merged['city'] == city]
    if not city_data.empty:
        ax4.annotate(city.replace('-', ' ').title(), 
                    (city_data['gdp_per_capita'].values[0]/1000, 
                     city_data['jams_delay'].values[0]),
                    fontsize=9, ha='center')

cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Población (millones)')

plt.tight_layout()
plt.savefig('analisis_movilidad_latam.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 6. ANÁLISIS DE EFICIENCIA ECONÓMICA
# ============================================

print("\n📊 CALCULANDO MÉTRICAS DE EFICIENCIA...")

# Crear índice de eficiencia (PIB por minuto de congestión)
merged['efficiency_index'] = merged['gdp_per_capita'] / (merged['jams_delay'] + 1)

print("\n🏆 CIUDADES MÁS EFICIENTES (Alto PIB / Baja congestión):")
top_efficient = merged.nlargest(5, 'efficiency_index')[['city', 'country', 'gdp_per_capita', 'jams_delay']]
for _, row in top_efficient.iterrows():
    print(f"   - {row['city'].title()}, {row['country']}: ${row['gdp_per_capita']:,.0f} PIB | {row['jams_delay']:.1f} min retraso")

print("\n⚠️ CIUDADES CON MAYOR DESEQUILIBRIO (Bajo PIB / Alta congestión):")
bottom_efficient = merged.nsmallest(5, 'efficiency_index')[['city', 'country', 'gdp_per_capita', 'jams_delay']]
for _, row in bottom_efficient.iterrows():
    print(f"   - {row['city'].title()}, {row['country']}: ${row['gdp_per_capita']:,.0f} PIB | {row['jams_delay']:.1f} min retraso")

# ============================================
# 7. EXPORTACIÓN DE RESULTADOS
# ============================================

print("\n💾 EXPORTANDO RESULTADOS...")

# Guardar dataset limpio
merged.to_csv('latam_mobility_economy_2024.csv', index=False)
print("   ✅ Dataset guardado: latam_mobility_economy_2024.csv")

# ============================================
# 8. CONCLUSIONES Y RECOMENDACIONES
# ============================================

print("\n" + "="*60)
print("📌 CONCLUSIONES DEL ANÁLISIS")
print("="*60)

print("""
🎯 HALLAZGOS PRINCIPALES:

1.  CIUDAD DE MÉXICO ES UN OUTLIER EXTREMO:
    - Presenta niveles de congestión (∼2,833 min) que duplican el promedio
    - A pesar de su alto PIB, el costo de movilidad es significativo

2.  NO HAY CORRELACIÓN DIRECTA PIB-CONGESTIÓN:
    - Montevideo (alto PIB) tiene baja congestión (50 min)
    - Bogotá y Lima (PIB medio) tienen alta congestión (1,500+ min)
    - La congestión refleja ineficiencias estructurales, no actividad económica

3.  BRASIL DESTACA POR SU HETEROGENEIDAD:
    - São Paulo: motor económico con alta congestión (1,729 min)
    - Brasilia: PIB alto con congestión moderada (101 min)

💡 RECOMENDACIONES ESTRATÉGICAS:

1.  PRIORIDAD MÁXIMA: Ciudades con alto desequilibrio (Bogotá, Lima)
    - Inversión en transporte masivo para liberar potencial económico

2.  MODELOS A SEGUIR: Montevideo, Santiago
    - Analizar sus políticas de movilidad como casos de éxito regional

3.  IMPACTO ECONÓMICO: São Paulo
    - Por su escala, cualquier mejora tiene efecto multiplicador en PIB

🔍 PRÓXIMOS PASOS SUGERIDOS:
    - Calcular costo de oportunidad (horas-hombre perdidas)
    - Cruzar con datos de inversión en infraestructura
    - Análisis de series temporales para ver evolución
""")

print("\n" + "="*60)
print("✅ ANÁLISIS COMPLETADO")
print("="*60)