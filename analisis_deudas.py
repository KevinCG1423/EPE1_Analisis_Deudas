"""
================================================================================
ANÁLISIS EXPLORATORIO DE CARTERA DE DEUDAS
Institución Financiera - Análisis de Riesgo
================================================================================
Script que ejecuta un análisis completo de la cartera de deudas de clientes,
incluyendo carga, limpieza, transformación, análisis y visualización de datos.

Requisitos: pandas, numpy, matplotlib
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Backend interactivo para mostrar gráficos
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuración de matplotlib
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("INICIO DEL ANÁLISIS DE CARTERA DE DEUDAS")
print("=" * 80)

# ================================================================================
# PARTE I - CARGA Y EXPLORACIÓN DE DATOS (10 pts.)
# ================================================================================

print("\n\n" + "=" * 80)
print("PARTE I - CARGA Y EXPLORACIÓN DE DATOS")
print("=" * 80)

# Parte I - Pregunta 1: Cargue el archivo y muestre las primeras 10 filas
print("\n# Parte I - Pregunta 1: Carga del archivo y primeras 10 filas")
print("-" * 80)
df = pd.read_csv('deudas_personas.csv')
print("\nPrimeras 10 filas del DataFrame:")
print(df.head(10))

# Parte I - Pregunta 2: Dimensiones y tipos de datos
print("\n\n# Parte I - Pregunta 2: Dimensiones y tipos de datos del DataFrame")
print("-" * 80)
print(f"\nDimensiones del DataFrame: {df.shape[0]} filas x {df.shape[1]} columnas")
print("\nTipos de datos de cada columna:")
print(df.dtypes)

# Parte I - Pregunta 3: Valores nulos y filas duplicadas
print("\n\n# Parte I - Pregunta 3: Valores nulos y filas duplicadas")
print("-" * 80)
print("\nCantidad de valores nulos por columna:")
print(df.isnull().sum())
print(f"\nTotal de valores nulos en el DataFrame: {df.isnull().sum().sum()}")
print(f"\nCantidad de filas duplicadas: {df.duplicated().sum()}")

# Parte I - Pregunta 4: Estadísticas descriptivas de variables numéricas
print("\n\n# Parte I - Pregunta 4: Estadísticas descriptivas de variables numéricas")
print("-" * 80)
print("\nEstadísticas descriptivas de: monto_deuda, ingreso_mensual, dias_mora, tasa_interes_anual")
print(df[['monto_deuda', 'ingreso_mensual', 'dias_mora', 'tasa_interes_anual']].describe())

# ================================================================================
# PARTE II - LIMPIEZA Y TRANSFORMACIÓN DE DATOS (10 pts.)
# ================================================================================

print("\n\n" + "=" * 80)
print("PARTE II - LIMPIEZA Y TRANSFORMACIÓN DE DATOS")
print("=" * 80)

# Parte II - Pregunta 1: Eliminar filas duplicadas
print("\n# Parte II - Pregunta 1: Eliminación de filas duplicadas")
print("-" * 80)
print(f"Cantidad de filas antes de eliminar duplicados: {len(df)}")
df = df.drop_duplicates()
print(f"Cantidad de filas después de eliminar duplicados: {len(df)}")

# Parte II - Pregunta 2: Crear columna rango_etario
print("\n\n# Parte II - Pregunta 2: Creación de columna 'rango_etario'")
print("-" * 80)

def asignar_rango_etario(edad):
    """Clasifica la edad en tramos: 18-25, 26-35, 36-50, 51-65, 66+"""
    if pd.isna(edad):
        return np.nan
    elif edad < 18:
        return np.nan
    elif edad <= 25:
        return "18-25"
    elif edad <= 35:
        return "26-35"
    elif edad <= 50:
        return "36-50"
    elif edad <= 65:
        return "51-65"
    else:
        return "66+"

df['rango_etario'] = df['edad'].apply(asignar_rango_etario)
print("\nDistribución de clientes por rango etario:")
print(df['rango_etario'].value_counts().sort_index())

# Parte II - Pregunta 3: Crear columna carga_financiera
print("\n\n# Parte II - Pregunta 3: Creación de columna 'carga_financiera'")
print("-" * 80)

def calcular_carga_financiera(row):
    """Calcula el porcentaje de deuda respecto del ingreso anual"""
    if pd.isna(row['ingreso_mensual']) or row['ingreso_mensual'] == 0:
        return np.nan
    ingreso_anual = row['ingreso_mensual'] * 12
    carga = (row['monto_deuda'] / ingreso_anual) * 100
    return carga

df['carga_financiera'] = df.apply(calcular_carga_financiera, axis=1)
print("\nEstadísticas de carga_financiera:")
print(df['carga_financiera'].describe())

# ================================================================================
# PARTE III - ANÁLISIS CON PANDAS (15 pts.)
# ================================================================================

print("\n\n" + "=" * 80)
print("PARTE III - ANÁLISIS CON PANDAS")
print("=" * 80)

# Parte III - Pregunta 1: Monto total de deuda por región
print("\n# Parte III - Pregunta 1: Monto total de deuda por región")
print("-" * 80)
deuda_por_region = df.groupby('region')['monto_deuda'].sum().sort_values(ascending=False)
print("\nMonto total adeudado por región:")
print(deuda_por_region)

# Parte III - Pregunta 2: Promedio de días de mora por tipo de deuda
print("\n\n# Parte III - Pregunta 2: Promedio de días de mora por tipo de deuda")
print("-" * 80)
mora_por_tipo = df.groupby('tipo_deuda')['dias_mora'].mean().sort_values(ascending=False)
print("\nPromedio de días de mora:")
print(mora_por_tipo)

# Parte III - Pregunta 3: Porcentaje de clientes por estado de deuda
print("\n\n# Parte III - Pregunta 3: Porcentaje de clientes por estado de deuda")
print("-" * 80)
estado_porcentaje = df['estado_deuda'].value_counts(normalize=True) * 100
print("\nPorcentaje de clientes:")
print(estado_porcentaje.round(2))

# Parte III - Pregunta 4: Institución financiera con mayor monto en clientes morosos
print("\n\n# Parte III - Pregunta 4: Institución financiera con mayor monto en clientes morosos")
print("-" * 80)
morosos = df[df['dias_mora'] > 0]
institucion_morosos = morosos.groupby('institucion_financiera')['monto_deuda'].sum().sort_values(ascending=False)
print("\nMonto total en clientes morosos:")
print(institucion_morosos.head(10))

# Parte III - Pregunta 5: Top 10 clientes con mayor carga_financiera
print("\n\n# Parte III - Pregunta 5: Top 10 clientes con mayor carga_financiera")
print("-" * 80)
top_10_carga = df.nlargest(10, 'carga_financiera')[['id_cliente', 'tipo_deuda', 'monto_deuda', 'ingreso_mensual', 'carga_financiera']]
print("\nTop 10:")
print(top_10_carga.to_string())

# ================================================================================
# PARTE IV - VISUALIZACIÓN CON MATPLOTLIB (15 pts.)
# ================================================================================

print("\n\n" + "=" * 80)
print("PARTE IV - VISUALIZACIÓN CON MATPLOTLIB")
print("=" * 80)

try:
    print("\nGenerando gráfico 1...")
    fig, ax = plt.subplots(figsize=(12, 6))
    deuda_por_region.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_title('Monto Total de Deuda por Region', fontsize=14, fontweight='bold')
    ax.set_xlabel('Region', fontsize=12)
    ax.set_ylabel('Monto Deuda (CLP)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('grafico_01_deuda_por_region.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("OK - grafico_01_deuda_por_region.png")
except Exception as e:
    print(f"Error en gráfico 1: {e}")

try:
    print("\nGenerando gráfico 2...")
    fig, ax = plt.subplots(figsize=(10, 8))
    tipo_deuda_counts = df['tipo_deuda'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    ax.pie(tipo_deuda_counts, labels=tipo_deuda_counts.index, autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax.set_title('Distribucion de Clientes por Tipo de Deuda', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('grafico_02_distribucion_tipo_deuda.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("OK - grafico_02_distribucion_tipo_deuda.png")
except Exception as e:
    print(f"Error en gráfico 2: {e}")

try:
    print("\nGenerando gráfico 3...")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(df['dias_mora'], bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax.set_title('Distribucion de Dias de Mora', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dias de Mora', fontsize=12)
    ax.set_ylabel('Cantidad de Clientes', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('grafico_03_histograma_dias_mora.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("OK - grafico_03_histograma_dias_mora.png")
except Exception as e:
    print(f"Error en gráfico 3: {e}")

try:
    print("\nGenerando gráfico 4...")
    fig, ax = plt.subplots(figsize=(12, 6))
    df.boxplot(column='monto_deuda', by='estado_deuda', ax=ax)
    ax.set_title('Distribucion del Monto de Deuda por Estado de Mora', fontsize=14, fontweight='bold')
    ax.set_xlabel('Estado de Deuda', fontsize=12)
    ax.set_ylabel('Monto Deuda (CLP)', fontsize=12)
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig('grafico_04_boxplot_monto_por_estado.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("OK - grafico_04_boxplot_monto_por_estado.png")
except Exception as e:
    print(f"Error en gráfico 4: {e}")

# ================================================================================
# PARTE V - INTERPRETACIÓN DE RESULTADOS
# ================================================================================

print("\n\n" + "=" * 80)
print("PARTE V - INTERPRETACIÓN DE RESULTADOS")
print("=" * 80)

total_clientes = len(df)
total_deuda = df['monto_deuda'].sum()
porcentaje_morosos = (df[df['dias_mora'] > 0].shape[0] / total_clientes) * 100
monto_moroso_total = df[df['dias_mora'] > 0]['monto_deuda'].sum()
porcentaje_deuda_morosa = (monto_moroso_total / total_deuda) * 100

interpretacion = f"""
RESUMEN EJECUTIVO DEL ANÁLISIS DE CARTERA DE DEUDAS

1. DIMENSIONES DE LA CARTERA:
   Total de clientes únicos: {total_clientes}
   Monto total adeudado: ${total_deuda:,.0f} CLP
   Monto promedio por cliente: ${df['monto_deuda'].mean():,.0f} CLP
   Monto mediano por cliente: ${df['monto_deuda'].median():,.0f} CLP

2. ANÁLISIS DE MOROSIDAD:
   Cantidad de clientes morosos: {df[df['dias_mora'] > 0].shape[0]}
   Porcentaje de clientes morosos: {porcentaje_morosos:.2f}%
   Monto total en mora: ${monto_moroso_total:,.0f} CLP
   Porcentaje de deuda en mora: {porcentaje_deuda_morosa:.2f}%
   Promedio de días de mora: {df[df['dias_mora'] > 0]['dias_mora'].mean():.1f} días

3. DISTRIBUCIÓN POR ESTADO DE DEUDA:
   Al día: {df[df['estado_deuda'] == 'Al día'].shape[0]} clientes
   Mora temprana: {df[df['estado_deuda'] == 'Mora temprana'].shape[0]} clientes
   Mora media: {df[df['estado_deuda'] == 'Mora media'].shape[0]} clientes
   Mora avanzada: {df[df['estado_deuda'] == 'Mora avanzada'].shape[0]} clientes

4. CONCENTRACIÓN REGIONAL:
   Región con mayor deuda: {deuda_por_region.idxmax()}
   Monto: ${deuda_por_region.max():,.0f} CLP

5. COMPOSICIÓN POR TIPO DE DEUDA:
   Tipo más frecuente: {df['tipo_deuda'].value_counts().idxmax()}
   Tipo con mayor morosidad: {df.groupby('tipo_deuda')['dias_mora'].mean().idxmax()}

6. CARGA FINANCIERA:
   Carga financiera promedio: {df['carga_financiera'].mean():.2f}%
   Carga financiera mediana: {df['carga_financiera'].median():.2f}%
   Clientes con carga > 100%: {(df['carga_financiera'] > 100).sum()}

7. RECOMENDACIONES:

   a) Mora Avanzada: {porcentaje_deuda_morosa:.1f}% de la cartera en estado moroso
      requiere intervención inmediata.

   b) Enfoque Regional: Priorizar cobranza en {deuda_por_region.idxmax()}
      que concentra {(deuda_por_region.max()/total_deuda)*100:.1f}% de la deuda.

   c) Revisión de Productos: {df.groupby('tipo_deuda')['dias_mora'].mean().idxmax()}
      presenta mayor morosidad promedio.

   d) Carga Financiera: {(df['carga_financiera'] > 100).sum()} clientes con
      carga > 100% requieren evaluación especial.

8. INDICADORES DE RIESGO:
   Ratio de morosidad: {porcentaje_morosos:.2f}%
   Exposición concentrada: {(deuda_por_region.max()/total_deuda)*100:.1f}%

CONCLUSIÓN:
La cartera presenta morosidad del {porcentaje_morosos:.1f}%. Se recomienda
seguimiento continuo y revisión de políticas crediticias.
"""

print(interpretacion)

with open('interpretacion.txt', 'w', encoding='utf-8') as f:
    f.write(interpretacion)

print("\nInterpretación guardada en: interpretacion.txt")

print("\n" + "=" * 80)
print("ANÁLISIS COMPLETADO EXITOSAMENTE")
print("=" * 80)
print("\nArchivos generados:")
print("  [OK] grafico_01_deuda_por_region.png")
print("  [OK] grafico_02_distribucion_tipo_deuda.png")
print("  [OK] grafico_03_histograma_dias_mora.png")
print("  [OK] grafico_04_boxplot_monto_por_estado.png")
print("  [OK] interpretacion.txt")
print("\nMostrando gráficos en ventanas emergentes...")
print("=" * 80)

# Crear gráficos interactivos para mostrar en ventanas
print("\nGráfico 1: Deuda por región")
fig1, ax1 = plt.subplots(figsize=(12, 6))
deuda_por_region.plot(kind='bar', ax=ax1, color='steelblue', edgecolor='black')
ax1.set_title('Monto Total de Deuda por Región', fontsize=14, fontweight='bold')
ax1.set_xlabel('Región', fontsize=12)
ax1.set_ylabel('Monto Deuda (CLP)', fontsize=12)
ax1.grid(axis='y', alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.tight_layout()

print("Gráfico 2: Distribución por tipo de deuda")
fig2, ax2 = plt.subplots(figsize=(10, 8))
tipo_deuda_counts = df['tipo_deuda'].value_counts()
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
ax2.pie(tipo_deuda_counts, labels=tipo_deuda_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90)
ax2.set_title('Distribución de Clientes por Tipo de Deuda', fontsize=14, fontweight='bold')
plt.tight_layout()

print("Gráfico 3: Histograma de días de mora")
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.hist(df['dias_mora'], bins=30, color='coral', edgecolor='black', alpha=0.7)
ax3.set_title('Distribución de Días de Mora en la Cartera', fontsize=14, fontweight='bold')
ax3.set_xlabel('Días de Mora', fontsize=12)
ax3.set_ylabel('Cantidad de Clientes', fontsize=12)
ax3.grid(axis='y', alpha=0.3)
plt.tight_layout()

print("Gráfico 4: Boxplot de monto por estado de deuda")
fig4, ax4 = plt.subplots(figsize=(12, 6))
df_sorted = df.sort_values('estado_deuda')
df_sorted.boxplot(column='monto_deuda', by='estado_deuda', ax=ax4)
ax4.set_title('Distribución del Monto de Deuda por Estado de Mora', fontsize=14, fontweight='bold')
ax4.set_xlabel('Estado de Deuda', fontsize=12)
ax4.set_ylabel('Monto Deuda (CLP)', fontsize=12)
plt.suptitle('')
plt.tight_layout()

# Mostrar todos los gráficos
plt.show()
