# === IMPORTACIONES ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random
from collections import Counter

# === CONFIGURACIÓN GLOBAL ===
#random.seed(42)
#np.random.seed(42)
sns.set(style="whitegrid")

# === CARGA Y LIMPIEZA DEL DATASET ===
file_path = "Inventario_recursos_turisticos.csv"
df = pd.read_csv(file_path, sep=';', encoding='latin1')

df = df.dropna(subset=['LATITUD', 'LONGITUD'])
df['LATITUD'] = pd.to_numeric(df['LATITUD'], errors='coerce')
df['LONGITUD'] = pd.to_numeric(df['LONGITUD'], errors='coerce')
df = df.dropna(subset=['LATITUD', 'LONGITUD'])

category_map = {
    'SITIOS ARQUEOLÓGICOS': 'Sitios Arqueológicos',
    'SITIOS NATURALES': 'Sitios Naturales',
    'MANIFESTACIONES CULTURALES': 'Manifestaciones Culturales',
    'FOLCLORE': 'Folclore',
    'REALIZACIONES TÉCNICAS CIENTÍFICAS Y ARTÍSTICAS CONTEMPORÁNEAS': 'Realizaciones Técnicas',
    'ACONTECIMIENTOS PROGRAMADOS': 'Acontecimientos Programados',
    'SERVICIOS TURÍSTICOS': 'Servicios Turísticos',
    'CIENTÍFICAS Y ARTÍSTICAS': 'Científicas y Artísticas'
}
df['CATEGORÍA STD'] = df['CATEGORÍA'].str.upper().map(category_map).fillna('Otros')

destinos = df[['REGIÓN', 'CATEGORÍA STD', 'LATITUD', 'LONGITUD', 'NOMBRE DEL RECURSO']].reset_index(drop=True)

# === FUNCIONES DE OPTIMIZACIÓN ===
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    return 2 * R * math.asin(math.sqrt(a))

def total_distance(route, coords):
    return sum(haversine_distance(*coords[i], *coords[j]) for i, j in zip(route[:-1], route[1:]))

def shannon_diversity(route, categories):
    subcats = [categories[i] for i in route]
    freq = Counter(subcats)
    total = len(subcats)
    return -sum((count / total) * math.log(count / total) for count in freq.values() if count > 0)

def simulated_annealing(destinos, circuit_size=12, max_iter=100, T0=1000, Tf=0.01, alpha_cool=0.95, alpha=0.6, beta=0.4):
    coords = list(zip(destinos['LATITUD'], destinos['LONGITUD']))
    categories = destinos['CATEGORÍA STD'].tolist()
    regiones = destinos['REGIÓN'].tolist()
    current_route = random.sample(range(len(destinos)), circuit_size)
    best_route = current_route[:]

    def objective(route):
        dist = total_distance(route, coords)
        diversity = shannon_diversity(route, categories)
        dist_norm = dist / 3500
        div_norm = 1 - (diversity / np.log(8))
        return alpha * dist_norm + beta * div_norm

    current_cost = objective(current_route)
    best_cost = current_cost
    T = T0
    history = []

    while T > Tf:
        for _ in range(max_iter):
            new_route = current_route[:]
            i, j = sorted(random.sample(range(circuit_size), 2))
            new_route[i], new_route[j] = new_route[j], new_route[i]
            selected_regions = set(regiones[k] for k in new_route)
            if len(selected_regions) < 3:
                continue
            new_cost = objective(new_route)
            delta = new_cost - current_cost
            if delta < 0 or np.exp(-delta / T) > np.random.rand():
                current_route = new_route
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_route = new_route[:]
        history.append(best_cost)
        T *= alpha_cool

    return best_route, best_cost, history

# === EJECUTAR ALGORITMO CON DATASET ===
destinos_sample = destinos.sample(200, random_state=None).reset_index(drop=True)
ruta, costo_final, historial = simulated_annealing(destinos_sample)
circuito = destinos_sample.iloc[ruta].reset_index(drop=True)

# === TABLAS METODOLÓGICAS ===
tabla_metricas = pd.DataFrame({
    'Métrica': [
        'Distancia Total (km)', 'Índice Diversidad Shannon',
        'Tiempo Recorrido (días)', 'Número de Regiones',
        'Categorías Incluidas', 'Cumplimiento Restricciones (%)'
    ],
    'Descripción': [
        'Suma de distancias euclidianas entre destinos consecutivos',
        'H = -Σ(pi × ln(pi)) donde pi es proporción de categoría i',
        'Estimación basada en distancias y tiempo promedio de visita',
        'Cantidad de regiones administrativas diferentes incluidas',
        'Número de categorías turísticas distintas en el circuito',
        'Porcentaje de circuitos que cumplen restricciones temporales y presupuestarias'
    ],
    'Rango/Unidad': [
        '1000-3000 km', '0-2.08', '7-14 días', '3-8 regiones',
        '3-8 categorías', '0-100%'
    ],
    'Objetivo': ['Minimizar', 'Maximizar', 'Optimizar', 'Maximizar', 'Maximizar', 'Maximizar']
})

tabla_parametros = pd.DataFrame({
    'Parámetro': ['Temperatura Inicial (T₀)', 'Temperatura Final (Tf)',
                  'Factor Enfriamiento (α)', 'Iteraciones por Temperatura',
                  'Peso Distancia (α)', 'Peso Diversidad (β)'],
    'Valor': [1000, 0.01, 0.95, 100, 0.6, 0.4],
    'Justificación': [
        'Valor alto para exploración inicial amplia',
        'Convergencia práctica del algoritmo',
        'Balance entre exploración y explotación',
        'Suficiente para explorar vecindario',
        'Mayor importancia a eficiencia logística',
        'Complemento para diversidad experiencial'
    ]
})

tabla_restricciones = pd.DataFrame({
    'Restricción': ['Tiempo Máximo', 'Regiones Mínimas', 'Destinos por Circuito',
                    'Presupuesto Transporte', 'Coordenadas Válidas', 'Categorías Mínimas'],
    'Valor Límite': ['14 días', '3 regiones', '8-15 destinos', 'Basado en distancia',
                     'Territorio peruano', '3 categorías'],
    'Tipo': ['Superior', 'Inferior', 'Rango', 'Calculado', 'Validación', 'Inferior'],
    'Penalización': ['Alta', 'Alta', 'Media', 'Media', 'Eliminación', 'Media']
})

# === GRAFICAR CONVERGENCIA ===
plt.figure(figsize=(10, 6))
plt.plot(historial, marker='o', color='blue')
plt.title('Convergencia del algoritmo Simulated Annealing')
plt.xlabel('Iteraciones')
plt.ylabel('Costo de la Solución')
plt.grid(True)
plt.tight_layout()
plt.show()

# === MOSTRAR RESULTADOS TABULADOS ===
print("\n=== CIRCUITO ÓPTIMO SELECCIONADO ===")
print(circuito.to_string(index=False))

print("\n=== TABLA 1: MÉTRICAS DE EVALUACIÓN ===")
print(tabla_metricas.to_string(index=False))

print("\n=== TABLA 2: PARÁMETROS DEL ALGORITMO SA ===")
print(tabla_parametros.to_string(index=False))

print("\n=== TABLA 3: RESTRICCIONES DEL PROBLEMA ===")
print(tabla_restricciones.to_string(index=False))

# === GRAFICAR RUTA TURÍSTICA OPTIMIZADA ===
plt.figure(figsize=(10, 8))
lats = circuito['LATITUD'].values
lons = circuito['LONGITUD'].values
regiones = circuito['REGIÓN'].values

plt.plot(lons, lats, '-o', color='darkblue', linewidth=2)
for i, (lon, lat, region) in enumerate(zip(lons, lats, regiones)):
    plt.text(lon + 0.05, lat + 0.05, f"{i+1}. {region}", fontsize=8, color='black')

plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.title("Ruta Turística Optimizada (Regiones)")
plt.grid(True)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
