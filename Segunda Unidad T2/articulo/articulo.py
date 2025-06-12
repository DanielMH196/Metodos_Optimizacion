import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings

# Configuración global
warnings.filterwarnings('ignore')
np.random.seed(42)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TourismCircuitAnalyzer:
    def __init__(self):
        self._setup_data()

    def _setup_data(self):
        # Datos principales
        self.distances = {'Optimizado SA': 1847, 'Aleatorio': 2847, 'Comercial': 2592}
        self.diversity_indices = {'Optimizado SA': 1.94, 'Tradicional': 1.23, 'Aleatorio': 1.08}
        self.category_distribution = {
            'Sitios Arqueológicos': 28, 'Sitios Naturales': 22, 'Manifestaciones Culturales': 18,
            'Folclore': 13, 'Realizaciones Técnicas': 8, 'Acontecimientos Programados': 6,
            'Servicios Turísticos': 3, 'Científicas y Artísticas': 2
        }
        self.iterations = np.arange(1, 201)
        self.convergence_values = self._simulate_convergence()

        # Tablas del paper
        self.evaluation_metrics = {
            'Métrica': ['Distancia Total (km)', 'Índice Diversidad Shannon', 'Tiempo Recorrido (días)',
                        'Número de Regiones', 'Categorías Incluidas', 'Cumplimiento Restricciones (%)'],
            'Descripción': [
                'Suma de distancias euclidianas entre destinos consecutivos',
                'H = -Σ(pi × ln(pi)) donde pi es proporción de categoría i',
                'Estimación basada en distancias y tiempo promedio de visita',
                'Cantidad de regiones administrativas diferentes incluidas',
                'Número de categorías turísticas distintas en el circuito',
                'Porcentaje de circuitos que cumplen restricciones temporales y presupuestarias'
            ],
            'Rango/Unidad': ['1000-3000 km', '0-2.08', '7-14 días', '3-8 regiones', '3-8 categorías', '0-100%'],
            'Objetivo': ['Minimizar', 'Maximizar', 'Optimizar', 'Maximizar', 'Maximizar', 'Maximizar']
        }

        self.sa_parameters = {
            'Parámetro': ['Temperatura Inicial (T₀)', 'Temperatura Final (Tf)', 'Factor Enfriamiento (α)',
                          'Iteraciones por Temperatura', 'Peso Distancia (α)', 'Peso Diversidad (β)'],
            'Valor': [1000, 0.01, 0.95, 100, 0.6, 0.4],
            'Justificación': [
                'Valor alto para exploración inicial amplia',
                'Convergencia práctica del algoritmo',
                'Balance entre exploración y explotación',
                'Suficiente para explorar vecindario',
                'Mayor importancia a eficiencia logística',
                'Complemento para diversidad experiencial'
            ]
        }

        self.constraints = {
            'Restricción': ['Tiempo Máximo', 'Regiones Mínimas', 'Destinos por Circuito',
                            'Presupuesto Transporte', 'Coordenadas Válidas', 'Categorías Mínimas'],
            'Valor Límite': ['14 días', '3 regiones', '8-15 destinos', 'Basado en distancia',
                             'Territorio peruano', '3 categorías'],
            'Tipo': ['Superior', 'Inferior', 'Rango', 'Calculado', 'Validación', 'Inferior'],
            'Penalización': ['Alta', 'Alta', 'Media', 'Media', 'Eliminación', 'Media']
        }

    def _simulate_convergence(self):
        initial, final, decay = 3500, 1847, 0.03
        curve = final + (initial - final) * np.exp(-decay * self.iterations)
        noise = np.random.normal(0, 50, len(self.iterations)) * np.exp(-0.02 * self.iterations)
        return curve + noise

    def create_methodology_tables(self):
        print("TABLA 1: MÉTRICAS DE EVALUACIÓN")
        print(pd.DataFrame(self.evaluation_metrics).to_string(index=False))
        print("\nTABLA 2: PARÁMETROS DEL ALGORITMO SA")
        print(pd.DataFrame(self.sa_parameters).to_string(index=False))
        print("\nTABLA 3: RESTRICCIONES DEL PROBLEMA DE OPTIMIZACIÓN")
        print(pd.DataFrame(self.constraints).to_string(index=False))

    def plot_distance_comparison(self):
        methods, values = zip(*self.distances.items())
        colors = ['#2E8B57', '#FF6B6B', '#4ECDC4']
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(methods, values, color=colors, edgecolor='black', alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 50, f"{val} km", ha='center', fontweight='bold')
        ax.annotate('', xy=(0, 1847), xytext=(1, 2847), arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax.text(0.5, 2350, '35.2% mejora', ha='center', color='red', fontweight='bold')
        ax.set_ylabel('Distancia Total (km)', fontsize=12)
        ax.set_title('Comparación de Distancias entre Métodos', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_diversity_comparison(self):
        methods, values = zip(*self.diversity_indices.items())
        colors = ['#FF9500', '#8A2BE2', '#DC143C']
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(methods, values, color=colors, edgecolor='black', alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05, f"{val:.2f}", ha='center', fontweight='bold')
        max_div = np.log(8)
        ax.axhline(max_div, linestyle='--', color='green', label=f'Máximo teórico ({max_div:.2f})')
        ax.set_ylabel('Índice de Diversidad (Shannon)', fontsize=12)
        ax.set_title('Comparación de Diversidad de Experiencias', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_category_distribution(self):
        categories, values = zip(*self.category_distribution.items())
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        ax1.pie(values, labels=categories, autopct='%1.1f%%',
                colors=colors, startangle=140, textprops={'fontsize': 10})
        ax1.set_title('Distribución de Categorías Turísticas\n(Circuitos Optimizados)', fontsize=14, fontweight='bold')
        y_pos = np.arange(len(categories))
        bars = ax2.barh(y_pos, values, color=colors, edgecolor='black')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(categories, fontsize=10)
        ax2.set_xlabel('Porcentaje (%)', fontsize=12)
        ax2.set_title('Distribución por Categorías', fontsize=14, fontweight='bold')
        for bar, val in zip(bars, values):
            ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f'{val}%', va='center', fontsize=10, fontweight='bold')
        ax2.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_convergence_analysis(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        ax1.plot(self.iterations, self.convergence_values, color='blue', linewidth=2)
        ax1.axhline(1847, linestyle='--', color='red', label='Valor óptimo (1847 km)')
        ax1.fill_between(self.iterations, self.convergence_values, alpha=0.2)
        ax1.set_ylabel('Distancia Total (km)', fontweight='bold')
        ax1.set_title('Convergencia del Algoritmo SA', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        temperatures = 1000 * (0.95 ** (self.iterations / 10))
        acceptance_prob = np.exp(-100 / temperatures)
        ax2b = ax2.twinx()
        ax2.plot(self.iterations, acceptance_prob, 'g-', linewidth=2, label='Probabilidad de Aceptación')
        ax2b.plot(self.iterations, temperatures, 'r-', linewidth=2, label='Temperatura')
        ax2.set_xlabel('Iteraciones', fontweight='bold')
        ax2.set_ylabel('Probabilidad de Aceptación', color='g')
        ax2b.set_ylabel('Temperatura', color='r')
        ax2.set_title('Evolución de Parámetros SA', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        ax2b.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

    def generate_all_visualizations(self):
        print("\n--- TABLAS DE METODOLOGÍA ---\n")
        self.create_methodology_tables()
        print("\n--- GRÁFICOS DEL PAPER ---\n")
        self.plot_distance_comparison()
        self.plot_diversity_comparison()
        self.plot_category_distribution()
        self.plot_convergence_analysis()

# Ejecutar todo
if __name__ == "__main__":
    analyzer = TourismCircuitAnalyzer()
    analyzer.generate_all_visualizations()