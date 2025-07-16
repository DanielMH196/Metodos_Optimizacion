from manim import *
import numpy as np

# Configuración global para mejor diseño
config.frame_width = 16
config.frame_height = 9

class IntroduccionProfesional(Scene):
    def construct(self):
        # Fondo elegante
        fondo = Rectangle(width=16, height=9, fill_color=DARK_BLUE, fill_opacity=0.1)
        self.add(fondo)
        
        # Título principal con mejor spacing
        titulo_principal = Text(
            "Análisis Comparativo de Algoritmos de", 
            font_size=42, 
            color=WHITE,
            weight=BOLD
        )
        titulo_principal.to_edge(UP, buff=1.5)
        
        titulo_secundario = Text(
            "Optimización Distribuida y Paralela",
            font_size=42,
            color=BLUE,
            weight=BOLD
        )
        titulo_secundario.next_to(titulo_principal, DOWN, buff=0.3)
        
        subtitulo = Text(
            "para Optimización de Precios en Retail y Supermercados",
            font_size=28,
            color=GRAY
        )
        subtitulo.next_to(titulo_secundario, DOWN, buff=0.5)
        
        # Información del autor en zona segura
        autor_info = VGroup(
            Text("Cliver Daniel Mamani Huatta", font_size=24, color=YELLOW),
            Text("Ing. Estadística e Informática", font_size=20, color=GRAY),
            Text("Universidad Nacional del Altiplano - Puno, Perú", font_size=18, color=GRAY)
        ).arrange(DOWN, buff=0.2)
        autor_info.to_edge(DOWN, buff=1)
        
        # Animaciones suaves
        self.play(
            Write(titulo_principal),
            run_time=2
        )
        self.wait(0.5)
        
        self.play(
            Write(titulo_secundario),
            run_time=2
        )
        self.wait(0.5)
        
        self.play(
            Write(subtitulo),
            run_time=1.5
        )
        self.wait(1)
        
        self.play(
            FadeIn(autor_info),
            run_time=2
        )
        self.wait(3)
        
        # Transición suave
        self.play(
            FadeOut(VGroup(titulo_principal, titulo_secundario, subtitulo, autor_info)),
            run_time=2
        )

class ProblemaRetailProfesional(Scene):
    def construct(self):
        # Título de sección limpio
        titulo = Text("El Desafío del Retail Moderno", font_size=40, color=BLUE, weight=BOLD)
        titulo.to_edge(UP, buff=1)
        
        underline = Line(LEFT * 3, RIGHT * 3, color=BLUE, stroke_width=3)
        underline.next_to(titulo, DOWN, buff=0.2)
        
        self.play(Write(titulo), Create(underline), run_time=2)
        self.wait(1)
        
        # Contexto del problema - bien espaciado
        contexto_titulo = Text("Desafíos Principales:", font_size=28, color=YELLOW)
        contexto_titulo.move_to(UP * 2)
        
        desafios = VGroup(
            Text("• Competencia intensa en múltiples canales", font_size=22),
            Text("• Decisiones de precios en tiempo real", font_size=22),
            Text("• Optimización simultánea de múltiples productos", font_size=22),
            Text("• Adaptación a diferentes modelos de negocio", font_size=22)
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        desafios.move_to(UP * 0.5)
        
        self.play(Write(contexto_titulo), run_time=1.5)
        self.wait(0.5)
        
        for desafio in desafios:
            self.play(Write(desafio), run_time=1.2)
            self.wait(0.3)
        
        self.wait(2)
        self.play(FadeOut(VGroup(contexto_titulo, desafios)), run_time=1.5)
        
        # Datasets comparativos - organizados horizontalmente
        datasets_titulo = Text("Datasets del Estudio", font_size=32, color=GREEN)
        datasets_titulo.move_to(UP * 2.5)
        
        # E-commerce box
        ecommerce_box = RoundedRectangle(
            width=5, height=4, corner_radius=0.3,
            color=BLUE, fill_opacity=0.1, stroke_width=2
        )
        ecommerce_box.shift(LEFT * 4)
        
        ecommerce_titulo = Text("E-commerce", font_size=28, color=BLUE, weight=BOLD)
        ecommerce_titulo.move_to(ecommerce_box.get_top() + DOWN * 0.5)
        
        ecommerce_datos = VGroup(
            Text("📊 676 registros", font_size=18),
            Text("🏷️ 9 categorías", font_size=18),
            Text("📅 2017-2018", font_size=18),
            Text("💰 $19.90 - $364", font_size=18),
            Text("🌎 Brasil", font_size=18)
        ).arrange(DOWN, buff=0.25)
        ecommerce_datos.move_to(ecommerce_box.get_center() + DOWN * 0.2)
        
        # Supermercado box
        super_box = RoundedRectangle(
            width=5, height=4, corner_radius=0.3,
            color=GREEN, fill_opacity=0.1, stroke_width=2
        )
        super_box.shift(RIGHT * 4)
        
        super_titulo = Text("Supermercado", font_size=28, color=GREEN, weight=BOLD)
        super_titulo.move_to(super_box.get_top() + DOWN * 0.5)
        
        super_datos = VGroup(
            Text("📊 1000 registros", font_size=18),
            Text("🏷️ 6 categorías", font_size=18),
            Text("📅 Ene-Mar 2019", font_size=18),
            Text("💰 $10.08 - $99.96", font_size=18),
            Text("🌎 Myanmar", font_size=18)
        ).arrange(DOWN, buff=0.25)
        super_datos.move_to(super_box.get_center() + DOWN * 0.2)
        
        # Animaciones organizadas
        self.play(Write(datasets_titulo), run_time=1.5)
        self.wait(0.5)
        
        self.play(
            Create(ecommerce_box),
            Create(super_box),
            run_time=2
        )
        
        self.play(
            Write(ecommerce_titulo),
            Write(super_titulo),
            run_time=1.5
        )
        
        self.play(
            Write(ecommerce_datos),
            Write(super_datos),
            run_time=2
        )
        
        self.wait(3)
        
        # Pregunta de investigación
        pregunta = Text(
            "¿Cómo optimizar precios eficientemente en ambos contextos?",
            font_size=26,
            color=YELLOW,
            weight=BOLD
        )
        pregunta.to_edge(DOWN, buff=1)
        
        self.play(Write(pregunta), run_time=2)
        self.wait(3)
        
        # Transición limpia
        todo = VGroup(
            titulo, underline, datasets_titulo, 
            ecommerce_box, ecommerce_titulo, ecommerce_datos,
            super_box, super_titulo, super_datos, pregunta
        )
        self.play(FadeOut(todo), run_time=2)

class TeoriaMatematicaProfesional(Scene):
    def construct(self):
        # Título principal
        titulo = Text("Fundamento Matemático", font_size=40, color=BLUE, weight=BOLD)
        titulo.to_edge(UP, buff=1)
        
        underline = Line(LEFT * 2.5, RIGHT * 2.5, color=BLUE, stroke_width=3)
        underline.next_to(titulo, DOWN, buff=0.2)
        
        self.play(Write(titulo), Create(underline), run_time=2)
        self.wait(1)
        
        # Función objetivo - bien centrada y espaciada
        funcion_titulo = Text("Función Objetivo:", font_size=28, color=YELLOW)
        funcion_titulo.move_to(UP * 2)
        
        # Ecuación principal con mejor formato
        ecuacion = MathTex(
            r"f(p) = 0.4 \cdot \pi(p) + 0.35 \cdot c(p) + 0.25 \cdot d(p)",
            font_size=36
        )
        ecuacion.move_to(UP * 1.2)
        
        # Explicación de componentes en tabla organizada
        componentes_box = RoundedRectangle(
            width=12, height=3, corner_radius=0.3,
            color=WHITE, fill_opacity=0.05, stroke_width=1
        )
        componentes_box.move_to(ORIGIN)
        
        componentes = VGroup(
            VGroup(
                MathTex(r"\pi(p)", color=YELLOW, font_size=28),
                Text("Función de beneficio (40%)", font_size=20)
            ).arrange(RIGHT, buff=0.5),
            VGroup(
                MathTex(r"c(p)", color=GREEN, font_size=28),
                Text("Competitividad del mercado (35%)", font_size=20)
            ).arrange(RIGHT, buff=0.5),
            VGroup(
                MathTex(r"d(p)", color=RED, font_size=28),
                Text("Función de demanda (25%)", font_size=20)
            ).arrange(RIGHT, buff=0.5)
        ).arrange(DOWN, buff=0.4)
        componentes.move_to(componentes_box.get_center())
        
        # Animaciones secuenciales
        self.play(Write(funcion_titulo), run_time=1.5)
        self.wait(0.5)
        
        self.play(Write(ecuacion), run_time=2.5)
        self.wait(1)
        
        self.play(Create(componentes_box), run_time=1)
        
        for componente in componentes:
            self.play(Write(componente), run_time=1.5)
            self.wait(0.5)
        
        self.wait(2)
        
        # Transición a PSO
        self.play(
            FadeOut(VGroup(funcion_titulo, ecuacion, componentes_box, componentes)),
            run_time=1.5
        )
        
        # PSO - Ecuación matemática
        pso_titulo = Text("Algoritmo PSO - Enjambre de Partículas", font_size=32, color=YELLOW)
        pso_titulo.move_to(UP * 2.5)
        
        # Ecuación PSO con mejor formato
        pso_ecuacion = MathTex(
            r"v_i(t+1) = w \cdot v_i(t) + c_1 r_1 (p_{best,i} - x_i(t)) + c_2 r_2 (g_{best} - x_i(t))",
            font_size=28
        )
        pso_ecuacion.move_to(UP * 1.5)
        
        # Parámetros en tabla organizada
        parametros_box = RoundedRectangle(
            width=13, height=3.5, corner_radius=0.3,
            color=YELLOW, fill_opacity=0.05, stroke_width=2
        )
        parametros_box.move_to(DOWN * 0.5)
        
        parametros_titulo = Text("Configuración de Parámetros:", font_size=24, color=YELLOW)
        parametros_titulo.move_to(parametros_box.get_top() + DOWN * 0.4)
        
        parametros = VGroup(
            Text("w = 0.7 (factor de inercia)", font_size=20),
            Text("c₁ = c₂ = 1.4 (coeficientes de aceleración)", font_size=20),
            Text("4 subgrupos × 25 partículas = 100 partículas totales", font_size=20),
            Text("Máximo 150 iteraciones con migración cada 10", font_size=20)
        ).arrange(DOWN, buff=0.3)
        parametros.move_to(parametros_box.get_center() + DOWN * 0.2)
        
        # Animaciones
        self.play(Write(pso_titulo), run_time=2)
        self.wait(0.5)
        
        self.play(Write(pso_ecuacion), run_time=3)
        self.wait(1)
        
        self.play(Create(parametros_box), run_time=1)
        self.play(Write(parametros_titulo), run_time=1)
        
        for param in parametros:
            self.play(Write(param), run_time=1.2)
            self.wait(0.3)
        
        self.wait(3)
        
        # Transición final
        todo_pso = VGroup(pso_titulo, pso_ecuacion, parametros_box, parametros_titulo, parametros)
        self.play(FadeOut(todo_pso), run_time=2)


class ResultadosElegantes(Scene):
    def construct(self):
        # === TÍTULO - ZONA SEGURA SUPERIOR ===
        titulo = Text("Resultados Comparativos", font_size=42, color=BLUE, weight=BOLD)
        titulo.to_edge(UP, buff=0.5)
        
        self.play(Write(titulo), run_time=2)
        self.wait(1)
        
        # === GRÁFICO PRINCIPAL - CENTRADO Y ESPACIADO ===
        axes = Axes(
            x_range=[0, 3, 1],
            y_range=[0, 90, 20],
            x_length=8,
            y_length=4.5,
            axis_config={
                "color": WHITE,
                "stroke_width": 2,
                "include_numbers": True,
                "font_size": 14
            }
        )
        axes.move_to(UP * 0.8)  # Posición fija, centrada
        
        # Etiquetas de ejes - SEPARADAS del gráfico
        y_label = Text("Mejora en Rentabilidad (%)", font_size=16, color=WHITE)
        y_label.rotate(PI/2)
        y_label.next_to(axes, LEFT, buff=0.8)
        
        # === DATOS EXACTOS DEL PAPER ===
        datos = {
            "E-commerce": {"PSO": 74.9, "GA": 71.2},
            "Supermercado": {"PSO": 82.3, "GA": 79.8}
        }
        
        # === CREAR BARRAS SIN SUPERPOSICIONES ===
        barras = VGroup()
        etiquetas = VGroup()
        
        x_positions = [0.8, 2.2]  # Posiciones fijas
        colores = {"PSO": BLUE, "GA": PURPLE}
        
        for i, (dataset, valores) in enumerate(datos.items()):
            x_base = x_positions[i]
            
            # Barra PSO
            barra_pso = Rectangle(
                width=0.3,
                height=valores["PSO"] / 20,
                fill_color=colores["PSO"],
                fill_opacity=0.8,
                stroke_color=WHITE,
                stroke_width=1
            )
            barra_pso.move_to(axes.coords_to_point(x_base - 0.2, valores["PSO"]/2, 0))
            
            # Barra GA
            barra_ga = Rectangle(
                width=0.3,
                height=valores["GA"] / 20,
                fill_color=colores["GA"],
                fill_opacity=0.8,
                stroke_color=WHITE,
                stroke_width=1
            )
            barra_ga.move_to(axes.coords_to_point(x_base + 0.2, valores["GA"]/2, 0))
            
            barras.add(barra_pso, barra_ga)
            
            # Etiquetas de valores - ENCIMA de las barras, SIN superposición
            label_pso = Text(f"{valores['PSO']}%", font_size=14, color=WHITE, weight=BOLD)
            label_pso.next_to(barra_pso, UP, buff=0.1)
            
            label_ga = Text(f"{valores['GA']}%", font_size=14, color=WHITE, weight=BOLD)
            label_ga.next_to(barra_ga, UP, buff=0.1)
            
            etiquetas.add(label_pso, label_ga)
            
            # Etiqueta del dataset - DEBAJO del gráfico
            dataset_label = Text(dataset, font_size=16, color=WHITE)
            dataset_label.next_to(axes.coords_to_point(x_base, 0, 0), DOWN, buff=0.4)
            etiquetas.add(dataset_label)
        
        # === LEYENDA - ESQUINA DERECHA, SIN INTERFERIR ===
        leyenda_pso = VGroup(
            Rectangle(width=0.4, height=0.2, fill_color=BLUE, fill_opacity=0.8, stroke_width=0),
            Text("PSO", font_size=14, color=WHITE)
        ).arrange(RIGHT, buff=0.2)
        
        leyenda_ga = VGroup(
            Rectangle(width=0.4, height=0.2, fill_color=PURPLE, fill_opacity=0.8, stroke_width=0),
            Text("GA", font_size=14, color=WHITE)
        ).arrange(RIGHT, buff=0.2)
        
        leyenda = VGroup(leyenda_pso, leyenda_ga).arrange(DOWN, buff=0.2)
        leyenda.to_corner(UR, buff=1)
        
        # === ANIMACIONES LIMPIAS ===
        self.play(Create(axes), Write(y_label), run_time=2)
        self.wait(0.5)
        
        self.play(Write(leyenda), run_time=1.5)
        self.wait(0.5)
        
        # Crear barras una por una
        for i, barra in enumerate(barras):
            self.play(GrowFromEdge(barra, DOWN), run_time=1.5)
            self.play(Write(etiquetas[i]), run_time=0.8)
            if i % 2 == 1:  # Después de cada par PSO-GA
                dataset_idx = i // 2
                self.play(Write(etiquetas[4 + dataset_idx]), run_time=0.8)
            self.wait(0.3)
        
        self.wait(2)
        
        # === TRANSICIÓN A ANÁLISIS - NUEVA PANTALLA ===
        self.play(
            FadeOut(VGroup(axes, y_label, barras, etiquetas, leyenda)),
            run_time=1.5
        )
        
        # === PANTALLA DE ANÁLISIS - SIN SUPERPOSICIONES ===
        analisis_titulo = Text("Hallazgos Principales", font_size=36, color=YELLOW, weight=BOLD)
        analisis_titulo.move_to(UP * 2.5)
        
        # Hallazgos organizados verticalmente, SIN gráfico de fondo
        hallazgos = VGroup(
            VGroup(
                Circle(radius=0.2, fill_color=GREEN, fill_opacity=0.8),
                Text("1", font_size=16, color=WHITE, weight=BOLD)
            ),
            Text("Supermercados superan a E-commerce por 7.4 puntos porcentuales", 
                font_size=20, color=WHITE)
        ).arrange(RIGHT, buff=0.5)
        
        hallazgo2 = VGroup(
            VGroup(
                Circle(radius=0.2, fill_color=BLUE, fill_opacity=0.8),
                Text("2", font_size=16, color=WHITE, weight=BOLD)
            ),
            Text("PSO consistentemente superior a GA en ambos contextos", 
                font_size=20, color=WHITE)
        ).arrange(RIGHT, buff=0.5)
        
        hallazgo3 = VGroup(
            VGroup(
                Circle(radius=0.2, fill_color=PURPLE, fill_opacity=0.8),
                Text("3", font_size=16, color=WHITE, weight=BOLD)
            ),
            Text("Rango de mejoras: 71.2% (mín) a 82.3% (máx)", 
                font_size=20, color=WHITE)
        ).arrange(RIGHT, buff=0.5)
        
        todos_hallazgos = VGroup(hallazgos, hallazgo2, hallazgo3)
        todos_hallazgos.arrange(DOWN, buff=0.8, aligned_edge=LEFT)
        todos_hallazgos.move_to(ORIGIN)
        
        # Animación del análisis
        self.play(Write(analisis_titulo), run_time=2)
        self.wait(0.5)
        
        for hallazgo in todos_hallazgos:
            self.play(Write(hallazgo), run_time=2)
            self.wait(0.8)
        
        self.wait(2)
        
        # === CONCLUSIÓN FINAL - ZONA INFERIOR SEGURA ===
        conclusion = Text(
            "Los supermercados presentan condiciones más favorables para optimización",
            font_size=22,
            color=GREEN,
            weight=BOLD
        )
        conclusion.to_edge(DOWN, buff=1)
        
        self.play(Write(conclusion), run_time=2.5)
        self.wait(3)
        
        # === FADE OUT FINAL LIMPIO ===
        todo_final = VGroup(titulo, analisis_titulo, todos_hallazgos, conclusion)
        self.play(FadeOut(todo_final), run_time=2)

class RendimientoParaleloProfesional(Scene):
    def construct(self):
        # === TÍTULO PRINCIPAL ===
        titulo = Text("Análisis de Rendimiento Paralelo", font_size=40, color=BLUE, weight=BOLD)
        titulo.to_edge(UP, buff=0.5)
        
        self.play(Write(titulo), run_time=2)
        self.wait(1)
        
        # === PANTALLA 1: CONCEPTOS ===
        conceptos_titulo = Text("Conceptos Fundamentales", font_size=28, color=YELLOW, weight=BOLD)
        conceptos_titulo.move_to(UP * 1.5)
        
        formula_speedup = MathTex(
            r"\text{Speedup} = \frac{T_{\text{secuencial}}}{T_{\text{paralelo}}}",
            font_size=28
        )
        formula_speedup.move_to(UP * 0.3)
        
        formula_eficiencia = MathTex(
            r"\text{Eficiencia} = \frac{\text{Speedup}}{4} \times 100\%",
            font_size=28
        )
        formula_eficiencia.move_to(DOWN * 0.3)
        
        explicacion = Text("Configuración: 4 procesos paralelos", font_size=18, color=GRAY)
        explicacion.move_to(DOWN * 1.5)
        
        self.play(Write(conceptos_titulo), run_time=1.5)
        self.wait(0.5)
        self.play(Write(formula_speedup), run_time=2)
        self.wait(1)
        self.play(Write(formula_eficiencia), run_time=2)
        self.wait(1)
        self.play(Write(explicacion), run_time=1.5)
        self.wait(2)
        
        # LIMPIAR PANTALLA COMPLETAMENTE
        self.play(FadeOut(VGroup(conceptos_titulo, formula_speedup, formula_eficiencia, explicacion)), run_time=1.5)
        self.wait(0.5)
        
        # === PANTALLA 2: GRÁFICO SPEEDUP ===
        # NO hay subtítulo que se superponga con el título principal
        
        axes = Axes(
            x_range=[0, 3, 1],
            y_range=[0, 4.5, 0.5],
            x_length=8,
            y_length=4,
            axis_config={"color": WHITE, "stroke_width": 2, "include_numbers": True, "font_size": 14}
        )
        axes.move_to(ORIGIN)
        
        # Línea ideal
        linea_ideal = DashedLine(
            axes.coords_to_point(0, 4, 0),
            axes.coords_to_point(2.5, 4, 0),
            color=RED, stroke_width=3
        )
        label_ideal = Text("Ideal (4x)", font_size=14, color=RED)
        label_ideal.next_to(linea_ideal, RIGHT, buff=0.1)
        
        # Datos del paper
        speedup_data = {"E-commerce": {"PSO": 3.6, "GA": 3.4}, "Supermercado": {"PSO": 3.7, "GA": 3.5}}
        
        barras = VGroup()
        etiquetas = VGroup()
        x_positions = [0.7, 1.8]
        
        for i, (dataset, valores) in enumerate(speedup_data.items()):
            x_base = x_positions[i]
            
            barra_pso = Rectangle(width=0.25, height=valores["PSO"] * 0.8, fill_color=BLUE, fill_opacity=0.8, stroke_color=WHITE, stroke_width=1)
            barra_pso.move_to(axes.coords_to_point(x_base - 0.15, valores["PSO"]/2, 0))
            
            barra_ga = Rectangle(width=0.25, height=valores["GA"] * 0.8, fill_color=PURPLE, fill_opacity=0.8, stroke_color=WHITE, stroke_width=1)
            barra_ga.move_to(axes.coords_to_point(x_base + 0.15, valores["GA"]/2, 0))
            
            barras.add(barra_pso, barra_ga)
            
            label_pso = Text(f"{valores['PSO']}x", font_size=12, color=WHITE, weight=BOLD)
            label_pso.next_to(barra_pso, UP, buff=0.05)
            
            label_ga = Text(f"{valores['GA']}x", font_size=12, color=WHITE, weight=BOLD)
            label_ga.next_to(barra_ga, UP, buff=0.05)
            
            etiquetas.add(label_pso, label_ga)
            
            dataset_label = Text(dataset, font_size=14, color=WHITE)
            dataset_label.next_to(axes.coords_to_point(x_base, 0, 0), DOWN, buff=0.3)
            etiquetas.add(dataset_label)
        
        y_label = Text("Speedup (x)", font_size=16, color=WHITE)
        y_label.rotate(PI/2)
        y_label.next_to(axes, LEFT, buff=0.5)
        
        leyenda_pso = VGroup(Rectangle(width=0.3, height=0.15, fill_color=BLUE, fill_opacity=0.8, stroke_width=0), Text("PSO", font_size=12, color=WHITE)).arrange(RIGHT, buff=0.2)
        leyenda_ga = VGroup(Rectangle(width=0.3, height=0.15, fill_color=PURPLE, fill_opacity=0.8, stroke_width=0), Text("GA", font_size=12, color=WHITE)).arrange(RIGHT, buff=0.2)
        leyenda = VGroup(leyenda_pso, leyenda_ga).arrange(DOWN, buff=0.2)
        leyenda.to_corner(UR, buff=1)
        
        # Animaciones sin títulos superpuestos
        self.play(Create(axes), Write(y_label), run_time=2)
        self.wait(0.5)
        self.play(Create(linea_ideal), Write(label_ideal), run_time=1.5)
        self.wait(0.5)
        self.play(Write(leyenda), run_time=1)
        self.wait(0.5)
        
        for i, barra in enumerate(barras):
            self.play(GrowFromEdge(barra, DOWN), run_time=1.5)
            self.play(Write(etiquetas[i]), run_time=0.8)
            if i % 2 == 1:
                dataset_idx = i // 2
                self.play(Write(etiquetas[4 + dataset_idx]), run_time=0.8)
            self.wait(0.3)
        
        self.wait(2)
        
        # LIMPIAR PANTALLA COMPLETAMENTE
        self.play(FadeOut(VGroup(axes, y_label, linea_ideal, label_ideal, barras, etiquetas, leyenda)), run_time=1.5)
        self.wait(0.5)
        
        # === PANTALLA 3: EFICIENCIA ===
        eficiencia_titulo = Text("Eficiencia Paralela", font_size=32, color=YELLOW, weight=BOLD)
        eficiencia_titulo.move_to(UP * 2)
        
        tabla_datos = VGroup(
            VGroup(Text("Dataset", font_size=18, color=WHITE, weight=BOLD), Text("PSO", font_size=18, color=BLUE, weight=BOLD), Text("GA", font_size=18, color=PURPLE, weight=BOLD)).arrange(RIGHT, buff=1.5),
            VGroup(Text("E-commerce", font_size=16, color=WHITE), Text("90.0%", font_size=16, color=BLUE), Text("85.0%", font_size=16, color=PURPLE)).arrange(RIGHT, buff=1.5),
            VGroup(Text("Supermercado", font_size=16, color=WHITE), Text("92.5%", font_size=16, color=BLUE), Text("87.5%", font_size=16, color=PURPLE)).arrange(RIGHT, buff=1.5)
        ).arrange(DOWN, buff=0.6)
        tabla_datos.move_to(UP * 0.2)
        
        conclusiones = VGroup(
            Text("• Supermercados: mayor eficiencia paralela", font_size=16, color=GREEN),
            Text("• PSO consistentemente más eficiente que GA", font_size=16, color=GREEN),
            Text("• Excelente aprovechamiento de 4 procesadores", font_size=16, color=GREEN)
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        conclusiones.move_to(DOWN * 1.5)
        
        self.play(Write(eficiencia_titulo), run_time=2)
        self.wait(0.5)
        
        for fila in tabla_datos:
            self.play(Write(fila), run_time=1.5)
            self.wait(0.4)
        
        self.wait(1)
        
        for conclusion in conclusiones:
            self.play(Write(conclusion), run_time=1.2)
            self.wait(0.3)
        
        self.wait(3)
        
        # FADE OUT FINAL
        self.play(FadeOut(VGroup(titulo, eficiencia_titulo, tabla_datos, conclusiones)), run_time=2)

class ConvergenciaProfesional(Scene):
    def construct(self):
        # === TÍTULO PRINCIPAL ===
        titulo = Text("Análisis de Convergencia", font_size=40, color=BLUE, weight=BOLD)
        titulo.to_edge(UP, buff=0.5)
        
        self.play(Write(titulo), run_time=2)
        self.wait(1)
        
        # === PANTALLA 1: SOLO GRÁFICO DE CONVERGENCIA ===
        axes = Axes(
            x_range=[0, 60, 10],
            y_range=[0, 90, 20],
            x_length=9,
            y_length=4.5,
            axis_config={
                "color": WHITE,
                "stroke_width": 2,
                "include_numbers": True,
                "font_size": 12
            }
        )
        axes.move_to(UP * 0.3)
        
        # Etiquetas de ejes
        x_label = Text("Iteraciones", font_size=14, color=WHITE)
        x_label.next_to(axes, DOWN, buff=0.5)
        
        y_label = Text("Fitness (%)", font_size=14, color=WHITE)
        y_label.rotate(PI/2)
        y_label.next_to(axes, LEFT, buff=0.5)
        
        # Generar curvas de convergencia
        def generar_curva_convergencia(max_iter, max_fitness, velocidad):
            x_vals = np.linspace(0, max_iter, 40)
            y_vals = max_fitness * (1 - np.exp(-x_vals/velocidad))
            ruido = np.random.normal(0, max_fitness*0.015, len(y_vals))
            y_vals = np.clip(y_vals + ruido, 0, max_fitness)
            return x_vals, y_vals
        
        # Datos exactos del paper
        curvas_data = {
            "E-comm PSO": {"iter": 52, "fitness": 74.9, "vel": 18, "color": BLUE},
            "E-comm GA": {"iter": 45, "fitness": 71.2, "vel": 15, "color": PURPLE},
            "Super PSO": {"iter": 48, "fitness": 82.3, "vel": 16, "color": GREEN},
            "Super GA": {"iter": 42, "fitness": 79.8, "vel": 14, "color": ORANGE}
        }
        
        # Crear curvas
        curvas = VGroup()
        for nombre, data in curvas_data.items():
            x_vals, y_vals = generar_curva_convergencia(data["iter"], data["fitness"], data["vel"])
            
            puntos = [axes.coords_to_point(x, y, 0) for x, y in zip(x_vals, y_vals)]
            curva = VMobject()
            curva.set_points_smoothly(puntos)
            curva.set_color(data["color"])
            curva.set_stroke_width(3)
            
            curvas.add(curva)


        # Leyenda compacta SIN marco
        leyenda_items = VGroup()
        nombres = ["E-comm PSO", "E-comm GA", "Super PSO", "Super GA"]
        colores = [BLUE, PURPLE, GREEN, ORANGE]

        for nombre, color in zip(nombres, colores):   
            item = VGroup(
            Line(ORIGIN, RIGHT*0.4, color=color, stroke_width=3),
            Text(nombre, font_size=11, color=WHITE)
            ).arrange(RIGHT, buff=0.15)
            leyenda_items.add(item)

        leyenda_items.arrange(DOWN, buff=0.15)
        leyenda_items.to_corner(UR, buff=0.8)
        
        # CREAR GRÁFICO COMPLETO
        self.play(Create(axes), run_time=2)
        self.wait(0.5)
        self.play(Write(x_label), Write(y_label), run_time=1.5)
        self.wait(0.5)
        self.play(Write(leyenda_items), run_time=1.5)
        self.wait(0.5)
        
        # Crear curvas una por una
        for i, (curva, (nombre, data)) in enumerate(zip(curvas, curvas_data.items())):
            self.play(Create(curva), run_time=2)
            
            # Punto de convergencia temporal
            punto_conv = Dot(
                axes.coords_to_point(data["iter"], data["fitness"], 0),
                color=data["color"],
                radius=0.06
            )
            
            label_conv = Text(f"{data['iter']}i", font_size=10, color=data["color"])
            label_conv.next_to(punto_conv, UP, buff=0.05)
            
            self.play(Create(punto_conv), Write(label_conv), run_time=0.8)
            self.wait(0.5)
            self.play(FadeOut(punto_conv), FadeOut(label_conv), run_time=0.3)
        
        self.wait(2)
        
        # === BORRAR TODO COMPLETAMENTE ===
        todo_grafico = VGroup(axes, x_label, y_label, curvas, leyenda_items)
        self.play(FadeOut(todo_grafico), run_time=2)
        self.wait(0.5)
        
        # === PANTALLA 2: SOLO CONCLUSIONES - PANTALLA COMPLETAMENTE LIMPIA ===
        analisis_titulo = Text("Conclusiones de Convergencia", font_size=36, color=YELLOW, weight=BOLD)
        analisis_titulo.move_to(UP * 2.5)
        
        # Tabla de resultados
        tabla_titulo = Text("Resultados de Convergencia:", font_size=24, color=WHITE, weight=BOLD)
        tabla_titulo.move_to(UP * 1.5)
        
        tabla_convergencia = VGroup(
            VGroup(
                Text("E-commerce PSO: 52 iteraciones", font_size=20, color=BLUE),
                Text("→ 74.9% mejora", font_size=18, color=BLUE)
            ).arrange(RIGHT, buff=0.5),
            VGroup(
                Text("E-commerce GA: 45 iteraciones", font_size=20, color=PURPLE),
                Text("→ 71.2% mejora", font_size=18, color=PURPLE)
            ).arrange(RIGHT, buff=0.5),
            VGroup(
                Text("Supermercado PSO: 48 iteraciones", font_size=20, color=GREEN),
                Text("→ 82.3% mejora", font_size=18, color=GREEN)
            ).arrange(RIGHT, buff=0.5),
            VGroup(
                Text("Supermercado GA: 42 iteraciones", font_size=20, color=ORANGE),
                Text("→ 79.8% mejora", font_size=18, color=ORANGE)
            ).arrange(RIGHT, buff=0.5)
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        tabla_convergencia.move_to(UP * 0.2)
        
        # Conclusiones principales
        conclusiones = VGroup(
            Text("• Supermercados convergen 4-10 iteraciones más rápido", font_size=18, color=WHITE),
            Text("• PSO alcanza mejor fitness final que GA", font_size=18, color=WHITE),
            Text("• Estructura de datos influye en velocidad de convergencia", font_size=18, color=WHITE)
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        conclusiones.move_to(DOWN * 1.8)
        
        # CREAR CONCLUSIONES COMPLETAS
        self.play(Write(analisis_titulo), run_time=2)
        self.wait(0.5)
        
        self.play(Write(tabla_titulo), run_time=1.5)
        self.wait(0.5)
        
        for fila in tabla_convergencia:
            self.play(Write(fila), run_time=1.5)
            self.wait(0.4)
        
        self.wait(1)
        
        for conclusion in conclusiones:
            self.play(Write(conclusion), run_time=1.5)
            self.wait(0.4)
        
        self.wait(3)
        
        # === FADE OUT FINAL ===
        todo_final = VGroup(titulo, analisis_titulo, tabla_titulo, tabla_convergencia, 
                        conclusiones)
        self.play(FadeOut(todo_final), run_time=2)

class ConclusionesProfesionales(Scene):
    def construct(self):
        # Título principal
        titulo = Text("Conclusiones y Hallazgos", font_size=42, color=BLUE, weight=BOLD)
        titulo.to_edge(UP, buff=0.8)
        
        underline = Line(LEFT * 3, RIGHT * 3, color=BLUE, stroke_width=4)
        underline.next_to(titulo, DOWN, buff=0.2)
        
        self.play(Write(titulo), Create(underline), run_time=2)
        self.wait(1)
        
        # Introducción
        intro = Text(
            "Este estudio reveló diferencias significativas en la efectividad de algoritmos",
            font_size=22,
            color=GRAY
        )
        intro2 = Text(
            "de optimización según el contexto de negocio analizado:",
            font_size=22,
            color=GRAY
        )
        intro_group = VGroup(intro, intro2).arrange(DOWN, buff=0.2)
        intro_group.next_to(underline, DOWN, buff=0.5)
        
        self.play(Write(intro_group), run_time=2.5)
        self.wait(1.5)
        
        self.play(FadeOut(intro_group), run_time=1)
        
        # Conclusiones principales organizadas en tarjetas
        conclusiones_data = [
            {
                "numero": "1",
                "titulo": "Superioridad de Supermercados",
                "puntos": [
                    "82.3% vs 74.9% mejora en rentabilidad",
                    "Estructura de datos más uniforme",
                    "Menor variabilidad competitiva"
                ],
                "color": GREEN
            },
            {
                "numero": "2", 
                "titulo": "Eficiencia de Convergencia",
                "puntos": [
                    "48 vs 52 iteraciones promedio",
                    "Búsqueda más eficiente en supermercados",
                    "Menor complejidad competitiva"
                ],
                "color": YELLOW
            },
            {
                "numero": "3",
                "titulo": "Rendimiento Paralelo",
                "puntos": [
                    "Speedup: 3.7x (Super) vs 3.6x (E-comm)",
                    "Eficiencia: 92.5% vs 90%",
                    "Mejor distribución de carga"
                ],
                "color": PURPLE
            },
            {
                "numero": "4",
                "titulo": "Superioridad de PSO",
                "puntos": [
                    "Consistentemente mejor que GA",
                    "Mayor estabilidad en convergencia",
                    "Mejor adaptación a ambos contextos"
                ],
                "color": ORANGE
            }
        ]
        
        # Mostrar conclusiones en secuencia
        for i, data in enumerate(conclusiones_data):
            # Crear tarjeta
            card = RoundedRectangle(
                width=13, height=2.8, corner_radius=0.3,
                color=data["color"], fill_opacity=0.1, stroke_width=2
            )
            card.move_to(ORIGIN)
            
            # Título de la conclusión
            numero = Circle(radius=0.3, color=data["color"], fill_opacity=0.8)
            numero.move_to(card.get_top() + LEFT * 5 + DOWN * 0.4)
            
            num_text = Text(data["numero"], font_size=24, color=WHITE, weight=BOLD)
            num_text.move_to(numero.get_center())
            
            titulo_conclusion = Text(
                data["titulo"], 
                font_size=24, 
                color=data["color"], 
                weight=BOLD
            )
            titulo_conclusion.next_to(numero, RIGHT, buff=0.5)
            
            # Puntos de la conclusión
            puntos = VGroup()
            for punto in data["puntos"]:
                bullet_point = VGroup(
                    Text("•", font_size=20, color=data["color"]),
                    Text(punto, font_size=18, color=WHITE)
                ).arrange(RIGHT, buff=0.3)
                puntos.add(bullet_point)
            
            puntos.arrange(DOWN, buff=0.25, aligned_edge=LEFT)
            puntos.move_to(card.get_center() + DOWN * 0.3 + LEFT * 1)
            
            # Animaciones
            self.play(Create(card), run_time=1)
            self.play(
                Create(numero),
                Write(num_text),
                Write(titulo_conclusion),
                run_time=1.5
            )
            
            for punto in puntos:
                self.play(Write(punto), run_time=1)
                self.wait(0.2)
            
            self.wait(1.5)
            
            # Fade out para la siguiente conclusión (excepto la última)
            if i < len(conclusiones_data) - 1:
                self.play(
                    FadeOut(VGroup(card, numero, num_text, titulo_conclusion, puntos)),
                    run_time=1
                )
        
        # Transición a implicaciones prácticas
        self.play(
            FadeOut(VGroup(card, numero, num_text, titulo_conclusion, puntos)),
            run_time=1.5
        )
        
        # Implicaciones prácticas
        impl_titulo = Text("Implicaciones Prácticas", font_size=36, color=RED, weight=BOLD)
        impl_titulo.move_to(UP * 2)
        
        impl_box = RoundedRectangle(
            width=14, height=4, corner_radius=0.3,
            color=RED, fill_opacity=0.1, stroke_width=2
        )
        impl_box.move_to(DOWN * 0.5)
        
        implicaciones = VGroup(
            VGroup(
                Text("✓", font_size=24, color=GREEN),
                Text("Supermercados: Implementación más directa y ROI más alto", font_size=20)
            ).arrange(RIGHT, buff=0.3),
            VGroup(
                Text("✓", font_size=24, color=GREEN),
                Text("E-commerce: Requiere personalización por categoría", font_size=20)
            ).arrange(RIGHT, buff=0.3),
            VGroup(
                Text("✓", font_size=24, color=GREEN),
                Text("PSO recomendado como algoritmo principal", font_size=20)
            ).arrange(RIGHT, buff=0.3),
            VGroup(
                Text("✓", font_size=24, color=GREEN),
                Text("4 procesos paralelos: configuración óptima", font_size=20)
            ).arrange(RIGHT, buff=0.3),
            VGroup(
                Text("✓", font_size=24, color=GREEN),
                Text("ROI esperado: 74-82% mejora en rentabilidad", font_size=20, color=YELLOW)
            ).arrange(RIGHT, buff=0.3)
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        implicaciones.move_to(impl_box.get_center())
        
        self.play(Write(impl_titulo), run_time=1.5)
        self.play(Create(impl_box), run_time=1)
        
        for impl in implicaciones:
            self.play(Write(impl), run_time=1.2)
            self.wait(0.3)
        
        self.wait(3)
        
        # Mensaje final
        self.play(
            FadeOut(VGroup(impl_titulo, impl_box, implicaciones)),
            run_time=1.5
        )
        
        mensaje_final = VGroup(
            Text("La implementación exitosa requiere considerar", font_size=28, color=WHITE),
            Text("las características específicas del modelo de negocio", font_size=28, color=YELLOW, weight=BOLD)
        ).arrange(DOWN, buff=0.3)
        mensaje_final.move_to(UP * 1)
        
        creditos = VGroup(
            Text("Gracias por su atención", font_size=32, color=BLUE, weight=BOLD),
            Text("Universidad Nacional del Altiplano", font_size=22, color=GRAY),
            Text("Puno, Perú", font_size=20, color=GRAY)
        ).arrange(DOWN, buff=0.3)
        creditos.move_to(DOWN * 1.5)
        
        self.play(Write(mensaje_final), run_time=2.5)
        self.wait(1)
        self.play(Write(creditos), run_time=2)
        self.wait(4)
        
        # Fade out final
        self.play(
            FadeOut(VGroup(titulo, underline, mensaje_final, creditos)),
            run_time=3
        )

# Comandos para renderizar (usar estos exactos):
# manim -pql video_profesional.py IntroduccionProfesional
# manim -pql video_profesional.py ProblemaRetailProfesional  
# manim -pql video_profesional.py TeoriaMatematicaProfesional
# manim -pql video_profesional.py ResultadosComparativosProfesional
# manim -pql video_profesional.py RendimientoParaleloProfesional
# manim -pql video_profesional.py ConvergenciaProfesional
# manim -pql video_profesional.py ConclusionesProfesionales

# Para alta calidad:
# manim -pqh video_profesional.py [NombreEscena]