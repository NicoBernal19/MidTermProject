# task_c.py
import pygame
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import io
import matplotlib.colors as mcolors

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.env_checker import check_env
    RL_AVAILABLE = True
    GYMNASIUM_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    GYMNASIUM_AVAILABLE = False
    print("RL libraries not available. Using simulation mode.")

# Configuración de Pygame
pygame.init()
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (100, 150, 255)
GREEN = (100, 255, 100)
RED = (255, 100, 100)
ORANGE = (255, 165, 0)
PURPLE = (200, 100, 255)
DARK_BLUE = (50, 100, 200)


class PricingEnvironment(gym.Env):
    """
    Entorno de Reinforcement Learning para pricing dinámico.
    Simula un mercado donde el agente aprende a ajustar precios
    para maximizar ingresos considerando demanda, inventario y elasticidad.

    El agente observa: precio actual, ventas, ingresos, semana e inventario
    El agente puede: aumentar/disminuir precios en diferentes magnitudes
    """
    def __init__(self, product_category="Men's Clothing"):
        super(PricingEnvironment, self).__init__()
        self.product_category = product_category
        self.base_price = self._get_base_price()
        self.current_price = self.base_price
        self.current_week = 0
        self.max_weeks = 52  # un año de simulación
        self.weeks_data = []

        # Límites de precio (50% a 200% del precio base)
        self.min_price_multiplier = 0.5
        self.max_price_multiplier = 2.0

        # Parámetros de demanda
        self.base_demand = self._get_base_demand()
        self.price_elasticity = -1.5  # negativo: cuando sube precio, baja demanda

        # Segmentos de clientes con diferentes sensibilidades al precio
        self.segments = {
            'budget': 0.4,      # 40% buscan precios bajos
            'regular': 0.5,     # 50% son normales
            'premium': 0.1      # 10% pagan más por calidad
        }

        # Definir espacio de acciones: 5 opciones de pricing
        self.action_space = spaces.Discrete(5)

        # Definir espacio de observación: [precio, ventas, ingresos, semana, inventario]
        # Todos normalizados para facilitar el aprendizaje
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 2.0, 10.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.reset()

    def _get_base_price(self):
        """Obtiene el precio base según la categoría del producto"""
        base_prices = {
            "Men's Clothing": 80,
            "Women's Clothing": 100,
            "Accessories": 40,
            "Shoes": 120,
            "Beauty": 60,
            "Electronics": 200,
            "Home Goods": 70
        }
        return base_prices.get(self.product_category, 80)

    def _get_base_demand(self):
        """Obtiene la demanda base según la categoría del producto"""
        base_demands = {
            "Men's Clothing": 200,
            "Women's Clothing": 250,
            "Accessories": 150,
            "Shoes": 100,
            "Beauty": 180,
            "Electronics": 80,
            "Home Goods": 120
        }
        return base_demands.get(self.product_category, 200)

    def reset(self, seed=None, options=None):
        """
        Resetea el entorno a su estado inicial.
        Se llama al comenzar cada episodio de entrenamiento.
        """
        if seed is not None:
            super().reset(seed=seed)
            random.seed(seed)
            np.random.seed(seed)

        self.current_price = self.base_price
        self.current_week = 0
        self.weeks_data = []
        self.inventory = 1000  # inventario inicial

        # Calcular estado inicial
        initial_sales = self._calculate_demand(self.current_price)
        initial_revenue = initial_sales * self.current_price

        self.weeks_data.append({
            'week': 0,
            'price': self.current_price,
            'sales': initial_sales,
            'revenue': initial_revenue,
            'inventory': self.inventory,
            'action': 2  # acción neutral
        })

        info = {}
        return self._get_state(), info

    def _get_state(self):
        """
        Devuelve el estado actual del entorno como observación para el agente.
        Todos los valores están normalizados para facilitar el aprendizaje.
        """
        if len(self.weeks_data) == 0:
            return np.array([self.base_price, 0, 0, 0, 1000], dtype=np.float32)

        current_data = self.weeks_data[-1]
        return np.array([
            current_data['price'] / self.base_price,  # precio normalizado
            current_data['sales'] / self.base_demand,  # ventas normalizadas
            current_data['revenue'] / 10000,  # ingresos normalizados
            self.current_week / self.max_weeks,  # progreso temporal
            current_data['inventory'] / 1000  # inventario normalizado
        ], dtype=np.float32)

    def _calculate_demand(self, price):
        """
        Calcula la demanda basándose en múltiples factores:
        - Elasticidad precio: cuando sube el precio, baja la demanda
        - Estacionalidad: patrones que se repiten en el tiempo
        - Segmentos de clientes: diferentes sensibilidades al precio
        - Ruido aleatorio: variabilidad natural del mercado
        """
        # Elasticidad básica: relación precio-demanda
        price_ratio = price / self.base_price
        demand = self.base_demand * (price_ratio ** self.price_elasticity)

        # Factor estacional (simula temporadas altas/bajas)
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * self.current_week / 26)

        # Ruido aleatorio para simular variabilidad del mercado
        noise = random.uniform(0.9, 1.1)

        # Calcular demanda por segmento de clientes
        segment_demand = 0
        for segment, proportion in self.segments.items():
            segment_price_sensitivity = {
                'budget': 2.0,    # muy sensibles al precio
                'regular': 1.5,   # moderadamente sensibles
                'premium': 0.8    # poco sensibles (priorizan calidad)
            }
            segment_base = demand * proportion
            segment_demand += segment_base * (price_ratio ** -segment_price_sensitivity[segment])

        final_demand = segment_demand * seasonal_factor * noise
        return max(0, int(final_demand))

    def step(self, action):
        """
        Ejecuta una acción (cambio de precio) y devuelve el nuevo estado.
        Esta es la función principal que el agente RL llama en cada paso.

        Returns:
            state: nuevo estado después de la acción
            reward: recompensa obtenida (ingresos principalmente)
            done: si terminó el episodio
            truncated: si se truncó el episodio
            info: información adicional
        """
        if self.current_week >= self.max_weeks:
            return self._get_state(), 0.0, True, False, {}

        # Convertir acción a cambio de precio
        price_changes = {
            0: -0.20,  # bajar 20%
            1: -0.10,  # bajar 10%
            2: 0.00,   # mantener
            3: 0.10,   # subir 10%
            4: 0.20    # subir 20%
        }

        price_change = price_changes.get(action, 0.0)
        new_price = self.current_price * (1 + price_change)

        # Aplicar límites de precio
        new_price = max(self.base_price * self.min_price_multiplier,
                        min(self.base_price * self.max_price_multiplier, new_price))

        self.current_price = new_price
        self.current_week += 1

        # Calcular ventas e ingresos
        sales = self._calculate_demand(new_price)
        revenue = sales * new_price

        # Actualizar inventario
        self.inventory = max(0, self.inventory - sales)

        # Calcular recompensa (lo que el agente intenta maximizar)
        reward = revenue / 1000  # normalizar recompensa

        # Penalizar si se queda sin inventario
        if self.inventory <= 0:
            reward -= 50

        # Guardar datos de la semana
        self.weeks_data.append({
            'week': self.current_week,
            'price': new_price,
            'sales': sales,
            'revenue': revenue,
            'inventory': self.inventory,
            'action': action
        })

        done = self.current_week >= self.max_weeks
        truncated = False
        info = {}

        return self._get_state(), reward, done, truncated, info

    def get_performance_metrics(self):
        """Calcula métricas de desempeño de toda la simulación"""
        if len(self.weeks_data) <= 1:
            return {}

        total_revenue = sum(week['revenue'] for week in self.weeks_data[1:])
        total_sales = sum(week['sales'] for week in self.weeks_data[1:])
        avg_price = np.mean([week['price'] for week in self.weeks_data[1:]])
        price_variance = np.var([week['price'] for week in self.weeks_data[1:]])

        return {
            'total_revenue': total_revenue,
            'total_sales': total_sales,
            'average_price': avg_price,
            'price_variance': price_variance,
            'revenue_per_week': total_revenue / len(self.weeks_data[1:])
        }


class StaticPricingEnvironment:
    """
    Estrategia de pricing estático para comparación.
    Mantiene el precio constante durante todo el año.
    Sirve como baseline para medir la mejora del RL.
    """

    def __init__(self, product_category="Men's Clothing"):
        self.env = PricingEnvironment(product_category)
        self.static_price = self.env.base_price

    def run_simulation(self):
        """Ejecuta la simulación con precio fijo (sin cambios)"""
        if GYMNASIUM_AVAILABLE:
            state, info = self.env.reset()
        else:
            state = self.env.reset()

        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            # Siempre tomar la acción neutral (no cambiar precio)
            action = 2

            if GYMNASIUM_AVAILABLE:
                state, reward, done, truncated, info = self.env.step(action)
            else:
                state, reward, done, info = self.env.step(action)

            total_reward += reward

        return self.env.get_performance_metrics()


class PricingRLManager:
    """
    Maneja el entrenamiento y ejecución de modelos de RL para pricing.
    Puede usar PPO o DQN como algoritmos de aprendizaje.
    """
    def __init__(self):
        self.product_categories = [
            "Men's Clothing", "Women's Clothing", "Accessories",
            "Shoes", "Beauty", "Electronics", "Home Goods"
        ]
        self.current_category = "Men's Clothing"
        self.env = PricingEnvironment(self.current_category)
        self.model = None
        self.training_complete = False
        self.simulation_history = []
        self.static_baseline = None

    def train_model(self, algorithm='PPO', timesteps=10000):
        """
        Entrena el modelo de RL usando el algoritmo seleccionado.

        PPO (Proximal Policy Optimization): Más estable, bueno para principiantes
        DQN (Deep Q-Network): Más clásico, basado en Q-learning

        Si no hay librerías de RL, usa una heurística simulada.
        """
        if not RL_AVAILABLE:
            print("RL libraries not available. Using simulated training.")
            self._simulate_training()
            return

        try:
            if algorithm == 'PPO':
                # PPO es generalmente más estable para problemas continuos
                self.model = PPO('MlpPolicy', self.env, verbose=1,
                                 learning_rate=0.0003, n_steps=2048)
            else:  # DQN
                # DQN funciona bien con acciones discretas
                self.model = DQN('MlpPolicy', self.env, verbose=1,
                                 learning_rate=0.0001, buffer_size=10000)

            # Entrenar el modelo con el número especificado de pasos
            self.model.learn(total_timesteps=timesteps)
            self.training_complete = True
            print(f"✅ Model trained with {algorithm} for {timesteps} timesteps")

        except Exception as e:
            print(f"❌ Training failed: {e}. Using simulated training.")
            self._simulate_training()

    def _simulate_training(self):
        """
        Simula el entrenamiento cuando no hay librerías de RL disponibles.
        Usa una heurística simple: bajar precio si hay inventario alto y ventas bajas,
        subir precio si las ventas son altas y el inventario es bajo.
        """
        print("Simulating RL training...")
        best_revenue = 0
        best_actions = []

        # Probar múltiples episodios y quedarse con la mejor estrategia
        for _ in range(10):
            if GYMNASIUM_AVAILABLE:
                state, info = self.env.reset()
            else:
                state = self.env.reset()

            current_actions = []
            current_revenue = 0
            done = False
            truncated = False

            while not (done or truncated):
                current_sales = self.env.weeks_data[-1]['sales']
                current_inventory = self.env.weeks_data[-1]['inventory']

                # Heurística simple: ajustar precio según ventas e inventario
                if current_sales < self.env.base_demand * 0.7 and current_inventory > 200:
                    action = random.choice([0, 1])  # bajar precio
                elif current_sales > self.env.base_demand * 1.3 and current_inventory < 100:
                    action = random.choice([3, 4])  # subir precio
                else:
                    action = 2  # mantener precio

                action = int(action)

                if GYMNASIUM_AVAILABLE:
                    state, reward, done, truncated, info = self.env.step(action)
                else:
                    state, reward, done, info = self.env.step(action)

                current_actions.append(action)
                current_revenue += reward

            # Guardar la mejor estrategia encontrada
            if current_revenue > best_revenue:
                best_revenue = current_revenue
                best_actions = current_actions

        self.training_complete = True
        self.simulated_policy = best_actions
        print("Simulated training completed")

    def run_trained_policy(self):
        """
        Ejecuta la política entrenada y compara con pricing estático.
        Devuelve métricas de ambas estrategias y el porcentaje de mejora.
        """
        if not self.training_complete:
            return None

        if GYMNASIUM_AVAILABLE:
            state, info = self.env.reset()
        else:
            state = self.env.reset()

        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            # Usar el modelo entrenado o la política simulada
            if RL_AVAILABLE and self.model is not None:
                action, _ = self.model.predict(state, deterministic=True)
                action = int(action) if isinstance(action, (np.ndarray, np.generic)) else action
            else:
                week_idx = min(self.env.current_week, len(self.simulated_policy) - 1)
                action = self.simulated_policy[week_idx] if self.simulated_policy else 2

            if GYMNASIUM_AVAILABLE:
                state, reward, done, truncated, info = self.env.step(action)
            else:
                state, reward, done, info = self.env.step(action)

            total_reward += reward

        # Obtener métricas de la política RL
        rl_metrics = self.env.get_performance_metrics()

        # Comparar con pricing estático
        static_env = StaticPricingEnvironment(self.current_category)
        static_metrics = static_env.run_simulation()

        # Calcular mejora porcentual
        comparison = {
            'rl_metrics': rl_metrics,
            'static_metrics': static_metrics,
            'improvement': ((rl_metrics['total_revenue'] - static_metrics['total_revenue'])
                            / static_metrics['total_revenue'] * 100)
        }

        self.simulation_history.append(comparison)
        return comparison

    def get_performance_plot(self):
        """
        Genera visualizaciones comparativas de RL vs pricing estático.
        Muestra 4 gráficos: ingresos, precios en el tiempo, ventas y métricas de mejora.
        """
        if not self.simulation_history:
            return None

        current_sim = self.simulation_history[-1]
        rl_metrics = current_sim['rl_metrics']
        static_metrics = current_sim['static_metrics']

        # Crear figura con 4 subgráficos
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7))

        # Convertir colores a formato matplotlib
        blue_mpl = tuple(c / 255 for c in BLUE)
        green_mpl = tuple(c / 255 for c in GREEN)
        red_mpl = tuple(c / 255 for c in RED)
        orange_mpl = tuple(c / 255 for c in ORANGE)

        # Gráfico 1: Comparación de ingresos totales
        strategies = ['Static\nPricing', 'RL Dynamic\nPricing']
        revenues = [static_metrics['total_revenue'], rl_metrics['total_revenue']]
        bars = ax1.bar(strategies, revenues, color=[blue_mpl, green_mpl], alpha=0.7)
        ax1.set_title('Total Revenue Comparison', fontsize=10)
        ax1.set_ylabel('Revenue ($)', fontsize=9)
        ax1.tick_params(axis='both', which='major', labelsize=8)
        for bar, revenue in zip(bars, revenues):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
                     f'${revenue / 1000:.0f}K', ha='center', va='bottom', fontsize=8)

        # Gráfico 2: Dinámica de precios en el tiempo
        weeks = [data['week'] for data in self.env.weeks_data[1:]]
        prices = [data['price'] for data in self.env.weeks_data[1:]]
        ax2.plot(weeks, prices, 'b-', linewidth=1.5, label='RL Pricing')
        ax2.axhline(y=static_metrics['average_price'], color='r', linestyle='--',
                    label='Static Price', linewidth=1.5)
        ax2.set_title('Price Dynamics Over Time', fontsize=10)
        ax2.set_xlabel('Week', fontsize=9)
        ax2.set_ylabel('Price ($)', fontsize=9)
        ax2.legend(fontsize=8)
        ax2.tick_params(axis='both', which='major', labelsize=8)
        ax2.grid(True, alpha=0.3)

        # Gráfico 3: Comparación de ventas totales
        sales_data = [static_metrics['total_sales'], rl_metrics['total_sales']]
        bars = ax3.bar(strategies, sales_data, color=[blue_mpl, green_mpl], alpha=0.7)
        ax3.set_title('Total Sales Comparison', fontsize=10)
        ax3.set_ylabel('Units Sold', fontsize=9)
        ax3.tick_params(axis='both', which='major', labelsize=8)
        for bar, sales in zip(bars, sales_data):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     f'{sales / 1000:.1f}K', ha='center', va='bottom', fontsize=8)

        # Gráfico 4: Métricas de mejora
        metrics = ['Revenue\nImprovement', 'Price\nFlexibility']
        values = [current_sim['improvement'], rl_metrics['price_variance'] * 100]
        colors_mpl = [green_mpl if current_sim['improvement'] > 0 else red_mpl, orange_mpl]
        bars = ax4.bar(metrics, values, color=colors_mpl, alpha=0.7)
        ax4.set_title('Performance Metrics', fontsize=10)
        ax4.set_ylabel('Percentage / Variance', fontsize=9)
        ax4.tick_params(axis='both', which='major', labelsize=8)
        for bar, value in zip(bars, values):
            if 'Improvement' in metrics[0]:
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
            else:
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f'{value:.1f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout(pad=2.0)

        # Convertir a superficie de Pygame
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=70, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return pygame.image.load(buf)


# Componentes de UI (Button y Slider)
class Button:
    """Botón interactivo con efecto hover"""
    def __init__(self, x, y, width, height, text, color, hover_color=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color or tuple(min(c + 30, 255) for c in color)
        self.hover = False

    def draw(self, screen, font):
        color = self.hover_color if self.hover else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, BLACK, self.rect, 2, border_radius=8)

        text_surface = font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class Slider:
    """Control deslizante para ajustar timesteps de entrenamiento"""
    def __init__(self, x, y, width, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, width, 10)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.dragging = False

    def draw(self, screen, font):
        pygame.draw.rect(screen, GRAY, self.rect, border_radius=5)
        handle_x = self.rect.x + (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width
        handle_rect = pygame.Rect(handle_x - 8, self.rect.y - 5, 16, 20)
        pygame.draw.rect(screen, BLUE, handle_rect, border_radius=3)
        pygame.draw.rect(screen, BLACK, handle_rect, 1, border_radius=3)

        label_text = font.render(f"{self.label}: {self.value}", True, BLACK)
        screen.blit(label_text, (self.rect.x, self.rect.y - 25))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            handle_x = self.rect.x + (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width
            handle_rect = pygame.Rect(handle_x - 8, self.rect.y - 5, 16, 20)
            if handle_rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            rel_x = event.pos[0] - self.rect.x
            rel_x = max(0, min(rel_x, self.rect.width))
            self.value = int(self.min_val + (rel_x / self.rect.width) * (self.max_val - self.min_val))


class PricingVisualizer:
    """
    Maneja toda la interfaz gráfica de la aplicación de pricing dinámico.
    Permite seleccionar categorías, entrenar modelos RL y visualizar resultados.
    """
    def __init__(self):
        self.rl_manager = PricingRLManager()
        self.current_plot = None
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)

        # Elementos de UI
        self.category_buttons = []
        self.train_button = Button(50, 120, 200, 40, "Train RL Model", GREEN)
        self.run_button = Button(260, 120, 200, 40, "Run Simulation", BLUE)

        # Botones para seleccionar algoritmo de RL
        self.algorithm_buttons = [
            Button(500, 120, 120, 40, "PPO", BLUE),
            Button(630, 120, 120, 40, "DQN", BLUE)
        ]
        self.timesteps_slider = Slider(770, 130, 200, 1000, 50000, 10000, "Training Timesteps")
        self.selected_algorithm = 0  # PPO por defecto

        self._create_category_buttons()

    def _create_category_buttons(self):
        """Crear botones para seleccionar categorías de productos"""
        categories = self.rl_manager.product_categories
        button_width = 140
        start_x, start_y = 50, 200
        spacing = 8

        for i, category in enumerate(categories):
            x = start_x + (i % 3) * (button_width + spacing)
            y = start_y + (i // 3) * 40
            button = Button(x, y, button_width, 35, category.split()[0], BLUE)
            self.category_buttons.append((category, button))

    def train_model(self):
        """Entrena el modelo RL con el algoritmo y timesteps seleccionados"""
        algorithm = "PPO" if self.selected_algorithm == 0 else "DQN"
        self.rl_manager.train_model(algorithm, self.timesteps_slider.value)

    def run_simulation(self):
        """Ejecuta la simulación con el modelo entrenado y genera visualizaciones"""
        result = self.rl_manager.run_trained_policy()
        if result:
            self.current_plot = self.rl_manager.get_performance_plot()
        return result

    def draw_interface(self, screen):
        """Dibuja todos los elementos de la interfaz"""
        # Título
        title = self.font.render("UrbanStyle - Task C: Dynamic Pricing with RL", True, BLACK)
        screen.blit(title, (50, 20))
        subtitle = self.small_font.render("Reinforcement Learning for Optimal Pricing Strategy", True, GRAY)
        screen.blit(subtitle, (50, 60))

        # Controles principales
        self.train_button.draw(screen, self.small_font)
        self.run_button.draw(screen, self.small_font)
        self.timesteps_slider.draw(screen, self.small_font)

        # Botones de algoritmo
        algo_text = self.small_font.render("Algorithm:", True, BLACK)
        screen.blit(algo_text, (500, 90))
        for button in self.algorithm_buttons:
            button.draw(screen, self.small_font)

        # Botones de categorías
        category_text = self.small_font.render("Product Category:", True, BLACK)
        screen.blit(category_text, (50, 150))

        for category, button in self.category_buttons:
            button.draw(screen, self.small_font)
            # Resaltar la categoría seleccionada
            if category == self.rl_manager.current_category:
                pygame.draw.rect(screen, RED, button.rect, 3, border_radius=8)

        # Mostrar gráficos de comparación
        if self.current_plot:
            plot_width = 700
            plot_height = 400
            scaled_plot = pygame.transform.scale(self.current_plot, (plot_width, plot_height))
            screen.blit(scaled_plot, (50, 300))

        # Estado del sistema y resultados
        status_y = 150
        status_color = GREEN if self.rl_manager.training_complete else RED
        status_text = "Model Trained" if self.rl_manager.training_complete else "Model Not Trained"
        status_surface = self.small_font.render(f"Status: {status_text}", True, status_color)
        screen.blit(status_surface, (850, status_y))

        # Mostrar resultados de la simulación si existen
        if self.rl_manager.simulation_history:
            current_result = self.rl_manager.simulation_history[-1]
            improvement = current_result['improvement']

            result_y = status_y + 40
            result_text = self.small_font.render("Simulation Results:", True, BLACK)
            screen.blit(result_text, (850, result_y))

            # Mejora porcentual (verde si es positiva, rojo si es negativa)
            improvement_color = GREEN if improvement > 0 else RED
            improvement_text = self.small_font.render(f"Revenue Improvement: {improvement:.1f}%",
                                                      True, improvement_color)
            screen.blit(improvement_text, (850, result_y + 30))

            # Ingresos de ambas estrategias
            rl_revenue = current_result['rl_metrics']['total_revenue']
            static_revenue = current_result['static_metrics']['total_revenue']

            revenue_text = self.small_font.render(f"RL Revenue: ${rl_revenue / 1000:.0f}K", True, BLACK)
            screen.blit(revenue_text, (850, result_y + 60))

            static_text = self.small_font.render(f"Static Revenue: ${static_revenue / 1000:.0f}K", True, BLACK)
            screen.blit(static_text, (850, result_y + 90))

            # Métricas de ventas
            rl_sales = current_result['rl_metrics']['total_sales']
            static_sales = current_result['static_metrics']['total_sales']

            sales_text = self.small_font.render(f"RL Sales: {rl_sales / 1000:.1f}K units", True, BLACK)
            screen.blit(sales_text, (850, result_y + 120))

            static_sales_text = self.small_font.render(f"Static Sales: {static_sales / 1000:.1f}K units", True, BLACK)
            screen.blit(static_sales_text, (850, result_y + 150))

    def handle_event(self, event):
        """
        Maneja todos los eventos de interacción del usuario.
        Procesa clicks en botones, cambios de categoría y ajustes del slider.
        """
        if self.train_button.handle_event(event):
            self.train_model()

        if self.run_button.handle_event(event):
            self.run_simulation()

        # Cambiar categoría de producto
        for category, button in self.category_buttons:
            if button.handle_event(event):
                self.rl_manager.current_category = category
                self.rl_manager.env = PricingEnvironment(category)
                self.rl_manager.training_complete = False

        # Seleccionar algoritmo de RL
        for i, button in enumerate(self.algorithm_buttons):
            if button.handle_event(event):
                self.selected_algorithm = i
                # Resetear todos los botones a azul
                for btn in self.algorithm_buttons:
                    btn.color = BLUE
                # Resaltar el botón seleccionado
                button.color = DARK_BLUE

        # Ajustar timesteps de entrenamiento
        self.timesteps_slider.handle_event(event)


def main():
    """
    Función principal que inicializa la ventana y ejecuta el loop del juego.
    Maneja la actualización y renderizado de la interfaz de pricing dinámico.
    """
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("UrbanStyle - Task C: Dynamic Pricing with RL")
    clock = pygame.time.Clock()

    visualizer = PricingVisualizer()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            visualizer.handle_event(event)

        screen.fill(WHITE)
        visualizer.draw_interface(screen)
        pygame.display.flip()
        clock.tick(30)  # 30 FPS

    pygame.quit()


if __name__ == "__main__":
    main()