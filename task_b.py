# task_b.py
import pygame
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import io
from datetime import datetime, timedelta
import random
import os
import requests
import zipfile

# Importaciones para LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Configuración Pygame
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


class DataLoader:
    """
    Maneja la carga y preparación de datos de ventas.
    Puede cargar desde CSV o generar datos de muestra si no hay datos disponibles.
    """
    def __init__(self):
        self.data = None
        self.data_loaded = False

    def download_sample_data(self):
        """
        Genera datos de ventas sintéticos cuando no hay datos reales disponibles.
        Crea transacciones diarias con patrones realistas basados en categorías de retail.
        """
        print("Generando datos de muestra...")

        # Crear rango de fechas de dos años
        start_date = '2022-01-01'
        end_date = '2023-12-31'
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Categorías de productos que se venden en la tienda
        categories = [
            "Men's Clothing", "Women's Clothing", "Accessories",
            "Shoes", "Beauty", "Electronics", "Home Goods"
        ]

        sample_data = []
        transaction_id = 1000

        # Generar transacciones para cada día
        for date in dates:
            daily_transactions = random.randint(50, 200)

            for _ in range(daily_transactions):
                category = random.choice(categories)
                customer_id = random.randint(1, 500)

                # Rangos de precios realistas por categoría
                price_ranges = {
                    "Men's Clothing": (25, 150),
                    "Women's Clothing": (30, 200),
                    "Accessories": (10, 80),
                    "Shoes": (40, 120),
                    "Beauty": (15, 100),
                    "Electronics": (50, 300),
                    "Home Goods": (20, 120)
                }

                min_price, max_price = price_ranges[category]
                price = random.uniform(min_price, max_price)
                quantity = random.randint(1, 3)
                total_amount = price * quantity

                # Asignar género basado en la categoría
                if category == "Men's Clothing":
                    gender = "Male"
                elif category == "Women's Clothing":
                    gender = "Female"
                else:
                    gender = random.choice(["Male", "Female"])

                age = random.randint(18, 70)

                sample_data.append({
                    'Transaction ID': transaction_id,
                    'Customer ID': customer_id,
                    'Gender': gender,
                    'Age': age,
                    'Product Category': category,
                    'Price': round(price, 2),
                    'Quantity': quantity,
                    'Total Amount': round(total_amount, 2),
                    'Date': date
                })

                transaction_id += 1

        self.data = pd.DataFrame(sample_data)
        self.data_loaded = True
        return self.data

    def load_kaggle_data(self, filepath):
        """
        Carga datos reales desde un archivo CSV o Excel.
        Valida que tenga las columnas necesarias para el análisis.
        """
        try:
            if filepath.endswith('.csv'):
                self.data = pd.read_csv(filepath)
            else:
                self.data = pd.read_excel(filepath)

            # Verificar que existan las columnas necesarias
            required_columns = ['Date', 'Product Category', 'Quantity', 'Price', 'Total Amount']
            missing_columns = [col for col in required_columns if col not in self.data.columns]

            if missing_columns:
                print(f"Columnas faltantes: {missing_columns}")
                print("Usando datos de muestra...")
                return self.download_sample_data()

            # Convertir fecha a formato datetime
            if 'Date' in self.data.columns:
                self.data['Date'] = pd.to_datetime(self.data['Date'])

            self.data_loaded = True
            print(f"Datos cargados: {len(self.data)} registros")
            return self.data

        except Exception as e:
            print(f"Error cargando datos: {e}")
            print("Usando datos de muestra...")
            return self.download_sample_data()

    def prepare_weekly_data(self):
        """
        Agrupa los datos diarios en ventas semanales por categoría.
        Esto reduce el ruido y hace más manejable el análisis temporal.
        """
        if self.data is None or not self.data_loaded:
            self.download_sample_data()

        # Agrupar transacciones por semana
        self.data['Week'] = self.data['Date'].dt.to_period('W')

        # Agregar métricas por semana y categoría
        weekly_sales = self.data.groupby(['Week', 'Product Category']).agg({
            'Quantity': 'sum',
            'Total Amount': 'sum',
            'Transaction ID': 'count'
        }).reset_index()

        weekly_sales.rename(columns={
            'Transaction ID': 'Transaction_Count',
            'Total Amount': 'Weekly_Revenue',
            'Quantity': 'Weekly_Sales'
        }, inplace=True)

        # Convertir Week a datetime para facilitar manipulación
        weekly_sales['Date'] = weekly_sales['Week'].dt.start_time
        weekly_sales.drop('Week', axis=1, inplace=True)

        return weekly_sales


class LSTMForecaster:
    """
    Implementa predicción de demanda usando redes LSTM.
    LSTM (Long Short-Term Memory) son redes neuronales diseñadas para
    aprender patrones en series temporales y hacer predicciones futuras.
    """
    def __init__(self, sequence_length=8, units=50, dropout_rate=0.2, learning_rate=0.001):
        self.data_loader = DataLoader()
        self.weekly_data = None
        self.models = {}  # un modelo por categoría
        self.scalers = {}  # un scaler por categoría para normalización
        self.predictions = {}
        self.metrics = {}

        # Parámetros del modelo LSTM
        self.sequence_length = sequence_length  # número de semanas pasadas que usa para predecir
        self.units = units  # número de neuronas LSTM
        self.dropout_rate = dropout_rate  # porcentaje de dropout para evitar overfitting
        self.learning_rate = learning_rate  # tasa de aprendizaje del optimizador

        self.categories = [
            "Men's Clothing", "Women's Clothing", "Accessories",
            "Shoes", "Beauty", "Electronics", "Home Goods"
        ]

    def load_data(self, filepath=None):
        """Cargar datos desde archivo o generar muestra"""
        if filepath and os.path.exists(filepath):
            self.data_loader.load_kaggle_data(filepath)
        else:
            self.data_loader.download_sample_data()

        self.weekly_data = self.data_loader.prepare_weekly_data()
        return self.weekly_data

    def prepare_time_series(self, category):
        """
        Prepara los datos para entrenar el LSTM.
        Crea secuencias donde cada punto usa las N semanas anteriores
        para predecir la siguiente semana.
        """
        if self.weekly_data is None:
            self.load_data()

        # Filtrar datos de la categoría específica
        category_data = self.weekly_data[self.weekly_data['Product Category'] == category].copy()
        category_data = category_data.sort_values('Date')

        # Verificar que haya suficientes datos
        if len(category_data) < self.sequence_length + 5:
            print(f"Insuficientes datos para {category}")
            return np.array([]), np.array([]), category_data

        # Extraer ventas semanales
        sales_data = category_data['Weekly_Sales'].values.reshape(-1, 1)

        # Normalizar datos entre 0 y 1 (importante para LSTM)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(sales_data)
        self.scalers[category] = scaler

        # Crear secuencias de entrenamiento
        # Cada X es una secuencia de N semanas, y su Y correspondiente es la siguiente semana
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i - self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)

        # Redimensionar para LSTM: [muestras, pasos temporales, características]
        X = X.reshape(X.shape[0], X.shape[1], 1)

        return X, y, category_data

    def build_lstm_model(self, input_shape):
        """
        Construye la arquitectura de la red LSTM.
        Usa dos capas LSTM con dropout para prevenir overfitting.
        """
        model = Sequential()

        # Primera capa LSTM - procesa la secuencia temporal
        # return_sequences=True porque hay otra capa LSTM después
        model.add(LSTM(units=self.units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(self.dropout_rate))

        # Segunda capa LSTM - extrae características más abstractas
        model.add(LSTM(units=self.units, return_sequences=False))
        model.add(Dropout(self.dropout_rate))

        # Capa de salida - predice un solo valor (ventas de la próxima semana)
        model.add(Dense(units=1))

        # Compilar con optimizador Adam y función de pérdida MSE
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')

        return model

    def train_models(self, epochs=50, batch_size=16):
        """
        Entrena un modelo LSTM separado para cada categoría de producto.
        Esto permite que cada modelo aprenda patrones específicos de su categoría.
        """
        print("Entrenando modelos LSTM...")

        if self.weekly_data is None:
            self.load_data()

        for category in self.categories:
            print(f"Entrenando modelo LSTM para {category}...")

            X, y, category_data = self.prepare_time_series(category)

            # Si no hay suficientes datos, crear datos sintéticos
            if len(X) == 0:
                print(f"No hay suficientes datos para {category}")
                self._create_backup_data(category)
                X, y, category_data = self.prepare_time_series(category)
                if len(X) == 0:
                    continue

            # Dividir datos en entrenamiento (80%) y prueba (20%)
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Construir el modelo
            model = self.build_lstm_model((X_train.shape[1], 1))

            # Early stopping para detener el entrenamiento si no mejora
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            # Entrenar el modelo
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=0
            )

            self.models[category] = model

            # Hacer predicciones y calcular métricas
            self._make_predictions(category, X_test, y_test, category_data)

            print(f"Modelo LSTM para {category} entrenado - MAE: {self.metrics[category]['MAE']:.1f}")

    def _create_backup_data(self, category):
        """
        Crea datos sintéticos con patrones estacionales para categorías
        que no tienen suficientes datos históricos.
        """
        if self.weekly_data is None:
            self.load_data()

        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='W')

        # Ventas base por categoría
        base_sales = {
            "Men's Clothing": 800,
            "Women's Clothing": 1200,
            "Accessories": 400,
            "Shoes": 600,
            "Beauty": 300,
            "Electronics": 200,
            "Home Goods": 350
        }

        synthetic_data = []
        for date in dates:
            sales = base_sales.get(category, 500)
            # Añadir estacionalidad usando función seno (simula temporadas)
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.month / 12)
            noise = random.uniform(0.8, 1.2)  # ruido aleatorio
            weekly_sales = int(sales * seasonal_factor * noise)

            synthetic_data.append({
                'Date': date,
                'Product Category': category,
                'Weekly_Sales': weekly_sales,
                'Weekly_Revenue': weekly_sales * random.uniform(20, 100),
                'Transaction_Count': weekly_sales // 2
            })

        synthetic_df = pd.DataFrame(synthetic_data)

        # Combinar con datos existentes
        if self.weekly_data is not None:
            self.weekly_data = pd.concat([self.weekly_data, synthetic_df], ignore_index=True)
        else:
            self.weekly_data = synthetic_df

    def _make_predictions(self, category, X_test, y_test, category_data):
        """
        Hace predicciones sobre el conjunto de prueba y calcula métricas de error.
        También predice la próxima semana no vista.
        """
        model = self.models[category]
        scaler = self.scalers[category]

        # Predecir en el conjunto de test
        test_predictions_scaled = model.predict(X_test)

        # Desnormalizar para obtener valores reales
        test_predictions = scaler.inverse_transform(test_predictions_scaled.reshape(-1, 1)).flatten()
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Calcular métricas de error
        mae = mean_absolute_error(y_test_actual, test_predictions)  # error absoluto promedio
        rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))  # raíz del error cuadrático medio

        # Predecir la próxima semana usando la última secuencia disponible
        if len(X_test) > 0:
            last_sequence = X_test[-1]
            next_week_prediction_scaled = model.predict(last_sequence.reshape(1, self.sequence_length, 1))
            next_week_prediction = scaler.inverse_transform(next_week_prediction_scaled)[0, 0]
        else:
            next_week_prediction = category_data['Weekly_Sales'].mean() if len(category_data) > 0 else 0

        # Guardar predicciones
        self.predictions[category] = {
            'test_actual': y_test_actual,
            'test_predicted': test_predictions,
            'next_week': max(0, next_week_prediction),  # evitar predicciones negativas
            'last_actual': category_data['Weekly_Sales'].iloc[-1] if len(category_data) > 0 else 0
        }

        # Guardar métricas
        self.metrics[category] = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': np.mean(np.abs((y_test_actual - test_predictions) / np.maximum(y_test_actual, 1))) * 100
        }

    def get_forecast_plot(self, category):
        """
        Genera gráficos usando matplotlib que muestran:
        1. Valores reales vs predicciones en el conjunto de test
        2. Predicción para la próxima semana comparada con la actual
        """
        if category not in self.predictions:
            return None

        pred_data = self.predictions[category]

        plt.figure(figsize=(10, 6))

        # Gráfico 1: Comparación de predicciones vs valores reales
        plt.subplot(2, 1, 1)
        x_points = range(len(pred_data['test_actual']))

        if len(x_points) > 0:
            plt.plot(x_points, pred_data['test_actual'], 'b-', label='Ventas Reales', linewidth=2, alpha=0.8)
            plt.plot(x_points, pred_data['test_predicted'], 'r--', label='Predicciones LSTM', linewidth=2, alpha=0.8)

        plt.title(f'Forecast LSTM para {category} - Conjunto de Test')
        plt.ylabel('Ventas Semanales')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Gráfico 2: Predicción de la próxima semana
        plt.subplot(2, 1, 2)
        weeks = ['Semana Anterior', 'Próxima Semana (Pred)']
        values = [pred_data['last_actual'], pred_data['next_week']]
        colors = ['blue', 'orange']

        bars = plt.bar(weeks, values, color=colors, alpha=0.7)
        plt.title('Predicción LSTM para la Próxima Semana')
        plt.ylabel('Ventas Semanales')

        # Mostrar valores en las barras
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f'{int(value)}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        # Convertir el gráfico matplotlib a una imagen Pygame
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return pygame.image.load(buf)


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
    """Control deslizante para ajustar parámetros del modelo"""
    def __init__(self, x, y, width, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, width, 10)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.dragging = False

    def draw(self, screen, font):
        # Barra de fondo
        pygame.draw.rect(screen, GRAY, self.rect, border_radius=5)

        # Indicador deslizante
        handle_x = self.rect.x + (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width
        handle_rect = pygame.Rect(handle_x - 8, self.rect.y - 5, 16, 20)
        pygame.draw.rect(screen, BLUE, handle_rect, border_radius=3)
        pygame.draw.rect(screen, BLACK, handle_rect, 1, border_radius=3)

        # Etiqueta y valor
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


class LSTMVisualizer:
    """
    Maneja toda la interfaz visual de la aplicación.
    Permite cargar datos, configurar parámetros del modelo, entrenar
    y visualizar las predicciones.
    """
    def __init__(self):
        self.forecaster = LSTMForecaster()
        self.current_category = "Men's Clothing"

        # Parámetros del modelo LSTM
        self.sequence_length = 8
        self.units = 50
        self.dropout_rate = 0.2
        self.learning_rate = 0.001
        self.epochs = 50
        self.batch_size = 16

        self.models_trained = False
        self.current_plot = None
        self.data_loaded = False

        # Elementos de UI
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)

        self.category_buttons = []
        self.load_data_button = Button(50, 120, 200, 40, "Cargar Datos CSV", GREEN)
        self.train_button = Button(260, 120, 200, 40, "Entrenar Modelos LSTM", BLUE)

        # Sliders para ajustar hiperparámetros del LSTM
        self.sequence_slider = Slider(500, 130, 200, 4, 16, 8, "Longitud de Secuencia")
        self.units_slider = Slider(500, 180, 200, 20, 100, 50, "Unidades LSTM")
        self.dropout_slider = Slider(500, 230, 200, 0.1, 0.5, 0.2, "Dropout Rate")
        self.learning_rate_slider = Slider(500, 280, 200, 0.0001, 0.01, 0.001, "Learning Rate")

        self._create_category_buttons()

    def _create_category_buttons(self):
        """Crear botones para seleccionar diferentes categorías de productos"""
        categories = self.forecaster.categories
        button_width = 150
        button_height = 35
        start_x = 50
        start_y = 350
        spacing = 10

        for i, category in enumerate(categories):
            x = start_x + (i % 4) * (button_width + spacing)
            y = start_y + (i // 4) * (button_height + spacing)

            button = Button(x, y, button_width, button_height,
                            category.split()[0], BLUE)
            self.category_buttons.append((category, button))

    def load_data(self):
        """
        Intenta cargar datos desde archivos CSV locales.
        Si no encuentra ninguno, genera datos de muestra.
        """
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

        if csv_files:
            filepath = csv_files[0]
            self.forecaster.load_data(filepath)
            self.data_loaded = True
            print(f"Datos cargados desde: {filepath}")
        else:
            self.forecaster.load_data()
            self.data_loaded = True
            print("Usando datos de muestra generados")

    def train_models(self):
        """
        Entrena todos los modelos LSTM con los parámetros configurados.
        Actualiza los modelos del forecaster con los valores de los sliders.
        """
        if not self.data_loaded:
            self.load_data()

        # Actualizar parámetros del modelo según los sliders
        self.forecaster.sequence_length = self.sequence_slider.value
        self.forecaster.units = self.units_slider.value
        self.forecaster.dropout_rate = self.dropout_slider.value
        self.forecaster.learning_rate = self.learning_rate_slider.value

        self.forecaster.train_models(
            epochs=self.epochs,
            batch_size=self.batch_size
        )
        self.models_trained = True
        self.update_plot()

    def update_plot(self):
        """Actualiza el gráfico de forecast para la categoría seleccionada"""
        if self.models_trained:
            self.current_plot = self.forecaster.get_forecast_plot(self.current_category)

    def draw_interface(self, screen):
        """Dibuja todos los elementos de la interfaz en la pantalla"""
        # Título principal
        title = self.font.render("UrbanStyle - Task B: Demand Forecasting con LSTM", True, BLACK)
        screen.blit(title, (50, 20))

        subtitle = self.small_font.render("Predicción de Ventas Semanales usando Redes Neuronales LSTM", True, GRAY)
        screen.blit(subtitle, (50, 60))

        # Botones de control
        self.load_data_button.draw(screen, self.small_font)
        self.train_button.draw(screen, self.small_font)

        # Sliders para parámetros
        self.sequence_slider.draw(screen, self.small_font)
        self.units_slider.draw(screen, self.small_font)
        self.dropout_slider.draw(screen, self.small_font)
        self.learning_rate_slider.draw(screen, self.small_font)

        # Botones de categorías
        category_title = self.small_font.render("Seleccionar Categoría:", True, BLACK)
        screen.blit(category_title, (50, 320))

        for category, button in self.category_buttons:
            button.draw(screen, self.small_font)
            # Resaltar categoría seleccionada
            if category == self.current_category:
                pygame.draw.rect(screen, RED, button.rect, 3, border_radius=8)

        # Mostrar gráfico de forecast
        if self.current_plot:
            plot_rect = pygame.Rect(50, 450, 700, 400)
            screen.blit(self.current_plot, plot_rect)
        else:
            no_plot_text = self.font.render("Presiona 'Cargar Datos' y 'Entrenar Modelos LSTM' para comenzar", True,
                                            GRAY)
            screen.blit(no_plot_text, (200, 600))

        # Mostrar métricas si los modelos están entrenados
        if self.models_trained and self.current_category in self.forecaster.metrics:
            self._draw_metrics(screen)

        # Información de estado del sistema
        status_text = "Datos Cargados" if self.data_loaded else "Datos No Cargados"
        status_color = GREEN if self.data_loaded else RED
        status_surface = self.small_font.render(f"Estado Datos: {status_text}", True, status_color)
        screen.blit(status_surface, (850, 190))

        model_status = "Modelos Entrenados" if self.models_trained else "Modelos No Entrenados"
        model_color = GREEN if self.models_trained else RED
        model_surface = self.small_font.render(f"Estado Modelos: {model_status}", True, model_color)
        screen.blit(model_surface, (850, 210))

        model_info = self.small_font.render(f"Modelo: LSTM", True, BLACK)
        screen.blit(model_info, (850, 240))

        # Mostrar parámetros actuales del LSTM
        params_title = self.small_font.render("Parámetros LSTM:", True, BLACK)
        screen.blit(params_title, (850, 280))

        params = [
            f"Secuencia: {self.sequence_slider.value}",
            f"Unidades: {self.units_slider.value}",
            f"Dropout: {self.dropout_slider.value}",
            f"Learning Rate: {self.learning_rate_slider.value}"
        ]

        for i, param in enumerate(params):
            param_surface = self.small_font.render(param, True, BLACK)
            screen.blit(param_surface, (850, 310 + i * 25))

    def _draw_metrics(self, screen):
        """
        Dibuja las métricas de evaluación del modelo y recomendaciones.
        Incluye MAE, RMSE, MAPE, predicción de la próxima semana y
        sugerencias de gestión de inventario basadas en la tendencia.
        """
        metrics = self.forecaster.metrics[self.current_category]
        pred_data = self.forecaster.predictions[self.current_category]

        metrics_x = 850
        metrics_y = 420

        # Título de métricas
        metrics_title = self.small_font.render("Métricas de Evaluación:", True, BLACK)
        screen.blit(metrics_title, (metrics_x, metrics_y))

        # Métricas de error del modelo
        metric_labels = [
            f"MAE: {metrics['MAE']:.1f}",  # Mean Absolute Error - error promedio
            f"RMSE: {metrics['RMSE']:.1f}",  # Root Mean Squared Error - penaliza errores grandes
            f"MAPE: {metrics['MAPE']:.1f}%"  # Mean Absolute Percentage Error - error porcentual
        ]

        for i, label in enumerate(metric_labels):
            metric_surface = self.small_font.render(label, True, BLACK)
            screen.blit(metric_surface, (metrics_x, metrics_y + 30 + i * 25))

        # Predicción para la próxima semana
        next_week_y = metrics_y + 120
        next_week_title = self.small_font.render("Predicción Próxima Semana:", True, BLACK)
        screen.blit(next_week_title, (metrics_x, next_week_y))

        pred_value = self.small_font.render(f"{int(pred_data['next_week'])} unidades", True, BLUE)
        screen.blit(pred_value, (metrics_x, next_week_y + 25))

        # Comparación con semana anterior
        comparison = self.small_font.render(
            f"vs. semana anterior: {int(pred_data['last_actual'])} unidades",
            True, GRAY
        )
        screen.blit(comparison, (metrics_x, next_week_y + 50))

        # Análisis de tendencia (si va a subir o bajar)
        trend = pred_data['next_week'] - pred_data['last_actual']
        trend_color = GREEN if trend > 0 else RED
        trend_text = f"Tendencia: {'+' if trend > 0 else ''}{int(trend)} unidades"
        trend_surface = self.small_font.render(trend_text, True, trend_color)
        screen.blit(trend_surface, (metrics_x, next_week_y + 75))

        # Recomendación automática de gestión de inventario
        inventory_y = next_week_y + 110
        inventory_title = self.small_font.render("Recomendación Inventario:", True, BLACK)
        screen.blit(inventory_title, (metrics_x, inventory_y))

        # Decidir recomendación basada en la tendencia
        if trend > 50:
            recommendation = "Aumentar stock en 15-20%"
            rec_color = GREEN
        elif trend > 10:
            recommendation = "Aumentar stock en 5-10%"
            rec_color = GREEN
        elif trend < -50:
            recommendation = "Reducir stock en 15-20%"
            rec_color = RED
        elif trend < -10:
            recommendation = "Reducir stock en 5-10%"
            rec_color = ORANGE
        else:
            recommendation = "Mantener stock actual"
            rec_color = BLUE

        rec_surface = self.small_font.render(recommendation, True, rec_color)
        screen.blit(rec_surface, (metrics_x, inventory_y + 25))

    def handle_event(self, event):
        """Procesa eventos de mouse para todos los elementos interactivos"""
        if self.load_data_button.handle_event(event):
            self.load_data()

        if self.train_button.handle_event(event):
            self.train_models()

        # Cambiar categoría al hacer click en su botón
        for category, button in self.category_buttons:
            if button.handle_event(event):
                self.current_category = category
                self.update_plot()

        # Actualizar valores de los sliders
        self.sequence_slider.handle_event(event)
        self.units_slider.handle_event(event)
        self.dropout_slider.handle_event(event)
        self.learning_rate_slider.handle_event(event)


def main():
    """
    Función principal que inicializa la ventana y ejecuta el loop principal.
    Maneja la actualización y renderizado de la interfaz de forecasting.
    """
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("UrbanStyle - Task B: Demand Forecasting con LSTM")
    clock = pygame.time.Clock()

    visualizer = LSTMVisualizer()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            visualizer.handle_event(event)

        screen.fill(WHITE)
        visualizer.draw_interface(screen)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()