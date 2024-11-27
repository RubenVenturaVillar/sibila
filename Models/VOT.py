import os
import json
import pickle
import joblib
import logging
from os.path import join
import numpy as np
import pandas as pd
from .BaseModel import BaseModel

# Configure logging
logging.basicConfig(
    filename="vot_execution_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define the prefix for VOT
PREFIX_OUT_VOT = '{}_{}'  # Model, Dataset

class VOT(BaseModel):
    class VotingModel:
        def __init__(self, models, weights, task):
            self.models = models
            self.weights = weights
            self.task = task

        def predict_proba(self, xts):
            """
            Combina las probabilidades de los modelos base utilizando sus pesos.
            """
            probabilities = None
            for model_name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    try:
                        model_proba = model.predict_proba(xts)
                        weight = self.weights.get(model_name, 1)
                        if probabilities is None:
                            probabilities = weight * model_proba
                        else:
                            probabilities += weight * model_proba
                    except Exception as e:
                        logging.error(f"Error in predict_proba with {model_name}: {e}")

            if probabilities is None:
                raise ValueError("No valid probabilities from base models.")

            return probabilities / probabilities.sum(axis=1, keepdims=True)

        def predict(self, xts):
            """
            Realiza predicciones utilizando votación ponderada para clasificación
            o promedio ponderado para regresión.
            """
            predictions = []
            weights = []

            for model_name, model in self.models.items():
                if hasattr(model, 'predict'):
                    try:
                        predictions.append(model.predict(xts))
                        weights.append(self.weights.get(model_name, 1))
                    except Exception as e:
                        logging.error(f"Error in predict with {model_name}: {e}")

            if not predictions:
                raise ValueError("No valid predictions from base models.")

            predictions = np.array(predictions)
            weights = np.array(weights)

            if self.task == "classification":
                # Weighted voting
                weighted_votes = np.apply_along_axis(
                    lambda x: np.bincount(x, weights=weights, minlength=np.max(x) + 1),
                    axis=0,
                    arr=predictions.astype(int)
                )
                return np.argmax(weighted_votes, axis=0)
            elif self.task == "regression":
                # Weighted averaging
                return np.average(predictions, axis=0, weights=weights)
            else:
                raise ValueError(f"Unsupported task type: {self.task}")

    def __init__(self, io_data, cfg, id_list):
        super().__init__(io_data, cfg, id_list)

        # Ruta para cargar el archivo de configuración
        config_path = os.path.join(
            os.getcwd(), "Common", "Config", "DefaultConfigs", "VOT.json"
        )
        logging.info(f"Cargando configuración desde: {config_path}")
        with open(config_path, 'r') as config_file:
            self.vot_config = json.load(config_file)

        # Inicializar atributos desde el archivo de configuración
        self.trained_models_dir = cfg.get_args()['folder']
        self.task = self.vot_config['type_ml']
        self.remove_outliers = self.vot_config.get('remove_outliers', False)

        # Cargar modelos base y sus datos asociados
        self.models = self.load_models()
        self.model_weights = self.load_interpretability()

        # Usar VotingModel como self.model
        self.model = self.VotingModel(self.models, self.model_weights, self.task)

    def get_prefix(self):
        """
        Devuelve la ruta para guardar los resultados del modelo VOT.
        """
        return join(self.cfg.get_folder(),
                    PREFIX_OUT_VOT.format(self.cfg.get_params()['model'], self.cfg.get_name_dataset()))

    def save_model(self):
        """
        Guarda el modelo VOT en la ruta especificada por el prefijo.
        """
        model_path = self.get_prefix()
        logging.info(f"Guardando modelo en: {model_path}")
        joblib.dump(self, model_path)

    def load_models(self):
        """
        Detecta y carga dinámicamente los modelos base disponibles en la ruta de resultados.
        """
        models = {}
        logging.info(f"Iterando en la ruta de resultados: {self.trained_models_dir}")

        valid_extensions = [".joblib"]  # Excluir .pkl ya que no aportan modelos funcionales

        for file in os.listdir(self.trained_models_dir):
            if any(file.endswith(ext) for ext in valid_extensions):
                model_name = file.split("_")[0]  # Usar el prefijo antes del primer "_"
                model_path = os.path.join(self.trained_models_dir, file)
                logging.info(f"Intentando cargar modelo: {model_name} desde {model_path}")

                try:
                    model_instance = joblib.load(model_path)

                    # Validar si el modelo tiene predict
                    if hasattr(model_instance, 'predict'):
                        models[model_name] = model_instance
                        logging.info(f"Modelo {model_name} cargado y añadido. Clase: {type(model_instance).__name__}")
                    else:
                        logging.warning(f"El modelo {model_name} no tiene implementado el método predict.")
                except Exception as e:
                    logging.error(f"Error al cargar el modelo {model_name}: {e}")

        if not models:
            logging.warning("No se encontraron modelos entrenados con predict en la ruta especificada.")
        return models

    def validate_features(self, xts, model_name, model_instance):
        """
        Valida que las características requeridas por el modelo estén presentes en xts.
        """
        if hasattr(model_instance, 'feature_names_'):  # Algunos modelos almacenan las características esperadas
            expected_features = model_instance.feature_names_
            available_features = xts.columns if isinstance(xts, pd.DataFrame) else []
            missing_features = [f for f in expected_features if f not in available_features]
            if missing_features:
                logging.error(
                    f"Modelo {model_name} requiere características faltantes: {missing_features}. "
                    f"Características disponibles: {available_features}."
                )
                raise ValueError(f"Faltan características requeridas para el modelo {model_name}: {missing_features}")

    def predict(self, xts, idx_xts=None):
        """
        Realiza predicciones utilizando los modelos base y sus pesos.
        """
        logging.info("Iniciando predicción con VOT.")

        # Validar características para cada modelo antes de predecir
        for model_name, model_instance in self.models.items():
            try:
                self.validate_features(xts, model_name, model_instance)
            except Exception as e:
                logging.warning(f"Validación de características fallida para {model_name}: {e}")

        # Realizar predicción
        predictions = self.model.predict(xts)
        logging.info(f"Predicciones generadas: {predictions}")
        return predictions

    def load_interpretability(self):
        """
        Carga los pesos de interpretabilidad desde archivos _PermutationImportance.csv.
        """
        weights = {}
        for model_name in self.models.keys():
            filepath = os.path.join(self.trained_models_dir, f"{model_name}_PermutationImportance.csv")
            if os.path.exists(filepath):
                try:
                    importance_data = pd.read_csv(filepath)
                    if 'attribution' in importance_data.columns:
                        weights[model_name] = importance_data['attribution'].mean()
                        logging.info(f"Pesos cargados para {model_name}: {weights[model_name]}")
                    else:
                        logging.warning(f"La columna 'attribution' no se encontró en {filepath}. Usando peso por defecto.")
                        weights[model_name] = 1
                except Exception as e:
                    logging.error(f"Error al procesar interpretabilidad para {model_name}: {e}")
                    weights[model_name] = 1  # Peso por defecto
            else:
                logging.warning(f"No se encontró interpretabilidad para {model_name}. Usando peso por defecto.")
                weights[model_name] = 1

        # Normalizar pesos
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            logging.warning("No se encontraron pesos de interpretabilidad válidos. Usando pesos iguales por defecto.")
            weights = {k: 1 / len(self.models) for k in self.models.keys()}
        return weights

    def train(self, xtr, ytr):
        """
        Verifica que los modelos base estén disponibles para el entrenamiento de VOT.
        """
        if xtr is None or ytr is None or len(xtr) == 0 or len(ytr) == 0:
            logging.error("Los datos de entrenamiento (xtr, ytr) no son válidos.")
            raise ValueError("Los datos de entrenamiento no pueden estar vacíos.")
    
        logging.info(f"Dimensiones de xtr: {xtr.shape if hasattr(xtr, 'shape') else len(xtr)}")
        logging.info(f"Dimensiones de ytr: {len(ytr)}")
    
        logging.info(f"Modelos base cargados: {list(self.models.keys())}")
        if not self.models:
            raise ValueError("No se han cargado modelos base. Verifique el directorio y la configuración.")

        # Si necesitas entrenar los modelos base aquí, añade lógica para hacerlo.
        logging.info("VOT está listo para entrenar utilizando los modelos base.")
    

def predict(self, xts):
    """
    Realiza predicciones utilizando votación ponderada para clasificación
    o promedio ponderado para regresión.
    """
    predictions = []
    weights = []

    for model_name, model in self.models.items():
        if hasattr(model, 'predict'):
            try:
                predictions.append(model.predict(xts))
                weights.append(self.weights.get(model_name, 1))
            except Exception as e:
                print(f"Error in predict with {model_name}: {e}")

    if not predictions:
        raise ValueError("No valid predictions from base models.")

    predictions = np.array(predictions)
    weights = np.array(weights)

    if self.task == "classification":
        # Identificar el rango de clases posibles
        n_classes = int(np.max(predictions)) + 1

        # Votación ponderada con dimensiones consistentes
        weighted_votes = np.apply_along_axis(
            lambda x: np.bincount(
                x, weights=weights, minlength=n_classes
            ),
            axis=0,
            arr=predictions.astype(int)
        )
        return np.argmax(weighted_votes, axis=0)

    elif self.task == "regression":
        # Promedio ponderado para regresión
        return np.average(predictions, axis=0, weights=weights)

    else:
        raise ValueError(f"Unsupported task type: {self.task}")
