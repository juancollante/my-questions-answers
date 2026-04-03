import os
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "question-0003")
os.makedirs(DATA_DIR, exist_ok=True)


def generar_caso_de_uso_predecir_precio_viviendas():
	"""
	Genera un caso de uso aleatorio (input, output) para la función predecir_precio_viviendas.

	Retorna:
		tuple: (input_data, output_data)
			- input_data (dict): Diccionario con claves 'ruta_csv', 'test_size' y 'random_state'
			- output_data (tuple): (predicciones, mse) como lo retornaría predecir_precio_viviendas()
	"""

	rng = np.random.default_rng()

	# -------------------------------------------------
	# 1. Generar datos aleatorios de viviendas
	# -------------------------------------------------
	n = int(rng.integers(120, 400))

	area_m2 = rng.uniform(30, 250, size=n).round(2)
	habitaciones = rng.integers(1, 7, size=n)
	banos = rng.integers(1, 5, size=n)
	antiguedad = rng.integers(0, 60, size=n)
	zona_indice = rng.uniform(0.5, 3.5, size=n).round(2)

	# Precio con relación semi-realista a las variables
	precio = (
		area_m2 * rng.uniform(1200, 2500)
		+ habitaciones * rng.uniform(8000, 20000)
		+ banos * rng.uniform(12000, 28000)
		- antiguedad * rng.uniform(500, 1500)
		+ zona_indice * rng.uniform(20000, 60000)
		+ rng.normal(0, 15000, size=n)
	).round(2)

	df = pd.DataFrame(
		{
			"area_m2": area_m2,
			"habitaciones": habitaciones,
			"banos": banos,
			"antiguedad": antiguedad,
			"zona_indice": zona_indice,
			"precio": precio,
		}
	)

	# Introducir algunos valores nulos aleatorios para simular datos reales
	num_nulos = max(1, n // 30)
	for columna in ["area_m2", "habitaciones", "banos", "antiguedad", "zona_indice"]:
		indices = rng.choice(df.index, size=num_nulos, replace=False)
		df.loc[indices, columna] = np.nan

	# -------------------------------------------------
	# 2. Guardar CSV persistente
	# -------------------------------------------------
	timestamp = int(time.time() * 1000)
	csv_name = f"viviendas_caso_{timestamp}.csv"
	csv_path = os.path.join(DATA_DIR, csv_name)
	df.to_csv(csv_path, index=False)

	# -------------------------------------------------
	# 3. Generar input aleatorio
	# -------------------------------------------------
	test_size = float(rng.choice([0.15, 0.2, 0.25, 0.3]))
	random_state = int(rng.integers(1, 10000))

	input_data = {
		"ruta_csv": csv_path,
		"test_size": test_size,
		"random_state": random_state,
	}

	# -------------------------------------------------
	# 4. Simular predecir_precio_viviendas()
	# -------------------------------------------------
	df_loaded = pd.read_csv(csv_path)
	df_loaded = df_loaded.dropna()

	X = df_loaded.drop(columns=["precio"])
	y = df_loaded["precio"]

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=test_size,
		random_state=random_state,
	)

	modelo = RandomForestRegressor(
		n_estimators=100,
		random_state=random_state,
	)
	modelo.fit(X_train, y_train)

	predicciones = modelo.predict(X_test)
	mse = mean_squared_error(y_test, predicciones)

	output_data = (predicciones.tolist(), float(mse))

	return input_data, output_data


if __name__ == "__main__":
	entrada, salida = generar_caso_de_uso_predecir_precio_viviendas()

	print("=" * 70)
	print("CASO DE USO GENERADO ALEATORIAMENTE")
	print("=" * 70)
	print("\nINPUT (argumentos para predecir_precio_viviendas):")
	print(f"  ruta_csv: {entrada['ruta_csv']}")
	print(f"  test_size: {entrada['test_size']}")
	print(f"  random_state: {entrada['random_state']}")

	if os.path.exists(entrada["ruta_csv"]):
		df_verify = pd.read_csv(entrada["ruta_csv"])
		print(f"  Datos guardados: OK ({len(df_verify)} registros)")
		print(f"  Columnas: {list(df_verify.columns)}")

	print("\nOUTPUT (predicciones y mse):")
	predicciones, mse = salida
	print(f"  Numero de predicciones: {len(predicciones)}")
	print(f"  Predicciones (primeras 10): {predicciones[:10]}")
	print(f"  MSE: {mse:.4f}")
	print("=" * 70)

