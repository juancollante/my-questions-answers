import os
import time

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "question-0004")
os.makedirs(DATA_DIR, exist_ok=True)


def generar_caso_de_uso_predecir_aprobacion_prestamo():
	"""
	Genera un caso de uso aleatorio (input, output) para la función predecir_aprobacion_prestamo.

	Retorna:
		tuple: (input_data, output_data)
			- input_data (dict): Diccionario con claves 'ruta_csv', 'test_size' y 'random_state'
			- output_data (tuple): (predicciones, accuracy) como lo retornaría predecir_aprobacion_prestamo()
	"""

	rng = np.random.default_rng()

	# -------------------------------------------------
	# 1. Generar datos aleatorios de solicitudes
	# -------------------------------------------------
	n = int(rng.integers(140, 420))

	ingreso_mensual = rng.uniform(800, 25000, size=n).round(2)
	edad = rng.integers(18, 75, size=n)
	score_crediticio = rng.integers(300, 851, size=n)
	deudas = rng.uniform(0, 40000, size=n).round(2)
	monto_solicitado = rng.uniform(1000, 90000, size=n).round(2)

	aprobacion_score = (
		(ingreso_mensual > 5000).astype(int)
		+ (score_crediticio > 650).astype(int)
		+ (deudas < 15000).astype(int)
		+ (monto_solicitado < ingreso_mensual * 12).astype(int)
		+ (edad >= 21).astype(int)
	)
	aprobado = (aprobacion_score >= 4).astype(int)

	df = pd.DataFrame(
		{
			"ingreso_mensual": ingreso_mensual,
			"edad": edad,
			"score_crediticio": score_crediticio,
			"deudas": deudas,
			"monto_solicitado": monto_solicitado,
			"aprobado": aprobado,
		}
	)

	# Introducir valores nulos aleatorios
	num_nulos = max(1, n // 28)
	for columna in ["ingreso_mensual", "edad", "score_crediticio", "deudas", "monto_solicitado"]:
		indices = rng.choice(df.index, size=num_nulos, replace=False)
		df.loc[indices, columna] = np.nan

	# -------------------------------------------------
	# 2. Guardar CSV persistente
	# -------------------------------------------------
	timestamp = int(time.time() * 1000)
	csv_name = f"solicitudes_prestamo_caso_{timestamp}.csv"
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
	# 4. Simular predecir_aprobacion_prestamo()
	# -------------------------------------------------
	df_loaded = pd.read_csv(csv_path)
	df_loaded = df_loaded.dropna()

	X = df_loaded.drop(columns=["aprobado"])
	y = df_loaded["aprobado"]

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=test_size,
		random_state=random_state,
		stratify=y if y.nunique() > 1 else None,
	)

	modelo = LogisticRegression(max_iter=1000, random_state=random_state)
	modelo.fit(X_train, y_train)

	predicciones = modelo.predict(X_test)
	acc = accuracy_score(y_test, predicciones)

	output_data = (predicciones.tolist(), float(acc))

	return input_data, output_data


if __name__ == "__main__":
	entrada, salida = generar_caso_de_uso_predecir_aprobacion_prestamo()

	print("=" * 70)
	print("CASO DE USO GENERADO ALEATORIAMENTE")
	print("=" * 70)
	print("\nINPUT (argumentos para predecir_aprobacion_prestamo):")
	print(f"  ruta_csv: {entrada['ruta_csv']}")
	print(f"  test_size: {entrada['test_size']}")
	print(f"  random_state: {entrada['random_state']}")

	if os.path.exists(entrada["ruta_csv"]):
		df_verify = pd.read_csv(entrada["ruta_csv"])
		print(f"  Datos guardados: OK ({len(df_verify)} registros)")
		print(f"  Columnas: {list(df_verify.columns)}")

	print("\nOUTPUT (predicciones y accuracy):")
	predicciones, accuracy = salida
	print(f"  Numero de predicciones: {len(predicciones)}")
	print(f"  Predicciones (primeras 10): {predicciones[:10]}")
	print(f"  Accuracy: {accuracy:.4f}")
	print("=" * 70)

