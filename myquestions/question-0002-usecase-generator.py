import os
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Compatibilidad con entornos que ejecutan el archivo via exec(code) sin __file__.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data", "question-0002")
os.makedirs(DATA_DIR, exist_ok=True)


def generar_caso_de_uso_detectar_fraude_transacciones():
	"""
	Genera un caso de uso aleatorio (input, output) para la función detectar_fraude_transacciones.

	Retorna:
		tuple: (input_data, output_data)
			- input_data (dict): Diccionario con claves 'ruta_csv', 'test_size' y 'random_state'
			- output_data (tuple): (predicciones, accuracy) como lo retornaría detectar_fraude_transacciones()
	"""

	rng = np.random.default_rng()

	# -------------------------------------------------
	# 1. Generar datos aleatorios de transacciones
	# -------------------------------------------------
	n = int(rng.integers(120, 350))

	monto = rng.uniform(5, 5000, size=n).round(2)
	hora = rng.integers(0, 24, size=n)
	edad_cliente = rng.integers(18, 80, size=n)
	num_intentos_previos = rng.integers(0, 8, size=n)

	fraude = (
		(monto > 2500).astype(int)
		+ (hora <= 5).astype(int)
		+ (num_intentos_previos >= 3).astype(int)
		+ (edad_cliente < 25).astype(int)
	)
	fraude = (fraude >= 2).astype(int)

	df = pd.DataFrame(
		{
			"monto": monto,
			"hora": hora,
			"edad_cliente": edad_cliente,
			"num_intentos_previos": num_intentos_previos,
			"fraude": fraude,
		}
	)

	# Introducir algunos valores nulos de forma aleatoria para simular datos reales
	num_nulos = max(1, n // 25)
	for columna in ["monto", "hora", "edad_cliente", "num_intentos_previos"]:
		indices = rng.choice(df.index, size=num_nulos, replace=False)
		df.loc[indices, columna] = np.nan

	# -------------------------------------------------
	# 2. Guardar CSV persistente
	# -------------------------------------------------
	timestamp = int(time.time() * 1000)
	csv_name = f"transacciones_caso_{timestamp}.csv"
	csv_path = os.path.join(DATA_DIR, csv_name)
	df.to_csv(csv_path, index=False)

	# -------------------------------------------------
	# 3. Generar input aleatorio
	# -------------------------------------------------
	test_size = float(rng.choice([0.2, 0.25, 0.3, 0.35]))
	random_state = int(rng.integers(1, 10000))

	input_data = {
		"ruta_csv": csv_path,
		"test_size": test_size,
		"random_state": random_state,
	}

	# -------------------------------------------------
	# 4. Simular detectar_fraude_transacciones()
	# -------------------------------------------------
	df_loaded = pd.read_csv(csv_path)
	df_loaded = df_loaded.dropna()

	X = df_loaded.drop(columns=["fraude"])
	y = df_loaded["fraude"]

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=test_size,
		random_state=random_state,
		stratify=y if y.nunique() > 1 else None,
	)

	modelo = RandomForestClassifier(
		n_estimators=100,
		random_state=random_state,
	)
	modelo.fit(X_train, y_train)

	predicciones = modelo.predict(X_test)
	acc = accuracy_score(y_test, predicciones)

	output_data = (predicciones.tolist(), float(acc))

	return input_data, output_data


if __name__ == "__main__":
	entrada, salida = generar_caso_de_uso_detectar_fraude_transacciones()

	print("=" * 70)
	print("CASO DE USO GENERADO ALEATORIAMENTE")
	print("=" * 70)
	print("\nINPUT (argumentos para detectar_fraude_transacciones):")
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

