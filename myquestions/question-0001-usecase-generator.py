import os
import time

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Compatibilidad con entornos que ejecutan el archivo via exec(code) sin __file__.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data", "question-0001")
os.makedirs(DATA_DIR, exist_ok=True)


def generar_caso_de_uso_predecir_falla_maquinaria():
    """
    Genera un caso de uso aleatorio (input, output) para la funcion predecir_falla_maquinaria.

    Retorna:
        tuple: (input_data, output_data)
            - input_data (dict): Diccionario con claves 'ruta_csv', 'test_size' y 'n_neighbors'
            - output_data (tuple): (predicciones, accuracy)
    """
    rng = np.random.default_rng()

    # -------------------------------------------------
    # 1. Generar datos aleatorios de sensores
    # -------------------------------------------------
    n = int(rng.integers(140, 420))

    temperatura_motor = rng.uniform(55, 130, size=n).round(2)
    vibracion = rng.uniform(0.2, 9.5, size=n).round(3)
    horas_uso = rng.uniform(50, 12000, size=n).round(1)
    presion = rng.uniform(1.0, 12.0, size=n).round(2)
    ciclos_mantenimiento = rng.integers(0, 20, size=n)

    # Regla semi-realista para falla
    score_falla = (
        (temperatura_motor > 95).astype(int)
        + (vibracion > 6.0).astype(int)
        + (horas_uso > 7000).astype(int)
        + (presion > 8.5).astype(int)
        + (ciclos_mantenimiento < 3).astype(int)
    )
    falla = (score_falla >= 3).astype(int)

    df = pd.DataFrame(
        {
            "temperatura_motor": temperatura_motor,
            "vibracion": vibracion,
            "horas_uso": horas_uso,
            "presion": presion,
            "ciclos_mantenimiento": ciclos_mantenimiento,
            "falla": falla,
        }
    )

    # Introducir nulos para validar limpieza de datos
    num_nulos = max(1, n // 30)
    for columna in ["temperatura_motor", "vibracion", "horas_uso", "presion"]:
        indices = rng.choice(df.index, size=num_nulos, replace=False)
        df.loc[indices, columna] = np.nan

    # -------------------------------------------------
    # 2. Guardar CSV persistente
    # -------------------------------------------------
    timestamp = int(time.time() * 1000)
    csv_name = f"maquinaria_caso_{timestamp}.csv"
    csv_path = os.path.join(DATA_DIR, csv_name)
    df.to_csv(csv_path, index=False)

    # -------------------------------------------------
    # 3. Generar input aleatorio
    # -------------------------------------------------
    test_size = float(rng.choice([0.15, 0.2, 0.25, 0.3]))
    n_neighbors = int(rng.integers(3, 11))

    input_data = {
        "ruta_csv": csv_path,
        "test_size": test_size,
        "n_neighbors": n_neighbors,
    }

    # -------------------------------------------------
    # 4. Simular predecir_falla_maquinaria()
    # -------------------------------------------------
    df_loaded = pd.read_csv(csv_path)
    df_loaded = df_loaded.dropna()

    X = df_loaded.drop(columns=["falla"])
    y = df_loaded["falla"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y if y.nunique() > 1 else None,
    )

    # Ajuste defensivo de vecinos para evitar errores de entrenamiento
    k = min(n_neighbors, len(X_train)) if len(X_train) > 0 else 1
    modelo = KNeighborsClassifier(n_neighbors=max(1, k))
    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)
    acc = accuracy_score(y_test, predicciones)

    output_data = (predicciones.tolist(), float(acc))

    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_predecir_falla_maquinaria()

    print("=" * 70)
    print("CASO DE USO GENERADO ALEATORIAMENTE")
    print("=" * 70)
    print("\nINPUT (argumentos para predecir_falla_maquinaria):")
    print(f"  ruta_csv: {entrada['ruta_csv']}")
    print(f"  test_size: {entrada['test_size']}")
    print(f"  n_neighbors: {entrada['n_neighbors']}")

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
