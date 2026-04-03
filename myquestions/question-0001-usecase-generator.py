import numpy as np
import pandas as pd
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Obtener la ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Crear directorio data si no existe
os.makedirs(DATA_DIR, exist_ok=True)


def generar_caso_de_uso_predecir_churn_clientes():
    """
    Genera un caso de uso aleatorio (input, output) para la función predecir_churn_clientes.
    
    Los datos se guardan en archivos CSV persistentes en la carpeta 'data/' del proyecto.
    
    Retorna:
        tuple: (input_data, output_data)
            - input_data (dict): Diccionario con claves 'ruta_csv' y 'test_size'
            - output_data (tuple): (predicciones, accuracy) como lo retornaría predecir_churn_clientes()
    """
    
    # ---------------------------------
    # 1. Generar datos aleatorios
    # ---------------------------------
    n = np.random.randint(100, 300)  # Número de clientes
    
    edad = np.random.randint(18, 70, size=n)
    ingresos = np.random.randint(800, 10000, size=n)
    tiempo_cliente = np.random.randint(1, 120, size=n)
    num_productos = np.random.randint(1, 5, size=n)
    
    # Generar churn con lógica semi-realista
    churn = (
        (tiempo_cliente < 12).astype(int) +
        (num_productos == 1).astype(int) +
        (ingresos < 2000).astype(int)
    )
    churn = (churn >= 2).astype(int)
    
    # ---------------------------------
    # 2. Crear DataFrame
    # ---------------------------------
    df = pd.DataFrame({
        "edad": edad,
        "ingresos": ingresos,
        "tiempo_cliente": tiempo_cliente,
        "num_productos": num_productos,
        "churn": churn
    })
    
    # ---------------------------------
    # 3. Guardar en CSV persistente
    # ---------------------------------
    # Crear nombre único con timestamp y contador
    timestamp = int(time.time() * 1000)  # millisegundos
    csv_filename = f"clientes_caso_{timestamp}.csv"
    csv_path = os.path.join(DATA_DIR, csv_filename)
    
    df.to_csv(csv_path, index=False)
    
    # ---------------------------------
    # 4. Generar input aleatorio
    # ---------------------------------
    test_size = np.random.choice([0.15, 0.2, 0.25, 0.3])
    
    input_data = {
        "ruta_csv": csv_path,
        "test_size": test_size
    }
    
    # ---------------------------------
    # 5. Simular predecir_churn_clientes()
    # ---------------------------------
    # Cargar datos desde CSV
    df_loaded = pd.read_csv(csv_path)
    
    # Manejo de valores nulos (si los hay)
    df_loaded = df_loaded.dropna()
    
    # Separar X e y
    X = df_loaded.drop(columns=["churn"])
    y = df_loaded["churn"]
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=None
    )
    
    # Entrenar modelo
    modelo = LogisticRegression(max_iter=1000, random_state=None)
    modelo.fit(X_train, y_train)
    
    # Realizar predicciones
    predicciones = modelo.predict(X_test)
    
    # Calcular accuracy
    acc = accuracy_score(y_test, predicciones)
    
    output_data = (predicciones.tolist(), float(acc))
    
    # ---------------------------------
    # 6. Retornar input y output
    # ---------------------------------
    return input_data, output_data


# Ejemplo de uso (descomentar para probar)
if __name__ == "__main__":
    # Generar un caso de uso aleatorio
    entrada, salida = generar_caso_de_uso_predecir_churn_clientes()
    
    print("=" * 70)
    print("CASO DE USO GENERADO ALEATORIAMENTE")
    print("=" * 70)
    print("\nINPUT (argumentos para predecir_churn_clientes):")
    print(f"  test_size: {entrada['test_size']}")
    print(f"  ruta_csv: {entrada['ruta_csv']}")
    
    # Verificar datos guardados
    if os.path.exists(entrada['ruta_csv']):
        df_verify = pd.read_csv(entrada['ruta_csv'])
        print(f"  Datos guardados: ✓ OK ({len(df_verify)} registros)")
        print(f"  Columnas: {list(df_verify.columns)}")
    
    print("\nOUTPUT (predicciones y accuracy):")
    predicciones, accuracy = salida
    print(f"  Número de predicciones: {len(predicciones)}")
    print(f"  Predicciones (primeras 10): {predicciones[:10]}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    print("\n" + "=" * 70)
    print(f"Los datos del cliente se encuentran en: {DATA_DIR}/")
    print("=" * 70)
