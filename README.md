
# Análisis de Sentimientos de Reseñas de Películas (IMDB) con Bi-RNN

## Descripción del Proyecto
Este proyecto implementa un modelo de Red Neuronal Recurrente Bidireccional (Bi-RNN) para clasificar el sentimiento de reseñas de películas (positivas o negativas) del dataset de IMDB. El objetivo es construir un clasificador robusto que pueda predecir con alta precisión el sentimiento de una reseña de texto.

## Dataset
Se utilizó el dataset de reseñas de películas de IMDB, el cual contiene 50,000 reseñas con su respectivo sentimiento (positivo/negativo). El dataset fue cargado desde Google Drive.

## Metodología
El proyecto sigue los siguientes pasos:

1.  **Carga de Datos**: El dataset inicial de IMDB fue cargado utilizando `pandas`.
2.  **Limpieza de Datos**: Se implementó una función `limpieza_dataset` para preprocesar las reseñas, incluyendo:
    *   Eliminación de etiquetas HTML y URLs.
    *   Conversión a minúsculas y manejo de contracciones.
    *   Eliminación de caracteres no alfabéticos.
    *   Eliminación de *stopwords* (excepto negaciones para preservar el sentimiento).
    *   Lemmatización de las palabras.
3.  **Manejo de Duplicados**: Se eliminaron 418 reseñas duplicadas para asegurar la calidad del dataset.
4.  **Balanceo del Dataset**: El dataset fue re-equilibrado para tener un número igual de reseñas positivas y negativas, utilizando un muestreo aleatorio para igualar la cantidad mínima entre ambas clases.
5.  **Visualización**: Se graficó la distribución de sentimientos para verificar el balance del dataset.
6.  **Tokenización y Vectorización**: Las reseñas limpias fueron tokenizadas y convertidas a secuencias numéricas utilizando `Tokenizer` de Keras. Se aplicó `pad_sequences` para estandarizar la longitud de las secuencias a 200 palabras, utilizando `MAX_WORDS = 10000` para el vocabulario.
7.  **Split de Datos**: El dataset fue dividido en conjuntos de entrenamiento (80%), validación (10%) y prueba (10%) para el desarrollo y evaluación del modelo.
8.  **Construcción del Modelo Bi-RNN**: Se definió una función `crear_modelo_birnn` para construir un modelo secuencial con las siguientes capas:
    *   `Embedding`: Para convertir los índices de palabras en vectores densos.
    *   `Bidirectional(SimpleRNN/LSTM/GRU)`: Dos capas bidireccionales con unidades recurrentes (SimpleRNN en este caso) para capturar dependencias en ambas direcciones del texto.
    *   `Dropout`: Para prevenir el sobreajuste.
    *   `Dense`: Capas densas con activación `relu` y una capa de salida con `sigmoid` para clasificación binaria.
9.  **Compilación y Entrenamiento**: El modelo fue compilado con el optimizador `Adam`, `binary_crossentropy` como función de pérdida y métricas de `accuracy` y `AUC`.
    Se utilizó `EarlyStopping`, `ReduceLROnPlateau` y `ModelCheckpoint` como callbacks para mejorar el entrenamiento y guardar el mejor modelo.
10. **Evaluación del Modelo**: El modelo entrenado fue evaluado en el conjunto de prueba, calculando:
    *   Accuracy
    *   Reporte de Clasificación (Precision, Recall, F1-Score)
    *   Matriz de Confusión
    *   Curva ROC y AUC (Area Under the Curve)
    *   Curvas de aprendizaje (pérdida y accuracy por época).

## Resultados
El modelo Bi-RNN SimpleRNN alcanzó los siguientes resultados en el conjunto de prueba:

*   **Accuracy**: 83.83%
*   **Precision**: 0.8376
*   **Recall**: 0.8393
*   **F1-Score**: 0.8384
*   **AUC-ROC**: 0.9067
*   **Umbral Óptimo ROC**: 0.5295

La matriz de confusión y la curva ROC se guardaron como imágenes (`matriz_confusion.png`, `curva_roc.png`). Las curvas de aprendizaje (`curvas_aprendizaje.png`) muestran la evolución de la pérdida y la precisión durante el entrenamiento.

## Tecnologías Utilizadas
*   Python 3.x
*   TensorFlow 2.x / Keras
*   Pandas
*   Numpy
*   NLTK (Natural Language Toolkit)
*   Scikit-learn
*   Matplotlib
*   Seaborn

## Cómo Ejecutar el Proyecto
1.  **Clonar el Repositorio**:
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd <nombre_del_repositorio>
    ```
2.  **Configurar el Entorno**:
    Se recomienda usar un entorno de Colab con GPU activada (Tipo de Entorno de Ejecución: `T4 GPU`).
3.  **Montar Google Drive**: El notebook requiere acceso a Google Drive para cargar el dataset y guardar el modelo.
4.  **Descargar Recursos NLTK**: Las celdas iniciales del notebook descargarán los recursos necesarios de NLTK.
5.  **Ejecutar las Celdas del Notebook**: Ejecutar las celdas en orden para cargar los datos, preprocesarlos, construir, entrenar y evaluar el modelo.

**Nota**: El archivo `Copia de IMDB Dataset.csv` debe estar disponible en la ruta especificada en Google Drive (`/content/drive/MyDrive/Copia de IMDB Dataset.csv`).
