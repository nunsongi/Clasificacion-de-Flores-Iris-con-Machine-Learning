# 🌸 Proyecto de Clasificación de Flores Iris con Machine Learning

Este proyecto aplica **algoritmos de Machine Learning supervisado** para clasificar las especies de flores **Iris**: *Setosa*, *Versicolor* y *Virginica*, a partir de medidas de sus pétalos y sépalos.  

Se trabajó con el dataset clásico de **Iris Species** (UCI Machine Learning Repository), ampliamente utilizado en investigación y enseñanza por su simplicidad y riqueza. El dataset fue obtenido de Kaggle.

---

### Objetivo
Entrenar modelos de clasificación que predigan correctamente la especie de una flor basándose en el largo y ancho de sus sépalos y pétalos.

### Dataset
Contiene 150 registros con las siguientes características:
- Longitud y ancho del sépalo (cm)
- Longitud y ancho del pétalo (cm)
- Especie (etiqueta)

---

## 📂 Contenido del Repositorio

- `Clasificación de Flores Iris con Machine Learning.ipynb` → Notebook con Análisis Exploratorio de Datos (EDA), entrenamiento de modelos y evaluación.
- `models` → Carpeta donde se almacenan los modelos entrenados (`.pkl`).
- `.gitignore` → Ignora la carpeta del entorno virtual (`.venv/`).
- `Iris.csv` → Dataset utilizado.
- `README.md` → Presentación del proyecto.
- `requirements.txt` → Dependencias necesarias para ejecutar el proyecto.

---

## ⚙️ Tecnologías y Librerías

- **Python 3.9+**
- `pandas` → Manejo de datos.
- `numpy` → Operaciones numéricas. (aunque no fue usado en este notebook)
- `matplotlib`, `seaborn` → Visualización de datos.
- `scikit-learn` → Modelos de machine learning y métricas.
- `joblib` → Guardar y cargar modelos.

---

## 🧪 Modelos de Clasificación Entrenados

Se compararon cinco algoritmos:

1. **Regresión Logística (`LogisticRegression`)**  
   Modelo lineal muy usado como baseline.

2. **K-Nearest Neighbors (`KNeighborsClassifier`)**  
   Clasificación basada en vecinos más cercanos.

3. **Support Vector Machines (`SVC(kernel="linear"`)**  
   Encuentra el hiperplano óptimo de separación.

4. **Random Forest (`RandomForestClassifier`)**  
   Ensamble de múltiples árboles de decisión.

5. **Gradient Boosting (`GradientBoostingClassifier`)**  
   Ensamble secuencial que corrige errores de modelos previos.

---

## 📈 Resultados de la Evaluación

| Modelo              | Accuracy | Observaciones                                                     |
| ------------------- | -------- | ----------------------------------------------------------------  |
| Regresión Logística |   93%    | Muy buen rendimiento, solo 1 error entre Versicolor y Virginica.  |                                    
| KNN(k=5)            |   93%    | Buen rendimiento, pero confunde más entre Virginica y Versicolor. |                                
| SVM (lineal)        | **100%** | Clasifica perfectamente todas las especies.                       |
| Random Forest       |   90%    | Correcto, pero menor precisión en Virginica.                      |
| Gradient Boosting   |   90%    | Similar a Random Forest, confusión en especies similares.         |                                         

SVM lineal es el modelo ganador en Iris, alcanzando **100% de accuracy** en el conjunto de prueba.

---

## 💾 Guardado del Modelo

El mejor modelo (SVM lineal) fue guardado en la carpeta `/models/` con el nombre:

```bash
models/svm_iris_classifier.pkl
