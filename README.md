# 🌸 Proyecto de Clasificación de Flores Iris con Machine Learning

Este proyecto personal aplica **algoritmos de Machine Learning supervisado** para clasificar las especies de flores **Iris**: *Setosa*, *Versicolor* y *Virginica*, a partir de medidas de sus pétalos y sépalos.  

Se trabajó con el dataset clásico de **Iris** (UCI Machine Learning Repository), ampliamente utilizado en investigación y enseñanza por su simplicidad y riqueza. El dataset fue obtenido de Kaggle.

---

## 📊 Objetivos

1. Analizar y explorar los datos.
2. Realizar un **Análisis Exploratorio de Datos (EDA)** con visualizaciones para comprender relaciones entre variables.
3. Entrenar, evaluar y comparar **5 modelos de clasificación**.
4. Seleccionar el mejor modelo según métricas de desempeño.
5. Guardar el modelo entrenado para su reutilización.

---

## 📂 Contenido del Repositorio

- `Clasificación de Flores Iris con Machine Learning.ipynb` → Notebook con todo el análisis, entrenamiento y evaluación.
- `/models/` → Carpeta donde se almacenan los modelos entrenados (`.pkl`).
- `Iris.csv` → Dataset utilizado.
- `requirements.txt` → Dependencias necesarias para ejecutar el proyecto.

---

## ⚙️ Tecnologías y Librerías

- **Python 3.9+**
- `pandas` → Manejo de datos.
- `numpy` → Operaciones numéricas.
- `matplotlib`, `seaborn` → Visualización de datos.
- `scikit-learn` → Modelos de machine learning y métricas.
- `joblib` → Guardar y cargar modelos.

---

## 🧪 Modelos de Clasificación Entrenados

Se compararon cinco algoritmos:

1. **Regresión Logística (`LogisticRegression`)**  
   Modelo lineal usado como baseline.

2. **K-Nearest Neighbors (`KNeighborsClassifier`)**  
   Clasificación basada en vecinos más cercanos.

3. **Support Vector Machines (`SVC`, kernel lineal)**  
   Encuentra el hiperplano óptimo de separación.

4. **Random Forest (`RandomForestClassifier`)**  
   Ensamble de múltiples árboles de decisión.

5. **Gradient Boosting (`GradientBoostingClassifier`)**  
   Ensamble secuencial que corrige errores de modelos previos.

---

## 📈 Resultados de la Evaluación

| Modelo              | Accuracy | Observaciones                                                 |
| ------------------- | -------- | ------------------------------------------------------------- |
| Regresión Logística | ~0.97    | Muy buen desempeño global.                                    |
| KNN                 | ~0.97    | Similar a la regresión logística.                             |
| SVM (lineal)        | **1.00** | 🏆 Mejor modelo, clasificación perfecta.                      |
| Random Forest       | 0.90     | Menor desempeño, confusiones entre *Versicolor* y *Virginica*.|
| Gradient Boosting   | 0.90     | Similar a Random Forest.                                      |

👉 El **mejor modelo** fue **SVM lineal**, alcanzando **100% de accuracy** en el conjunto de prueba.

---

## 💾 Guardado del Modelo

El mejor modelo (SVM lineal) fue guardado en la carpeta `/models/` con el nombre:

```bash
models/svm_iris_classifier.pkl
