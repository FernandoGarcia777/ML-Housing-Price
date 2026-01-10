# 🏡 Predicción de Precios de Viviendas (Housing)



El objetivo principal es demostrar el uso avanzado de **Scikit-Learn Pipelines**, **Ingeniería de Características** y **Transformadores Personalizados**.

## 🛠 Tecnologías Utilizadas

* **Python** (3.x)
* **Scikit-Learn:** Pipelines, ColumnTransformer, RandomForest, SVM.
* **Pandas & NumPy:** Manipulación y limpieza de datos.
* **Matplotlib & Seaborn:** Visualización y Análisis Exploratorio de Datos (EDA).

## 🚀 Características del Proyecto

Este notebook no es solo un modelo, incluye el ciclo de vida completo de los datos:

1.  **Análisis Exploratorio (EDA):** Visualización de distribuciones geográficas y correlaciones.
2.  **Limpieza de Datos:** Imputación de valores faltantes (SimpleImputer).
3.  **Ingeniería de Características (Feature Engineering):**
    * Creación de nuevas variables (ej. `habitaciones_por_hogar`).
    * **Clustering Geoespacial:** Implementación de una clase personalizada `ClusterSimilarity` para agrupar distritos por cercanía geográfica usando K-Means, lo que mejoró significativamente el modelo.
4.  **Transformación:** Manejo de variables categóricas (OneHotEncoding) y escalado numérico (StandardScaler) dentro de un Pipeline unificado.
5.  **Selección de Modelos:** Comparación entre Regresión Lineal, Árboles de Decisión y Random Forest.
6.  **Afinamiento (Fine-Tuning):** Búsqueda de hiperparámetros usando `GridSearchCV` y `RandomizedSearchCV`.

## 🧠 Snippet de Código Destacado

Implementación de un Transformador Personalizado compatible con Pipelines de Scikit-Learn para manejar la similitud geográfica:

```python
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, input_features=None):
        return [f"Cluster_{i}_similarity" for i in range(self.n_clusters)]
