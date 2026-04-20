# Lab 7 — Regresión Logística y Modelos de Clasificación

**Curso:** Minería de Datos  
**Dataset:** Airbnb listings (`listings.RData`) — 76,246 propiedades, 80 variables  
**Problema:** Clasificación binaria — predecir si una propiedad es **"cara"** (precio > $250) o **"no cara"** (precio ≤ $250)

---

## Contenido del notebook (`main.ipynb`)

### Paso 1 — Creación de variables dicotómicas
Se generaron tres variables binarias a partir de la variable de precio categorizada: `es_cara`, `es_media`, `es_economica`.

### Paso 2 — Conjuntos de entrenamiento y prueba
División 70/30 estratificada con `random_state=42`. Train: 43,905 muestras | Test: 18,817 muestras.

### Paso 3 — Modelo de Regresión Logística (base)
Entrenamiento con `GridSearchCV` (5-fold CV). Mejor configuración: `C=100, penalty=L2, solver=liblinear`. Accuracy en test: **0.7903**.

### Paso 4 — Revisión de Multicolinealidad
Análisis con regularización L1 (coeficientes) y matriz de correlación. Se identificaron correlaciones fuertes entre variables de tamaño del alojamiento (>0.74), sin multicolinealidad severa.

### Paso 5 — Eficiencia del algoritmo
Evaluación de accuracy, matriz de confusión y reporte de clasificación del modelo base sobre el conjunto de prueba.

### Paso 6 — Overfitting y Curvas de Aprendizaje
Error de entrenamiento: 0.2032 | Error de prueba: 0.2100. Diferencia de 0.0069 — sin evidencia de overfitting. Las curvas convergen establemente.

### Paso 7 — Tuneo de Parámetros
`GridSearchCV` con `RepeatedStratifiedKFold` (5×3). Espacio de búsqueda: solvers, penalizaciones L1/L2, valores de C. Mejor configuración tuneada: `C=10, penalty=L1, solver=liblinear`. Accuracy CV: **0.7965**.

### Paso 8 — Matriz de Confusión, Tiempo y Memoria
Profiling del modelo tuneado: tiempo ~0.95s, memoria pico ~11,429 KB. TN=11,187 | FP=1,378 | FN=2,574 | TP=3,678.

### Paso 9 — Selección del Mejor Modelo de Regresión Logística
Comparación de M1 (base) vs M2 (tuneado) usando AIC, BIC, métricas y eficiencia.  
**Ganador: M1 (C=100, L2)** — AIC 17494 vs 17531, tiempo 7.7× menor (0.12s vs 0.92s), métricas equivalentes.

### Paso 10 — Modelos de Clasificación Adicionales
Se entrenaron cuatro clasificadores adicionales con los mismos datos y partición:

| Modelo | Accuracy | F1 "cara" |
|--------|----------|-----------|
| M3 — Árbol de Decisión | 0.7364 | 0.60 |
| M4 — Random Forest (100 árboles) | 0.7967 | 0.67 |
| M5 — Naive Bayes (GaussianNB) | 0.7673 | 0.67 |
| M6 — KNN (k=5) | 0.7737 | 0.64 |

### Paso 11 — Comparación de Eficiencia de Modelos
Análisis de tiempo de procesamiento, memoria y errores de clasificación.  
- Más lento: KNN (2.17s) | Más rápido: Naive Bayes (0.02s)  
- Más errores: Árbol de Decisión (4,969) | Menos errores: Random Forest (3,831)

### Paso 12 — Análisis Comparativo Final
Análisis integral en 6 dimensiones: tiempo, accuracy, métricas por clase, FP/FN, interpretabilidad y robustez.  
**Modelo ganador absoluto: M4 — Random Forest** (accuracy 0.7967, F1 cara 0.67, errores mínimos).  
Excepción: si se prioriza recall de la clase "cara", **Naive Bayes** es preferible (recall 0.70).

---

## Variables predictoras utilizadas

| Variable | Descripción |
|----------|-------------|
| `accommodates` | Capacidad máxima de huéspedes |
| `bathrooms` | Número de baños |
| `bedrooms` | Número de habitaciones |
| `beds` | Número de camas |
| `minimum_nights` | Noches mínimas de reserva |
| `number_of_reviews` | Total de reseñas |
| `review_scores_rating` | Puntuación promedio |
| `reviews_per_month` | Frecuencia de reseñas |
| `room_type_Hotel room` | Dummy: tipo hotel |
| `room_type_Private room` | Dummy: habitación privada |
| `room_type_Shared room` | Dummy: habitación compartida |

## Dependencias

```
pandas, numpy, scikit-learn, statsmodels, seaborn, matplotlib, pyreadr, scipy
```
