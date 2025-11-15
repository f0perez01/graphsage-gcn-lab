# Clasificación de Nodos utilizando Redes Neuronales Gráficas (GCN y GraphSAGE)

**Proyecto:** Desafío de clasificación transductiva en el dataset Cora utilizando **Graph Neural Networks (GNNs)** con TensorFlow.

---

## 📌 Descripción General

Este proyecto implementa un sistema de **clasificación de nodos en grafos** aplicado al dataset académico **Cora**, compuesto por papers científicos representados como nodos y sus relaciones de citación como aristas.
El enfoque es **transductivo**, donde el modelo observa toda la estructura del grafo durante el entrenamiento, aunque solo utiliza las etiquetas de una porción de los nodos.

Se implementan dos arquitecturas principales:

* **GCN (Graph Convolutional Network)**
* **GraphSAGE** (versión base con agregación promedio)

Ambas modelos incluyen capas personalizadas en `tensorflow.keras` para realizar la convolución sobre grafos.

---

## ⚙️ Setup del Proyecto

**Requisitos técnicos:**

* Python 3.10+
* TensorFlow 2.16+
* NumPy / Pandas
* NetworkX
* Matplotlib (opcional)
* Scikit-learn

**Instalación:**

```bash
pip install -r requirements.txt
```

**Ejecución del script principal:**

```bash
python main.py
```

---

## 📚 Dataset: Cora

El dataset **Cora** es un estándar para benchmarking en GNNs.
Características principales:

* **2,708 nodos** (papers)
* **7 clases** (temáticas científicas)
* **5,429 aristas dirigidas** (citaciones)
* **1,433 features binarias** por nodo (*bag-of-words*)
* Grafo homogéneo: un único tipo de nodo y un tipo de arista

La tarea consiste en predecir el *subject* de cada paper.

---

## 🧠 Modelos Implementados

### 1. Red Preprocesadora (FFN)

Antes de aplicar la convolución gráfica, ambos modelos utilizan una **Feed Forward Network** que incluye:

* Batch Normalization
* Dense + ReLU
* Dropout **0.3**

Esto permite estabilizar el aprendizaje y reducir sobreajuste.

### 2. GCN — Graph Convolutional Network

Capa GCN implementada manualmente:

* Normalización simétrica del grafo
* Propagación de mensajes utilizando agregación basada en matriz Laplaciana
* Arquitectura simple de 2 capas

### 3. GraphSAGE (Aggregación Promedio)

Versión base:

* Muestreo y agregación **promedio** de vecinos
* Concatenación de embeddings
* Arquitectura de dos capas con activación ReLU

---

## ⚗️ Metodología

### División de Datos

Para asegurar equilibrio entre clases:

* **Entrenamiento:** 50%
* **Prueba:** 50%
* División estratificada por clase

### Técnicas exploradas

* Dropout en FFN
* Potencial *edge dropping* (data augmentation eliminando aristas)
* Posible *feature masking*

---

## 📊 Análisis del Grafo (NetworkX)

El grafo fue analizado antes del entrenamiento para entender su estructura global:

* **Grado promedio:** 4.01
* **Diámetro:** 19
* **Densidad:** 0.000741

Este grafo disperso y profundo es desafiante para modelos que dependen fuertemente de la agregación local.

---

## 🧪 Resultados

### Rendimiento Base (GCN vs GraphSAGE)

| Métrica                            | GCN        | GraphSAGE |
| ---------------------------------- | ---------- | --------- |
| Accuracy Final (Test)              | **0.5761** | 0.4801    |
| Mejor Accuracy durante el training | **0.5783** | 0.4808    |

**Conclusión:**
GCN supera a GraphSAGE en aproximadamente **9.6 puntos porcentuales**, lo cual es una diferencia significativa en un dataset pequeño como Cora.

---

## 🏆 Resultados del Desafío (Versión Mejorada)

Aplicando técnicas adicionales como:

* BatchNorm más agresivo
* Dropout optimizado
* Edge Dropout
* Feature Masking

Se logró aumentar el desempeño significativamente:

* **Accuracy alcanzado:** **83.23%**

Este resultado demuestra la importancia de la regularización y el preprocesamiento al trabajar con grafos dispersos.

---

## 🧩 Estructura del Proyecto

```
.
├── data/
│   └── cora/
├── models/
│   ├── gcn_layer.py
│   ├── graphsage_layer.py
│   └── ffn_preprocessor.py
├── utils/
│   ├── loaders.py
│   ├── graph_ops.py
│   └── metrics.py
├── main.py
├── README.md
└── requirements.txt
```


