# 🧠 Q-Learning en FrozenLake ❄️ | Proyecto de Aprendizaje por Refuerzo

Este proyecto implementa el algoritmo de **Q-Learning** en el entorno FrozenLake utilizando Gymnasium.

El agente aprende a llegar desde el estado inicial hasta la meta evitando agujeros peligrosos en el hielo.

---

# 🚀 Algoritmo utilizado

👉 Q-learning (Reinforcement Learning sin modelo)

:contentReference[oaicite:0]{index=0}  

El aprendizaje se basa en la actualización iterativa de la Q-table mediante la ecuación de Bellman:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \big[r + \gamma \max Q(s',a') - Q(s,a)\big]
\]

---

# ⚙️ Parámetros del modelo

- 📌 Alpha (tasa de aprendizaje): 0.1  
- 📌 Gamma (factor de descuento): 0.95  
- 📌 Epsilon inicial: 1.0  
- 📌 Episodios de entrenamiento: 2000  
- 📌 Episodios de evaluación: 500  

---

# 🧠 Estrategia de aprendizaje

El agente utiliza una política ε-greedy:

- 🎲 Exploración: acciones aleatorias  
- 🧭 Explotación: mejor acción conocida  

Epsilon disminuye progresivamente para estabilizar el aprendizaje.

---

# 📊 Resultados

El agente es evaluado sin exploración (modo greedy).

- Métrica: porcentaje de éxito  
- Objetivo: llegar a la meta con recompensa = 1  

---

# 📁 Estructura del proyecto

📦 Desafio_Alura_Aprendizaje_Reforzado
│
├── train.py        → entrenamiento del agente  
├── evaluate.py     → evaluación del agente  
├── q_learning.py   → actualización Q-learning  
├── q_table.npy     → Q-table aprendida  
└── README.md       → documentación del proyecto  

---

# ▶️ Cómo ejecutar el proyecto

## 1. Instalar dependencias
```bash
pip install numpy gymnasium
