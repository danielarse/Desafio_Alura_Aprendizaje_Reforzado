import numpy as np

# =========================
# Q-LEARNING UPDATE FUNCTION
# =========================
# Implementa la actualización de la Q-table usando la ecuación de Bellman

def update_q(Q, estado, accion, recompensa, nuevo_estado, alpha, gamma):
    Q[estado, accion] = Q[estado, accion] + alpha * (
        recompensa + gamma * np.max(Q[nuevo_estado]) - Q[estado, accion]
    )
    return Q
