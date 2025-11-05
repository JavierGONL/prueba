# -*- coding: utf-8 -*-
"""
Cálculo de omega_2 dado phi y C para el mecanismo de retorno rápido.
Basado en el procedimiento del informe LaTeX.
"""
import numpy as np
from scipy.optimize import root_scalar

def calcular_omega2(phi_deg, C, r, d, R, K, D, m_c, g=9.81):
    """
    Calcula omega_2 tal que la aceleración vertical del martillo satisface:
    C = m_c (g - ddot_yc)
    """
    phi = np.deg2rad(phi_deg)
    # Paso 1: aceleración deseada
    ddot_yc_obj = g - C / m_c

    # Paso 2: cinemática instantánea
    L = np.sqrt(r**2 + d**2 + 2*r*d*np.cos(phi))
    sin_theta = r * np.sin(phi) / L
    theta = np.arcsin(sin_theta)
    beta = np.arcsin((R * np.cos(theta) - D) / K)
    y_c = R * np.sin(theta) - K * np.cos(beta)

    # Paso 3: funciones dependientes de omega_2
    def ecuacion_equilibrio(omega_2):
        omega_1 = omega_2 * r / L * np.cos(phi - theta)
        V_af = omega_2 * r * np.sin(phi - theta)
        alpha_1 = (omega_2**2 * r * np.sin(theta - phi) + 2 * omega_1 * V_af) / L
        # Paso 4: velocidad y aceleración vertical del martillo
        A = D * np.sin(theta) - y_c * np.cos(theta)
        B = D * np.cos(theta) + y_c * np.sin(theta)
        Den = R * np.sin(theta) - y_c
        dot_yc = R * omega_1 * A / Den
        ddot_yc = R * ((alpha_1 * A - omega_1**2 * B - omega_1 * dot_yc * np.cos(theta)) * Den - omega_1 * A * (dot_yc + R * omega_1 * np.cos(theta))) / Den**2
        return ddot_yc - ddot_yc_obj

    # Paso 5: resolver omega_2
    # Supón un rango razonable para omega_2 (ejemplo: 0.1 a 100 rad/s)
    sol = root_scalar(ecuacion_equilibrio, bracket=[0.1, 100], method='brentq')
    if sol.converged:
        return sol.root
    else:
        raise RuntimeError("No se pudo encontrar omega_2 que satisface la ecuación de equilibrio.")

if __name__ == "__main__":
    print("Cálculo de omega_2 dado phi y C")
    # Parámetros fijos del mecanismo
    r = 0.09      # Longitud de la barra motriz O2-A (m)
    d = 0.12      # Distancia entre O1 y O2 (m)
    R = 0.20      # Longitud fija de la barra O1-B (m)
    K = 0.09      # Longitud de la barra BC (m)
    D = 0.15      # Distancia horizontal de O1 a C (m)
    m_c = 0.01    # Masa efectiva del martillo (kg)

    phi_deg = float(input("Ángulo phi [grados]: "))
    C = float(input("Fuerza de contacto C [N]: "))
    omega2 = calcular_omega2(phi_deg, C, r, d, R, K, D, m_c)
    print(f"omega_2 requerido: {omega2:.4f} rad/s")
