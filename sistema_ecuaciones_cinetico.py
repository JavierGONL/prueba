"""
Sistema de Ecuaciones Cinéticas del Mecanismo de Martillo de Forja
Basado en sistema_ecuaciones.tex e informe dinamica.tex

Este módulo implementa el sistema completo de ecuaciones para el análisis de fuerzas
y momentos en todas las barras del mecanismo.

COMPONENTES DEL MECANISMO:
- Barra R: Palanca oscilante (pivote en O₁)
- Barra r: Manivela motriz (pivote en O₂)
- Barra K: Biela conectora
- Martillo M: Bloque deslizante vertical

SISTEMA DE 11 ECUACIONES:
1-3: Barra R (2 equilibrio de fuerzas + 1 momento)
4-6: Barra r (2 equilibrio de fuerzas + 1 momento)
7-9: Barra K (2 equilibrio de fuerzas + 1 momento)
10-11: Martillo M (2 equilibrio de fuerzas)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from dataclasses import dataclass
from typing import Tuple, Dict, List
import os
from analisis_dependencias import MecanismoCinematico

# Crear carpeta de imágenes si no existe
os.makedirs('imagenes', exist_ok=True)


@dataclass
class ParametrosGeometricos:
    """Parámetros geométricos del mecanismo"""
    R: float  # Longitud barra R (palanca oscilante) [m]
    r: float  # Longitud barra r (manivela) [m]
    K: float  # Longitud barra K (biela) [m]
    d: float  # Distancia entre O₁ y O₂ [m]
    D: float  # Distancia vertical D [m]


@dataclass
class PropiedadesFisicas:
    """Propiedades físicas de las barras"""
    m_R: float  # Masa de la barra R [kg]
    m_r: float  # Masa de la barra r (manivela) [kg]
    m_K: float  # Masa de la barra K (biela) [kg]
    m_M: float  # Masa del martillo M [kg]
    g: float = 9.81  # Aceleración gravitacional [m/s²]


@dataclass
class EstadoCinematico:
    """Estado cinemático del sistema en un instante dado"""
    # Ángulos [rad]
    delta: float      # Ángulo de la manivela (entrada del motor)
    alpha: float      # Ángulo de la barra R (Gamma en las ecuaciones)
    beta: float       # Ángulo de la barra K (biela)
    
    # Longitud variable
    L: float          # Longitud del eslabón virtual O₁-A [m]
    
    # Velocidades angulares [rad/s]
    omega_2: float    # Velocidad angular del motor (constante)
    omega_1: float    # Velocidad angular de la barra R
    omega_K: float    # Velocidad angular de la barra K
    
    # Aceleraciones angulares [rad/s²]
    alpha_2: float    # Aceleración angular del motor (= 0 para velocidad constante)
    alpha_1: float    # Aceleración angular de la barra R
    alpha_K: float    # Aceleración angular de la barra K
    
    # Posición y cinemática del martillo
    X_E: float        # Posición vertical del martillo [m]
    X_E_dot: float    # Velocidad del martillo [m/s]
    X_E_ddot: float   # Aceleración del martillo [m/s²]


class SistemaEcuacionesCinetico:
    """
    Resuelve el sistema completo de ecuaciones cinéticas del mecanismo
    
    ECUACIONES (según sistema_ecuaciones.tex):
    
    BARRA R (Palanca oscilante):
    Eq. 1 (n-R): Equilibrio en dirección normal
    Eq. 2 (t-R): Equilibrio en dirección tangencial
    Eq. 3 (θ-R): Ecuación de momento respecto a G_R
    
    BARRA r (Manivela):
    Eq. 4 (n-r): Equilibrio en dirección normal
    Eq. 5 (t-r): Equilibrio en dirección tangencial
    Eq. 6 (θ-r): Ecuación de momento respecto a G_r → α̈_r = 0
    
    BARRA K (Biela):
    Eq. 7 (x-K): Equilibrio en dirección x
    Eq. 8 (y-K): Equilibrio en dirección y
    Eq. 9 (θ-K): Ecuación de momento respecto a G_K
    
    MARTILLO M:
    Eq. 10 (x-M): M_x = 0
    Eq. 11 (y-M): V_N - M_y = m_M · a_M_y
    """
    
    def __init__(self, geom: ParametrosGeometricos, props: PropiedadesFisicas, omega_2: float):
        self.geom = geom
        self.props = props
        self.omega_2 = omega_2
        
        # Crear instancia del mecanismo cinemático para cálculos geométricos
        self.mecanismo = MecanismoCinematico(geom.r, geom.d, geom.R, geom.K, geom.D, omega_2)
    
    def calcular_pesos(self) -> Tuple[float, float, float, float]:
        """Calcula los pesos de cada componente [N]"""
        W_R = self.props.m_R * self.props.g
        W_r = self.props.m_r * self.props.g
        W_K = self.props.m_K * self.props.g
        W_M = self.props.m_M * self.props.g
        return W_R, W_r, W_K, W_M
    
    def calcular_inercias(self) -> Tuple[float, float, float]:
        """
        Calcula los momentos de inercia respecto al centro de masa
        Para barras delgadas: I = (1/12) · m · L²
        """
        I_R = (1/12) * self.props.m_R * self.geom.R**2
        I_r = (1/12) * self.props.m_r * self.geom.r**2
        I_K = (1/12) * self.props.m_K * self.geom.K**2
        return I_R, I_r, I_K
    
    def calcular_aceleraciones_centros_masa(self, estado: EstadoCinematico) -> Dict:
        """
        Calcula las aceleraciones de los centros de masa de cada barra
        
        Usa el módulo MecanismoCinematico de analisis_dependencias.py para obtener
        las aceleraciones angulares correctas.
        
        Para una barra que rota respecto a un pivote fijo:
        Las ecuaciones en el sistema usan componentes NORMAL y TANGENCIAL
        respecto al eje de la barra:
        
        - Aceleración NORMAL (a_n): componente perpendicular a la barra (centrípeta)
          a_n = ω² · r_cm  donde r_cm es la distancia del pivote al centro de masa
        
        - Aceleración TANGENCIAL (a_t): componente en dirección de la barra
          a_t = α · r_cm  donde α es la aceleración angular
        
        Returns:
            Diccionario con componentes de aceleración en coordenadas n-t y x-y
        """
        # Obtener estado completo del mecanismo usando analisis_dependencias
        t = estado.delta / self.omega_2  # Calcular tiempo desde δ
        estado_mecanismo = self.mecanismo.estado_completo(t)
        
        # Barra R: centro de masa a distancia R/2 del pivote O₁
        # Coordenadas NORMAL-TANGENCIAL (respecto a la barra R con ángulo Γ=α)
        # Usar valores de analisis_dependencias
        omega_1 = estado_mecanismo['omega_1']
        alpha_1 = estado_mecanismo['alpha_ddot']  # Aceleración angular de R
        
        a_R_n = omega_1**2 * (self.geom.R / 2)  # Aceleración NORMAL (centrípeta)
        a_R_t = alpha_1 * (self.geom.R / 2)     # Aceleración TANGENCIAL
        
        # Barra r: centro de masa a distancia r/2 del pivote O₂
        # Coordenadas NORMAL-TANGENCIAL (respecto a la barra r con ángulo δ)
        a_r_n = self.omega_2**2 * (self.geom.r / 2)  # Aceleración NORMAL (centrípeta)
        a_r_t = 0.0  # Aceleración TANGENCIAL = 0 porque α₂ = 0 (motor a velocidad constante)
        
        # Barra K: movimiento general de plano
        # Coordenadas CARTESIANAS (x, y) porque las ecuaciones están en x-y
        # La biela K tiene movimiento de plano general (no solo rotación)
        # El centro de masa de K está a K/2 desde el punto B (extremo de la barra R)
        
        # Obtener ángulos y velocidades angulares
        alpha_mec = estado.alpha
        omega_1_mec = estado.omega_1
        omega_K_mec = estado.omega_K
        alpha_K_mec = estado.alpha_K
        beta_mec = estado.beta
        
        # Aceleración del punto B (extremo de la barra R)
        # x_B = R·cos(α), y_B = R·sin(α)
        # ẍ_B = -R[α̈·sin(α) + ω₁²·cos(α)]
        # ÿ_B = R[α̈·cos(α) - ω₁²·sin(α)]
        x_B_ddot = -self.geom.R * (alpha_1 * np.sin(alpha_mec) + omega_1_mec**2 * np.cos(alpha_mec))
        y_B_ddot = self.geom.R * (alpha_1 * np.cos(alpha_mec) - omega_1_mec**2 * np.sin(alpha_mec))
        
        # Aceleración del centro de masa de K (a K/2 desde B)
        # a_K_x = ẍ_B - (K/2)[β̈·sin(β) + β̇²·cos(β)]
        # a_K_y = ÿ_B + (K/2)[β̈·cos(β) - β̇²·sin(β)]
        a_K_x = x_B_ddot - (self.geom.K/2) * (alpha_K_mec * np.sin(beta_mec) + omega_K_mec**2 * np.cos(beta_mec))
        a_K_y = y_B_ddot + (self.geom.K/2) * (alpha_K_mec * np.cos(beta_mec) - omega_K_mec**2 * np.sin(beta_mec))
        
        return {
            'a_R_n': a_R_n,  # Normal a la barra R
            'a_R_t': a_R_t,  # Tangencial a la barra R
            'a_r_n': a_r_n,  # Normal a la barra r
            'a_r_t': a_r_t,  # Tangencial a la barra r
            'a_K_x': a_K_x,  # Componente x (cartesiana)
            'a_K_y': a_K_y   # Componente y (cartesiana)
        }
    
    def sistema_ecuaciones(self, incognitas: np.ndarray, estado: EstadoCinematico) -> np.ndarray:
        """
        Sistema de 10 ecuaciones con 10 incógnitas
        
        Incógnitas:
        [0] r_x: Fuerza de reacción en punto A (componente x)
        [1] r_y: Fuerza de reacción en punto A (componente y)
        [2] K_x: Fuerza de reacción en punto B (componente x)
        [3] K_y: Fuerza de reacción en punto B (componente y)
        [4] M_x: Fuerza de reacción del martillo (componente x) = 0
        [5] M_y: Fuerza de reacción del martillo (componente y)
        [6] Theta_1_x: Reacción en pivote O₁ (componente x)
        [7] Theta_1_y: Reacción en pivote O₁ (componente y)
        [8] Theta_2_x: Reacción en pivote O₂ (componente x)
        [9] Theta_2_y: Reacción en pivote O₂ (componente y)
        
        Returns:
            Array con los residuos de las 10 ecuaciones
        """
        # Desempaquetar incógnitas (10 incógnitas)
        r_x, r_y, K_x, K_y, M_x, M_y = incognitas[0:6]
        Theta_1_x, Theta_1_y = incognitas[6:8]
        Theta_2_x, Theta_2_y = incognitas[8:10]
        
        # Pesos
        W_R, W_r, W_K, W_M = self.calcular_pesos()
        
        # Inercias
        I_R, I_r, I_K = self.calcular_inercias()
        
        # Aceleraciones
        acels = self.calcular_aceleraciones_centros_masa(estado)
        
        # Ángulos
        Gamma = estado.alpha  # Ángulo de la barra R
        delta = estado.delta  # Ángulo de la manivela r
        beta = estado.beta    # Ángulo de la biela K
        
        # Inicializar array de residuos (10 ecuaciones, 10 incógnitas)
        # M_x será ≈0 por la física del problema (el martillo se mueve solo verticalmente)
        F = np.zeros(10)
        
        # ========== ECUACIONES BARRA R (Palanca oscilante) ==========
        
        # Eq. 1: Dirección n (normal)
        # EXACTA del documento: (W_R + r_y + K_y - Θ₁_y)cos(Γ) - (K_x + r_x - Θ₁_x)sin(Γ) = m_R · a_R_n
        F[0] = ((W_R + r_y + K_y - Theta_1_y) * np.cos(Gamma) 
                - (K_x + r_x - Theta_1_x) * np.sin(Gamma) 
                - self.props.m_R * acels['a_R_n'])
        
        # Eq. 1: Dirección t (tangencial)
        # EXACTA del documento: (Θ₁_y - W_R - r_y - K_y)sin(Γ) - (K_x + r_x - Θ₁_x)cos(Γ) = m_R · a_R_t
        F[1] = ((Theta_1_y - W_R - r_y - K_y) * np.sin(Gamma) 
                - (K_x + r_x - Theta_1_x) * np.cos(Gamma) 
                - self.props.m_R * acels['a_R_t'])
        
        # Eq. 2: Momento respecto a G_R
        # EXACTA del documento (CORREGIDA con valor absoluto):
        # -|R/2 - L|[(r_y·sin(Γ) + cos(Γ)·r_x) - (R/2)(K_y·sin(Γ) + K_x·cos(Γ))]
        # - (R/2)(Θ₁_y·sin(Γ) + Θ₁_x·cos(Γ)) = (1/12)m_R·R²·α_R
        F[2] = (-np.abs(self.geom.R/2 - estado.L) * 
                ((r_y * np.sin(Gamma) + np.cos(Gamma) * r_x) 
                 - (self.geom.R/2) * (K_y * np.sin(Gamma) + K_x * np.cos(Gamma)))
                - (self.geom.R/2) * (Theta_1_y * np.sin(Gamma) + Theta_1_x * np.cos(Gamma))
                - I_R * estado.alpha_1)
        
        # ========== ECUACIONES BARRA r (Manivela) ==========
        
        # Eq. 3: Dirección n (normal)
        # EXACTA del documento: (W_r + Θ₂_y - r_y)cos(δ) + (r_x - Θ₂_x)sin(δ) = m_r · a_r_n
        F[3] = ((W_r + Theta_2_y - r_y) * np.cos(delta) 
                + (r_x - Theta_2_x) * np.sin(delta) 
                - self.props.m_r * acels['a_r_n'])
        
        # Eq. 4: Dirección t (tangencial)
        # EXACTA del documento: (r_y - W_r - Θ₂_y)sin(δ) + (r_x - Θ₂_x)cos(δ) = m_r · a_r_t
        F[4] = ((r_y - W_r - Theta_2_y) * np.sin(delta) 
                + (r_x - Theta_2_x) * np.cos(delta) 
                - self.props.m_r * acels['a_r_t'])
        
        # Eq. 5: Momento respecto a G_r
        # EXACTA del documento: (r/2)[r_x·cos(δ) + r_y·sin(δ) + Θ₂_x·cos(δ) + Θ₂_y·sin(δ)] 
        # = (1/12)m_r·r²·α̈_r = 0  →  α̈_r = 0
        F[5] = ((self.geom.r/2) * 
                (r_x * np.cos(delta) + r_y * np.sin(delta) 
                 + Theta_2_x * np.cos(delta) + Theta_2_y * np.sin(delta))
                - 0.0)  # = 0 porque α̈_r = 0
        
        # ========== ECUACIONES BARRA K (Biela) ==========
        
        # Eq. 6: Dirección x
        # EXACTA del documento: K_x - M_x = m_K · a_K_x
        F[6] = K_x - M_x - self.props.m_K * acels['a_K_x']
        
        # Eq. 7: Dirección y
        # EXACTA del documento: K_y - W_K - M_y = m_K · a_K_y
        F[7] = K_y - W_K - M_y - self.props.m_K * acels['a_K_y']
        
        # Eq. 8: Momento respecto a G_K
        # EXACTA del documento: (K/2)(K_x·cos(β) + K_y·sin(β)) + M_x·cos(β) + M_y·sin(β) 
        # = (1/12)m_K·K²·α̈_K
        F[8] = ((self.geom.K/2) * (K_x * np.cos(beta) + K_y * np.sin(beta))
                + M_x * np.cos(beta) + M_y * np.sin(beta)
                - I_K * estado.alpha_K)
        
        # ========== ECUACIONES MARTILLO M ==========
        
        # Eq. 9: Dirección y del martillo
        # EXACTA del documento: W_M - M_y = m_M · a_M_y
        # donde W_M es el peso del martillo = m_M · g
        # NOTA: M_x no tiene ecuación, pero será ≈0 por diseño (movimiento solo vertical)
        F[9] = W_M - M_y - self.props.m_M * estado.X_E_ddot
        
        return F
    
    def resolver_sistema(self, estado: EstadoCinematico) -> Dict:
        """
        Resuelve el sistema de ecuaciones para un estado cinemático dado
        
        Returns:
            Diccionario con todas las fuerzas de reacción
        """
        # Estimación inicial de las incógnitas
        W_R, W_r, W_K, W_M = self.calcular_pesos()
        
        incognitas_inicial = np.array([
            0.0,    # r_x
            W_r,    # r_y (estimación: peso de r)
            0.0,    # K_x
            W_K,    # K_y (estimación: peso de K)
            0.0,    # M_x (sabemos que es 0)
            W_M,    # M_y (estimación: peso del martillo)
            0.0,    # Theta_1_x
            W_R,    # Theta_1_y (estimación: peso de R)
            0.0,    # Theta_2_x
            W_r     # Theta_2_y (estimación: peso de r)
        ])
        
        # Resolver el sistema no lineal
        solucion = fsolve(self.sistema_ecuaciones, incognitas_inicial, args=(estado,))
        
        # Empaquetar resultados
        return {
            'r_x': solucion[0],
            'r_y': solucion[1],
            'K_x': solucion[2],
            'K_y': solucion[3],
            'M_x': solucion[4],
            'M_y': solucion[5],
            'Theta_1_x': solucion[6],
            'Theta_1_y': solucion[7],
            'Theta_2_x': solucion[8],
            'Theta_2_y': solucion[9]
        }


class VisualizadorSistemaCinetico:
    """Genera visualizaciones del análisis cinético"""
    
    @staticmethod
    def graficar_fuerzas_reaccion(t_vals: np.ndarray, resultados: List[Dict],
                                   nombre_archivo: str):
        """Grafica las fuerzas de reacción en función del tiempo"""
        
        # Extraer fuerzas
        r_x = [r['r_x'] for r in resultados]
        r_y = [r['r_y'] for r in resultados]
        K_x = [r['K_x'] for r in resultados]
        K_y = [r['K_y'] for r in resultados]
        M_y = [r['M_y'] for r in resultados]
        
        # Crear figura con subplots
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('Fuerzas de Reacción en el Mecanismo', fontsize=16, fontweight='bold')
        
        # Gráfica 1: r_x y r_y
        axes[0, 0].plot(t_vals, r_x, 'b-', linewidth=2, label='r_x')
        axes[0, 0].plot(t_vals, r_y, 'r-', linewidth=2, label='r_y')
        axes[0, 0].set_ylabel('Fuerza (N)', fontsize=11)
        axes[0, 0].set_title('Fuerzas en articulación manivela-palanca (punto A)', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Gráfica 2: K_x y K_y
        axes[0, 1].plot(t_vals, K_x, 'g-', linewidth=2, label='K_x')
        axes[0, 1].plot(t_vals, K_y, 'm-', linewidth=2, label='K_y')
        axes[0, 1].set_ylabel('Fuerza (N)', fontsize=11)
        axes[0, 1].set_title('Fuerzas en articulación palanca-biela (punto B)', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Gráfica 3: M_y
        axes[1, 0].plot(t_vals, M_y, 'orange', linewidth=2)
        axes[1, 0].set_ylabel('Fuerza (N)', fontsize=11)
        axes[1, 0].set_title('Fuerza vertical en conexión biela-martillo', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Gráfica 4: Magnitud M_y
        axes[1, 1].plot(t_vals, np.abs(M_y), 'purple', linewidth=2)
        axes[1, 1].set_ylabel('Fuerza (N)', fontsize=11)
        axes[1, 1].set_title('Magnitud de fuerza biela-martillo', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Gráfica 5: Magnitud de fuerzas en A
        mag_A = [np.sqrt(rx**2 + ry**2) for rx, ry in zip(r_x, r_y)]
        axes[2, 0].plot(t_vals, mag_A, 'darkblue', linewidth=2)
        axes[2, 0].set_xlabel('Tiempo (s)', fontsize=11)
        axes[2, 0].set_ylabel('Magnitud (N)', fontsize=11)
        axes[2, 0].set_title('Magnitud de fuerza en articulación manivela-palanca', fontweight='bold')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Gráfica 6: Magnitud de fuerzas en B
        mag_B = [np.sqrt(kx**2 + ky**2) for kx, ky in zip(K_x, K_y)]
        axes[2, 1].plot(t_vals, mag_B, 'darkgreen', linewidth=2)
        axes[2, 1].set_xlabel('Tiempo (s)', fontsize=11)
        axes[2, 1].set_ylabel('Magnitud (N)', fontsize=11)
        axes[2, 1].set_title('Magnitud de fuerza en articulación palanca-biela', fontweight='bold')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'imagenes/{nombre_archivo}', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Gráfica guardada: {nombre_archivo}")
    
    @staticmethod
    def graficar_reacciones_pivotes(t_vals: np.ndarray, resultados: List[Dict],
                                      nombre_archivo: str):
        """Grafica las reacciones en los pivotes"""
        
        # Extraer reacciones
        Theta_1_x = [r['Theta_1_x'] for r in resultados]
        Theta_1_y = [r['Theta_1_y'] for r in resultados]
        Theta_2_x = [r['Theta_2_x'] for r in resultados]
        Theta_2_y = [r['Theta_2_y'] for r in resultados]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Reacciones en los Pivotes', fontsize=16, fontweight='bold')
        
        # Pivote O₁ (palanca R)
        axes[0, 0].plot(t_vals, Theta_1_x, 'b-', linewidth=2)
        axes[0, 0].set_ylabel('Θ₁ₓ (N)', fontsize=11)
        axes[0, 0].set_title('Reacción en O₁ (eje x)', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        axes[0, 1].plot(t_vals, Theta_1_y, 'r-', linewidth=2)
        axes[0, 1].set_ylabel('Θ₁ᵧ (N)', fontsize=11)
        axes[0, 1].set_title('Reacción en O₁ (eje y)', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Pivote O₂ (manivela r)
        axes[1, 0].plot(t_vals, Theta_2_x, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Tiempo (s)', fontsize=11)
        axes[1, 0].set_ylabel('Θ₂ₓ (N)', fontsize=11)
        axes[1, 0].set_title('Reacción en O₂ (eje x)', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        axes[1, 1].plot(t_vals, Theta_2_y, 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Tiempo (s)', fontsize=11)
        axes[1, 1].set_ylabel('Θ₂ᵧ (N)', fontsize=11)
        axes[1, 1].set_title('Reacción en O₂ (eje y)', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'imagenes/{nombre_archivo}', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Gráfica guardada: {nombre_archivo}")


def ejemplo_analisis_sistema_completo():
    """
    Ejemplo completo de análisis del sistema de ecuaciones cinéticas
    """
    
    print("="*80)
    print("SISTEMA DE ECUACIONES CINÉTICAS - MECANISMO DE MARTILLO DE FORJA")
    print("="*80)
    print()
    
    # Parámetros geométricos del mecanismo
    geom = ParametrosGeometricos(
        R=0.15,  # 15 cm - longitud de la palanca oscilante (reducido significativamente)
        r=0.06,  # 6 cm - radio de la manivela (reducido)
        K=0.10,  # 10 cm - longitud de la biela
        d=0.09,  # 9 cm - distancia entre pivotes O₁ y O₂ (reducido)
        D=0.12  # 12 cm - distancia vertical hasta línea del martillo (ajustado)
    )
    
    # Propiedades físicas (masas realistas para prototipo impreso en 3D - PLA)
    props = PropiedadesFisicas(
        m_R=0.2,   # 0.25 kg - palanca de PLA impresa en 3D
        m_r=0.05,   # 0.05 kg - manivela de PLA
        m_K=0.15,   # 0.15 kg - biela de PLA
        m_M=0.50,   # 0.50 kg - martillo de PLA (para golpear plastilina)
        g=9.81
    )
    
    print("PARÁMETROS DEL SISTEMA:")
    print("-" * 80)
    print(f"Geometría:")
    print(f"  R = {geom.R*100:.1f} cm (palanca)")
    print(f"  r = {geom.r*100:.1f} cm (manivela)")
    print(f"  K = {geom.K*100:.1f} cm (biela)")
    print(f"  d = {geom.d*100:.1f} cm (distancia entre pivotes)")
    print()
    print(f"Masas:")
    print(f"  m_R = {props.m_R:.2f} kg")
    print(f"  m_r = {props.m_r:.2f} kg")
    print(f"  m_K = {props.m_K:.2f} kg")
    print(f"  m_M = {props.m_M:.2f} kg")
    print(f"  Masa total = {props.m_R + props.m_r + props.m_K + props.m_M:.2f} kg")
    print()
    
    # Parámetros de simulación
    omega_2 = 2.0  # rad/s - velocidad angular del motor reducida (~19 RPM) para menores fuerzas
    
    # Crear sistema cinético (ahora requiere omega_2)
    sistema = SistemaEcuacionesCinetico(geom, props, omega_2)
    
    # Pesos e inercias
    W_R, W_r, W_K, W_M = sistema.calcular_pesos()
    I_R, I_r, I_K = sistema.calcular_inercias()
    
    print(f"Pesos:")
    print(f"  W_R = {W_R:.2f} N")
    print(f"  W_r = {W_r:.2f} N")
    print(f"  W_K = {W_K:.2f} N")
    print(f"  W_M = {W_M:.2f} N")
    print()
    print(f"Momentos de inercia (respecto al centro de masa):")
    print(f"  I_R = {I_R:.6f} kg·m²")
    print(f"  I_r = {I_r:.6f} kg·m²")
    print(f"  I_K = {I_K:.6f} kg·m²")
    print()
    T = 2 * np.pi / omega_2  # Período de una revolución
    n_ciclos = 2
    t_vals = np.linspace(0, n_ciclos * T, 200)
    
    print(f"Simulación:")
    print(f"  ω₂ = {omega_2:.2f} rad/s")
    print(f"  Período = {T:.3f} s")
    print(f"  Duración = {n_ciclos * T:.3f} s ({n_ciclos} ciclos)")
    print()
    
    # Resolver sistema para cada instante de tiempo
    print("Resolviendo sistema de ecuaciones...")
    print("-" * 80)
    
    resultados = []
    
    for i, t in enumerate(t_vals):
        # Calcular estado cinemático usando MecanismoCinematico de analisis_dependencias
        estado_mec = sistema.mecanismo.estado_completo(t)
        
        # Extraer valores del estado calculado
        delta = estado_mec['delta']
        alpha = estado_mec['alpha']
        L_t = estado_mec['L']
        omega_1 = estado_mec['omega_1']
        alpha_1 = estado_mec['alpha_ddot']  # Aceleración angular de R
        
        # Ángulo beta de la biela K - calculado del lazo vectorial
        # Del lazo: R·sin(α) - K·sin(β) - D = 0
        # Despejando: sin(β) = (R·sin(α) - D) / K
        sin_beta = (geom.R * np.sin(alpha) - geom.D) / geom.K
        # Limitar sin_beta al rango válido [-1, 1]
        sin_beta = np.clip(sin_beta, -1.0, 1.0)
        beta = np.arcsin(sin_beta)
        
        # Calcular omega_K (velocidad angular de la biela K)
        # Derivando el lazo vectorial: R·sin(α) - K·sin(β) - D = 0
        # → R·cos(α)·α̇ - K·cos(β)·β̇ = 0
        # → β̇ = (R·ω₁·cos(α)) / (K·cos(β))
        cos_beta = np.cos(beta)
        sin_beta = np.sin(beta)
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        
        if abs(cos_beta) > 1e-3:  # Evitar singularidad cuando β ≈ ±90°
            omega_K = (geom.R * omega_1 * cos_alpha) / (geom.K * cos_beta)
            
            # Calcular alpha_K (aceleración angular de la biela K)
            # α_K = β̈ = [R·α̈·cos(α) - R·ω₁²·sin(α) + R·ω₁·cos(α)·sin(β)·β̇] / (K·cos²(β))
            numerador_alpha_K = (geom.R * alpha_1 * cos_alpha 
                                - geom.R * omega_1**2 * sin_alpha 
                                + geom.R * omega_1 * cos_alpha * sin_beta * omega_K)
            alpha_K = numerador_alpha_K / (geom.K * cos_beta**2)
        else:
            omega_K = 0.0  # Singularidad: la biela está vertical
            alpha_K = 0.0
        
        # Calcular posición del martillo del lazo vectorial
        # R·cos(α) - X_E - K·cos(β) = 0
        X_E = geom.R * np.cos(alpha) - geom.K * np.cos(beta)
        
        # Calcular velocidad del martillo (del lazo vectorial)
        # Ẋ_E = [R·ω₁·(D·cos(α) - X_E·sin(α))] / [X_E - R·cos(α)]
        numerador = geom.R * omega_1 * (geom.D * np.cos(alpha) - X_E * np.sin(alpha))
        denominador = X_E - geom.R * np.cos(alpha)
        if abs(denominador) > 1e-6:
            X_E_dot = numerador / denominador
        else:
            X_E_dot = 0.0
        
        # Calcular aceleración del martillo (del lazo vectorial)
        # Ẍ_E = R · {[α̈·A - α̇²·B - α̇·Ẋ_E·sin(α)]·D_en - α̇·A·(Ẋ_E + R·α̇·sin(α))} / D_en²
        # donde: A = D·cos(α) - X_E·sin(α)
        #        B = D·sin(α) + X_E·cos(α)
        #        D_en = X_E - R·cos(α)
        A = geom.D * np.cos(alpha) - X_E * np.sin(alpha)
        B = geom.D * np.sin(alpha) + X_E * np.cos(alpha)
        D_en = denominador  # Ya calculado arriba
        
        if abs(D_en) > 1e-6:
            numerador_acel = (alpha_1 * A - omega_1**2 * B - omega_1 * X_E_dot * np.sin(alpha)) * D_en \
                           - omega_1 * A * (X_E_dot + geom.R * omega_1 * np.sin(alpha))
            X_E_ddot = geom.R * numerador_acel / (D_en**2)
        else:
            X_E_ddot = 0.0
        
        estado = EstadoCinematico(
            delta=delta,
            alpha=alpha,
            beta=beta,
            L=L_t,  # Longitud variable del eslabón virtual (calculada)
            omega_2=omega_2,
            omega_1=omega_1,
            omega_K=omega_K,  # Calculado correctamente
            alpha_2=0.0,  # Motor a velocidad constante
            alpha_1=alpha_1,
            alpha_K=alpha_K,  # Calculado como derivada de omega_K
            X_E=X_E,  # Calculado del lazo vectorial
            X_E_dot=X_E_dot,  # Calculado del lazo vectorial
            X_E_ddot=X_E_ddot  # Calculado del lazo vectorial
        )
        
        # Resolver sistema
        solucion = sistema.resolver_sistema(estado)
        solucion['omega_1'] = omega_1  # Guardar para siguiente iteración
        resultados.append(solucion)
        
        # Mostrar progreso
        if (i + 1) % 50 == 0 or i == len(t_vals) - 1:
            print(f"  Progreso: {i+1}/{len(t_vals)} ({100*(i+1)/len(t_vals):.1f}%)")
    
    print()
    print("✓ Sistema resuelto exitosamente")
    print()
    
    # Generar gráficas
    print("Generando visualizaciones...")
    print("-" * 80)
    
    visualizador = VisualizadorSistemaCinetico()
    
    visualizador.graficar_fuerzas_reaccion(
        t_vals, resultados,
        'fuerzas_reaccion_sistema.png'
    )
    
    visualizador.graficar_reacciones_pivotes(
        t_vals, resultados,
        'reacciones_pivotes_sistema.png'
    )
    
    print()
    print("="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)
    print()
    print("Archivos generados en la carpeta 'imagenes/':")
    print("  - fuerzas_reaccion_sistema.png")
    print("  - reacciones_pivotes_sistema.png")
    print()


if __name__ == "__main__":
    ejemplo_analisis_sistema_completo()