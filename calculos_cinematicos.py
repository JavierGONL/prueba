"""
Cálculos Cinemáticos - Análisis de Mecanismos
Implementación de las ecuaciones del informe de dinámica
"""

import numpy as np
from typing import Tuple


class CalculosCinematicos:
    """Clase para realizar cálculos cinemáticos de mecanismos"""
    
    def __init__(self, r: float, d: float, R: float, K: float, D: float, omega_2: float = 0):
        """
        Inicializa los parámetros del mecanismo
        
        Args:
            r: Radio de la barra O2A (r minúscula)
            d: Distancia fija del mecanismo
            R: Longitud de la barra R (R mayúscula)
            K: Longitud de la barra rígida K (constante del mecanismo)
            D: Distancia vertical D (constante del mecanismo)
            omega_2: Velocidad angular del motor (constante) en rad/s
        """
        self.r = r
        self.d = d
        self.R = R  # Longitud de la barra R
        self.K = K  # Longitud de la barra rígida K (constante)
        self.D = D  # Distancia vertical D (constante)
        self.omega_2 = omega_2  # Velocidad angular del motor (constante)
    
    # ===== RELACIÓN ENTRE α Y δ =====
    
    def calcular_delta(self, t: float) -> float:
        """
        Calcula el ángulo δ en función del tiempo
        δ(t) = ω₂ · t (asumiendo δ₀ = 0)
        
        Args:
            t: Tiempo en segundos
            
        Returns:
            Ángulo delta en radianes
        """
        return self.omega_2 * t
    
    def calcular_L(self, delta: float) -> float:
        """
        Calcula la longitud L del mecanismo
        Ecuación (2): L = sqrt(r² + d² + 2rd*sin(δ))
        
        Args:
            delta: Ángulo delta en radianes
            
        Returns:
            Longitud L
        """
        return np.sqrt(self.r**2 + self.d**2 + 2*self.r*self.d*np.sin(delta))
    
    def calcular_alpha(self, delta: float) -> float:
        """
        Calcula el ángulo α a partir de δ
        Ecuación (6): cos(α) = r*cos(δ) / sqrt(r² + d² + 2rd*sin(δ))
        
        Args:
            delta: Ángulo delta en radianes
            
        Returns:
            Ángulo alpha en radianes
        """
        L = self.calcular_L(delta)
        cos_alpha = (self.r * np.cos(delta)) / L
        return np.arccos(cos_alpha)
    
    # ===== ANÁLISIS DE VELOCIDADES =====
    
    def calcular_V_af(self, alpha: float, delta: float) -> float:
        """
        Calcula la velocidad relativa V_a/f
        Ecuación (7): V_a/f = (sin(α)*cos(δ) + sin(δ)*cos(α)) * ω₂ * r
        
        Args:
            alpha: Ángulo alpha en radianes
            delta: Ángulo delta en radianes
            
        Returns:
            Velocidad relativa V_a/f
        """
        return (np.sin(alpha + delta)) * self.omega_2 * self.r
    
    def calcular_alpha_dot(self, alpha: float, delta: float) -> float:
        """
        Calcula α̇ (velocidad angular del eslabón)
        Ecuación (9): α̇ = ω₁ = (ω₂*r/L) * [cos(α-δ)]
        
        Nota: α̇ y ω₁ son equivalentes (velocidad angular del eslabón)
        
        Args:
            alpha: Ángulo alpha en radianes
            delta: Ángulo delta en radianes
            
        Returns:
            Velocidad angular α̇ = ω₁
        """
        L = self.calcular_L(delta)
        return (self.omega_2 * self.r / L) * (np.cos(alpha-delta))
    
    def calcular_alpha_ddot(self, alpha: float, delta: float) -> float:
        """
        Calcula α̈ (aceleración angular del eslabón) = α₁''
        Resuelve el sistema de ecuaciones de aceleración para obtener α̈
        
        Args:
            alpha: Ángulo alpha en radianes
            delta: Ángulo delta en radianes
            
        Returns:
            Aceleración angular α̈ = α₁''
        """
        L = self.calcular_L(delta)
        omega_1 = self.calcular_alpha_dot(alpha, delta)
        V_af = self.calcular_V_af(alpha, delta)
        alpha_2 = 0  # Motor a velocidad constante
        
        alpha_1, _ = self.resolver_aceleraciones(alpha, delta, omega_1, self.omega_2,
                                                 alpha_2, V_af, L, self.R)
        return alpha_1
    
    # ===== LAZO VECTORIAL Y VELOCIDAD DEL MARTILLO =====
    
    def calcular_beta(self, alpha: float) -> float:
        """
        Calcula el ángulo β (beta) del mecanismo
        Ecuación del lazo vectorial: R*sen(α) - D = K*sen(β)
        Despejando: β = arcsen((R*sen(α) - D) / K)
        
        Args:
            alpha: Ángulo alpha en radianes
            
        Returns:
            Ángulo beta en radianes
        """
        sin_beta = (self.R * np.sin(alpha) - self.D) / self.K
        return np.arcsin(sin_beta)
    
    def calcular_X_E(self, alpha: float, beta: float) -> float:
        """
        Calcula la posición X del punto E (martillo)
        Ecuación: X_E = R*cos(α) - K*cos(β)
        
        Args:
            alpha: Ángulo alpha en radianes
            beta: Ángulo beta en radianes
            
        Returns:
            Posición X del punto E
        """
        return self.R * np.cos(alpha) - self.K * np.cos(beta)
    
    def calcular_velocidad_martillo(self, R: float, X_E: float, D: float, 
                                   alpha: float, alpha_dot: float) -> float:
        """
        Calcula la velocidad del punto E (martillo)
        Ecuación (17): Ẋ_E = R*α̇[D*cos(α) - X_E*sin(α)] / (X_E - R*cos(α))
        
        Args:
            R: Longitud de la barra R
            X_E: Posición X del punto E
            D: Distancia vertical D
            alpha: Ángulo alpha en radianes
            alpha_dot: Velocidad angular α̇ (dα/dt)
            
        Returns:
            Velocidad del punto E (Ẋ_E)
        """
        numerador = R * alpha_dot * (D*np.cos(alpha) - X_E*np.sin(alpha))
        denominador = X_E - R*np.cos(alpha) 
        return numerador / denominador
    
    # ===== ANÁLISIS DE ACELERACIONES =====
    
    def calcular_aceleracion_martillo(self, R: float, X_E: float, D: float,
                                     alpha: float, alpha_dot: float, 
                                     alpha_ddot: float, X_E_dot: float) -> float:
        """
        Calcula la aceleración del punto E (martillo)
        Ecuación (20): Ẍ_E = R * [α̈*A - α̇²*B - α̇*Ẋ_E*sin(α)] * D_en - α̇*A*(Ẋ_E + R*α̇*sin(α)) / D_en²
        
        donde:
            A = D*cos(α) - X_E*sin(α)
            B = D*sin(α) + X_E*cos(α)
            D_en = X_E - R*cos(α)
        
        Args:
            R: Longitud de la barra R
            X_E: Posición X del punto E
            D: Distancia vertical D
            alpha: Ángulo alpha en radianes
            alpha_dot: Velocidad angular α̇
            alpha_ddot: Aceleración angular α̈
            X_E_dot: Velocidad del punto E (Ẋ_E)
            
        Returns:
            Aceleración del punto E (Ẍ_E)
        """
        # Variables auxiliares
        A = D*np.cos(alpha) - X_E*np.sin(alpha)
        B = D*np.sin(alpha) + X_E*np.cos(alpha)
        D_en = X_E - R*np.cos(alpha)
        
        # Numerador
        termino1 = alpha_ddot * A - alpha_dot**2 * B - alpha_dot * X_E_dot * np.sin(alpha)
        termino2 = alpha_dot * A * (X_E_dot + R * alpha_dot * np.sin(alpha))
        numerador = termino1 * D_en - termino2
        
        # Denominador
        denominador = D_en**2
        
        return R * numerador / denominador
    
    def resolver_aceleraciones(self, alpha: float, delta: float, 
                              omega_1: float, omega_2: float, 
                              alpha_2: float, V_af: float, L: float, R: float) -> Tuple[float, float]:
        """
        Resuelve el sistema de ecuaciones para encontrar α₁ y a_a/f
        Ecuaciones (19a) y (19b)
        
        Args:
            alpha: Ángulo alpha en radianes
            delta: Ángulo delta en radianes
            omega_1: Velocidad angular ω₁
            omega_2: Velocidad angular ω₂
            alpha_2: Aceleración angular α₂
            V_af: Velocidad relativa V_a/f
            L: Longitud L
            R: Radio R
            
        Returns:
            Tupla (alpha_1, a_af) con las aceleraciones
        """
        # Sistema de ecuaciones lineales: Ax = b
        # (a'_a)_t * sin(α) + a_a/f * cos(α) = RHS1
        # (a'_a)_t * cos(α) - a_a/f * sin(α) = RHS2
        
        # Lado derecho de las ecuaciones
        RHS1 = (omega_2**2 * self.r * np.cos(delta) + 
                alpha_2 * R * np.sin(delta) - 
                2*omega_1 * V_af * np.sin(alpha) - 
                omega_1**2 * L * np.cos(alpha))
        
        RHS2 = (alpha_2 * R * np.cos(delta) - 
                omega_2**2 * self.r * np.sin(delta) - 
                2*omega_1 * V_af * np.cos(alpha) + 
                omega_1**2 * L * np.sin(alpha))
        
        # Matriz de coeficientes
        A_matrix = np.array([
            [np.sin(alpha), np.cos(alpha)],
            [np.cos(alpha), -np.sin(alpha)]
        ])
        
        b_vector = np.array([RHS1, RHS2])
        
        # Resolver el sistema
        solucion = np.linalg.solve(A_matrix, b_vector)
        
        a_prime_a_t = solucion[0]  # (a'_a)_t - aceleración tangencial
        a_af = solucion[1]          # a_a/f - aceleración relativa
        
        # Calcular α₁ a partir de (a'_a)_t
        alpha_1 = a_prime_a_t / L
        
        return alpha_1, a_af


# ===== FUNCIONES DE UTILIDAD =====

def grados_a_radianes(grados: float) -> float:
    """Convierte grados a radianes"""
    return np.deg2rad(grados)

def radianes_a_grados(radianes: float) -> float:
    """Convierte radianes a grados"""
    return np.rad2deg(radianes)


# ===== EJEMPLO DE USO =====

if __name__ == "__main__":
    # Parámetros del mecanismo
    r = 0.075  # metros (9 cm) - r minúscula
    d = 0.1  # metros
    R = 0.2  # metros (10 cm) - R mayúscula
    K = 0.1  # metros - Longitud de la barra rígida K (constante)
    D = 0.17  # metros - Distancia vertical D (constante)
    omega_2 = 4  # rad/s - Velocidad angular del motor (constante)
    
    # Crear instancia de la clase
    mecanismo = CalculosCinematicos(r, d, R, K, D, omega_2)
    
    # Simular en diferentes tiempos
    tiempos = np.arange(0, 10, 0.2)  # segundos (de 0 a 10s con intervalos de 0.1s)
    
    print("=== CÁLCULOS CINEMÁTICOS ===\n")
    print(f"Parámetros del sistema:")
    print(f"  r = {r} m")
    print(f"  d = {d} m")
    print(f"  R = {R} m")
    print(f"  K = {K} m")
    print(f"  D = {D} m")
    print(f"  ω₂ (motor) = {omega_2} rad/s (constante)\n")
    print("="*80)
    
    for t in tiempos:
        # Calcular δ en función del tiempo
        delta = mecanismo.calcular_delta(t)
        delta_grados = radianes_a_grados(delta)
        
        # Calcular L y α
        L = mecanismo.calcular_L(delta)
        alpha = mecanismo.calcular_alpha(delta)
        
        print(f"\n>>> Tiempo t = {t} s")
        print(f"-"*80)
        print(f"  δ(t) = ω₂·t = {delta:.4f} rad = {delta_grados:.2f}°")
        print(f"  L = {L:.4f} m")
        print(f"  α = {radianes_a_grados(alpha):.2f}°")
        
        # Análisis de velocidades
        V_af = mecanismo.calcular_V_af(alpha, delta)
        alpha_dot = mecanismo.calcular_alpha_dot(alpha, delta)  # α̇ = ω₁
        
        # Análisis de aceleraciones
        alpha_ddot = mecanismo.calcular_alpha_ddot(alpha, delta)  # α̈ = α₁''
        
        print(f"\n  Velocidades:")
        print(f"    δ̇ = ω₂ = {omega_2} rad/s")
        print(f"    α̇ = ω₁ = {alpha_dot:.4f} rad/s")
        print(f"    V_a/f = {V_af:.4f} m/s")
        
        print(f"\n  Aceleraciones:")
        print(f"    α̈ = α₁'' = {alpha_ddot:.4f} rad/s²")
    
    print("\n" + "="*80)
    print("\n>>> ANÁLISIS DEL MARTILLO (ejemplo en t = 0.1s)")
    print("-"*80)
    
    # Usar un tiempo específico
    t = 0.1
    delta = mecanismo.calcular_delta(t)
    alpha = mecanismo.calcular_alpha(delta)
    
    # Velocidad del martillo
    beta = mecanismo.calcular_beta(alpha)
    X_E = mecanismo.calcular_X_E(alpha, beta)
    alpha_dot = mecanismo.calcular_alpha_dot(alpha, delta)  # α̇ = ω₁
    
    X_E_dot = mecanismo.calcular_velocidad_martillo(mecanismo.R, X_E, mecanismo.D, alpha, alpha_dot)
    
    print(f"  R = {mecanismo.R} m (constante del mecanismo)")
    print(f"  K = {mecanismo.K} m (constante del mecanismo)")
    print(f"  D = {mecanismo.D} m (constante del mecanismo)")
    print(f"  β = {radianes_a_grados(beta):.2f}°")
    print(f"  X_E = {X_E:.4f} m")
    print(f"  Ẋ_E = {X_E_dot:.4f} m/s")
    
    # Aceleración del martillo
    alpha_ddot = 2  # rad/s²
    X_E_ddot = mecanismo.calcular_aceleracion_martillo(mecanismo.R, X_E, mecanismo.D, alpha, 
                                                        alpha_dot, alpha_ddot, X_E_dot)
    
    print(f"\n  Aceleración:")
    print(f"    α̈ = {alpha_ddot} rad/s²")
    print(f"    Ẍ_E = {X_E_ddot:.4f} m/s²")
    
    # Resolver sistema de aceleraciones
    L = mecanismo.calcular_L(delta)
    omega_1 = mecanismo.calcular_alpha_dot(alpha, delta)  # ω₁ = α̇
    V_af = mecanismo.calcular_V_af(alpha, delta)
    alpha_2 = 0  # rad/s² - El motor tiene velocidad constante, por lo tanto α₂ = 0
    
    alpha_1, a_af = mecanismo.resolver_aceleraciones(alpha, delta, omega_1, omega_2, 
                                                     alpha_2, V_af, L, mecanismo.R)
    
    print(f"\n  Sistema de aceleraciones:")
    print(f"    α₂ (motor) = {alpha_2} rad/s² (motor a velocidad constante)")
    print(f"    α₁ = {alpha_1:.4f} rad/s²")
    print(f"    a_a/f = {a_af:.4f} m/s²")
    print("\n" + "="*80)
