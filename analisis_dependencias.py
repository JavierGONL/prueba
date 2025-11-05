"""
Análisis de Dependencias - Todas las variables en función de δ(t)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from calculos_cinematicos import CalculosCinematicos

# Crear carpeta de imágenes si no existe
os.makedirs('imagenes', exist_ok=True)

class MecanismoCinematico:
    """
    Análisis cinemático completo del mecanismo
    Todas las variables se calculan en función de δ(t) = ω₂·t
    """
    
    def __init__(self, r: float, d: float, R: float, K: float, D: float, omega_2: float):
        """
        Args:
            r: Radio de la barra O2A
            d: Distancia fija
            R: Longitud de la barra R
            K: Longitud de la barra rígida K
            D: Distancia vertical D
            omega_2: Velocidad angular del motor (constante)
        """
        self.r = r
        self.d = d
        self.R = R
        self.K = K
        self.D = D
        self.omega_2 = omega_2
    
    # NIVEL 1: δ es la variable independiente (entrada del motor)
    def delta(self, t: float) -> float:
        """δ(t) = ω₂·t"""
        return self.omega_2 * t
    
    def delta_dot(self) -> float:
        """δ̇ = ω₂ (constante)"""
        return self.omega_2
    
    def delta_ddot(self) -> float:
        """δ̈ = 0 (motor a velocidad constante)"""
        return 0.0
    
    # NIVEL 2: Variables que dependen directamente de δ
    def L(self, delta: float) -> float:
        """L(δ) = √(r² + d² + 2rd·sin(δ))"""
        return np.sqrt(self.r**2 + self.d**2 + 2*self.r*self.d*np.sin(delta))
    
    def alpha(self, delta: float) -> float:
        """α(δ) = arccos(r·cos(δ) / L(δ))"""
        L_val = self.L(delta)
        return np.arccos((self.r * np.cos(delta)) / L_val)
    
    # NIVEL 3: Derivadas de L y α respecto a δ
    def dL_ddelta(self, delta: float) -> float:
        """dL/dδ = r·d·cos(δ) / L(δ)"""
        L_val = self.L(delta)
        return (self.r * self.d * np.cos(delta)) / L_val
    
    def dalpha_ddelta(self, delta: float) -> float:
        """dα/dδ - Derivada de α respecto a δ"""
        L_val = self.L(delta)
        dL = self.dL_ddelta(delta)
        
        numerador = -self.r * np.sin(delta) * L_val - self.r * np.cos(delta) * dL
        denominador = L_val**2 * np.sqrt(1 - (self.r * np.cos(delta) / L_val)**2)
        
        return numerador / denominador
    
    # NIVEL 4: Velocidades (usando regla de la cadena)
    def alpha_dot(self, delta: float) -> float:
        """α̇ = ω₁ (velocidad angular del eslabón)"""
        return self.omega_1(delta)
    
    def V_af(self, delta: float) -> float:
        """V_a/f(δ) = sin(α(δ) + δ)·ω₂·r"""
        alpha_val = self.alpha(delta)
        return np.sin(alpha_val + delta) * self.omega_2 * self.r
    
    def beta(self, delta: float) -> float:
        """β(δ) - Ángulo beta calculado a partir de alpha"""
        alpha_val = self.alpha(delta)
        sin_beta = (self.R * np.sin(alpha_val) - self.D) / self.K
        return np.arcsin(sin_beta)
    
    def X_E(self, delta: float) -> float:
        """X_E(δ) = R·cos(α) - K·cos(β)"""
        alpha_val = self.alpha(delta)
        beta_val = self.beta(delta)
        return self.R * np.cos(alpha_val) - self.K * np.cos(beta_val)
    
    def omega_1(self, delta: float) -> float:
        """ω₁(δ) = (ω₂·r / L(δ))·cos(α(δ) - δ)"""
        L_val = self.L(delta)
        alpha_val = self.alpha(delta)
        return (self.omega_2 * self.r / L_val) * np.cos(alpha_val - delta)
    
    # NIVEL 5: Aceleraciones
    def alpha_ddot(self, delta: float) -> float:
        """α̈ = dω₁/dt (aceleración angular del eslabón)"""
        # Aproximación numérica de la derivada de ω₁
        h = 1e-6
        t = delta / self.omega_2  # Aproximación del tiempo
        omega1_plus = self.omega_1(delta + h)
        omega1_minus = self.omega_1(delta - h)
        domega1_ddelta = (omega1_plus - omega1_minus) / (2 * h)
        
        # α̈ = dω₁/dt = (dω₁/dδ)·(dδ/dt) = (dω₁/dδ)·ω₂
        return domega1_ddelta * self.omega_2
    
    # RESUMEN: Todo en función de t
    def estado_completo(self, t: float) -> dict:
        """
        Calcula todas las variables cinemáticas en función del tiempo
        
        Returns:
            Diccionario con todas las variables
        """
        # Nivel 1: Entrada
        delta_val = self.delta(t)
        delta_dot_val = self.delta_dot()
        delta_ddot_val = self.delta_ddot()
        
        # Nivel 2: Geometría
        L_val = self.L(delta_val)
        alpha_val = self.alpha(delta_val)
        beta_val = self.beta(delta_val)
        X_E_val = self.X_E(delta_val)
        
        # Nivel 3: Velocidades
        alpha_dot_val = self.alpha_dot(delta_val)
        V_af_val = self.V_af(delta_val)
        omega_1_val = self.omega_1(delta_val)
        
        # Nivel 4: Aceleraciones
        alpha_ddot_val = self.alpha_ddot(delta_val)
        
        return {
            't': t,
            'delta': delta_val,
            'delta_dot': delta_dot_val,
            'delta_ddot': delta_ddot_val,
            'L': L_val,
            'alpha': alpha_val,
            'beta': beta_val,
            'X_E': X_E_val,
            'alpha_dot': alpha_dot_val,
            'alpha_ddot': alpha_ddot_val,
            'V_af': V_af_val,
            'omega_1': omega_1_val
        }


def imprimir_cadena_dependencias():
    """Imprime el árbol de dependencias"""
    print("="*80)
    print("CADENA DE DEPENDENCIAS - TODO EN FUNCIÓN DE δ(t)")
    print("="*80)
    print()
    print("NIVEL 0: Parámetros constantes")
    print("  └─ r, d, R, K, D, ω₂")
    print()
    print("NIVEL 1: Variable independiente (entrada del motor)")
    print("  └─ δ(t) = ω₂·t")
    print("  └─ δ̇ = ω₂")
    print("  └─ δ̈ = 0")
    print()
    print("NIVEL 2: Geometría (dependen directamente de δ)")
    print("  ├─ L(δ) = √(r² + d² + 2rd·sin(δ))")
    print("  ├─ α(δ) = arccos(r·cos(δ) / L(δ))")
    print("  ├─ β(α) = arcsin((R·sin(α) - D) / K)")
    print("  └─ X_E(α,β) = R·cos(α) - K·cos(β)")
    print()
    print("NIVEL 3: Derivadas respecto a δ")
    print("  ├─ dL/dδ")
    print("  └─ dα/dδ")
    print()
    print("NIVEL 4: Velocidades (regla de la cadena)")
    print("  ├─ α̇ = ω₁ (velocidad angular del eslabón)")
    print("  ├─ V_a/f(δ) = sin(α(δ) + δ)·ω₂·r")
    print("  └─ ω₁(δ) = (ω₂·r / L(δ))·cos(α(δ) - δ)")
    print()
    print("NIVEL 5: Aceleraciones (segunda derivada)")
    print("  └─ α̈(δ) = dω₁/dt")
    print()
    print("="*80)
    print()
    print("✓ TODAS LAS VARIABLES ESTÁN EN FUNCIÓN DE δ(t) = ω₂·t")
    print("✓ CONOCIENDO t, ω₂, r, d, R, K, D → SE CALCULA TODO EL SISTEMA")
    print()
    print("="*80)


if __name__ == "__main__":
    # Parámetros del sistema (geometría modificada para reducir fuerzas)
    r = 0.06      # metros (6 cm) - reducido
    d = 0.09      # metros (9 cm) - reducido
    R = 0.15      # metros (15 cm) - reducido significativamente
    K = 0.1       # metros (10 cm)
    D = 0.12      # metros (12 cm) - ajustado
    omega_2 = 2   # rad/s - reducido para menores fuerzas
    
    # Crear mecanismo
    mecanismo = MecanismoCinematico(r, d, R, K, D, omega_2)
    
    # Mostrar cadena de dependencias
    imprimir_cadena_dependencias()
    
    # Calcular estados en diferentes tiempos
    print("\nESTADOS DEL SISTEMA EN FUNCIÓN DEL TIEMPO")
    print("="*80)
    
    tiempos = [0, 0.1, 0.2, 0.5, 1.0]
    
    for t in tiempos:
        estado = mecanismo.estado_completo(t)
        
        print(f"\nt = {t:.2f} s")
        print("-"*80)
        print(f"  Entrada (motor):")
        print(f"    δ = {estado['delta']:.4f} rad = {np.rad2deg(estado['delta']):.2f}°")
        print(f"    δ̇ = {estado['delta_dot']:.4f} rad/s")
        print(f"    δ̈ = {estado['delta_ddot']:.4f} rad/s²")
        print(f"  Geometría:")
        print(f"    L = {estado['L']:.4f} m")
        print(f"    α = {estado['alpha']:.4f} rad = {np.rad2deg(estado['alpha']):.2f}°")
        print(f"    β = {estado['beta']:.4f} rad = {np.rad2deg(estado['beta']):.2f}°")
        print(f"    X_E = {estado['X_E']:.4f} m")
        print(f"  Velocidades:")
        print(f"    α̇ = {estado['alpha_dot']:.4f} rad/s")
        print(f"    V_a/f = {estado['V_af']:.4f} m/s")
        print(f"    ω₁ = {estado['omega_1']:.4f} rad/s")
        print(f"  Aceleraciones:")
        print(f"    α̈ = {estado['alpha_ddot']:.4f} rad/s²")
    
    print("\n" + "="*80)
    
    # Generar gráficas
    print("\nGenerando gráficas de las variables en función del tiempo...")
    
    # Rango de tiempo
    t_vals = np.linspace(0, 2, 200)
    
    # Calcular todas las variables del mecanismo
    delta_vals = [mecanismo.delta(t) for t in t_vals]
    alpha_vals = [mecanismo.alpha(mecanismo.delta(t)) for t in t_vals]
    beta_vals = [mecanismo.beta(mecanismo.delta(t)) for t in t_vals]
    X_E_vals = [mecanismo.X_E(mecanismo.delta(t)) for t in t_vals]
    L_vals = [mecanismo.L(mecanismo.delta(t)) for t in t_vals]
    omega_1_vals = [mecanismo.omega_1(mecanismo.delta(t)) for t in t_vals]
    alpha_dot_vals = [mecanismo.alpha_dot(mecanismo.delta(t)) for t in t_vals]
    V_af_vals = [mecanismo.V_af(mecanismo.delta(t)) for t in t_vals]
    alpha_ddot_vals = [mecanismo.alpha_ddot(mecanismo.delta(t)) for t in t_vals]
    delta_dot_vals = [mecanismo.delta_dot() for _ in t_vals]
    delta_ddot_vals = [mecanismo.delta_ddot() for _ in t_vals]
    
    # Calcular variables del martillo (usando CalculosCinematicos)
    calc = CalculosCinematicos(r, d, R, K, D, omega_2)
    
    # Velocidad y aceleración del martillo
    X_E_dot_vals = []
    X_E_ddot_vals = []
    
    for t in t_vals:
        delta = calc.calcular_delta(t)
        alpha = calc.calcular_alpha(delta)
        beta = calc.calcular_beta(alpha)
        X_E = calc.calcular_X_E(alpha, beta)
        alpha_dot = calc.calcular_alpha_dot(alpha, delta)
        
        # Velocidad del martillo
        X_E_dot = calc.calcular_velocidad_martillo(R, X_E, D, alpha, alpha_dot)
        X_E_dot_vals.append(X_E_dot)
        
        # Aceleración del martillo
        alpha_ddot = mecanismo.alpha_ddot(delta)
        X_E_ddot = calc.calcular_aceleracion_martillo(R, X_E, D, alpha, alpha_dot, alpha_ddot, X_E_dot)
        X_E_ddot_vals.append(X_E_ddot)
    
    # ========== GRÁFICAS INDIVIDUALES DE POSICIONES ==========
    
    # Gráfica 1: δ(t)
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, np.rad2deg(delta_vals), 'b-', linewidth=2.5)
    plt.xlabel('Tiempo (s)', fontsize=13)
    plt.ylabel('δ (°)', fontsize=13)
    plt.title('Ángulo de Entrada δ(t) = ω₂·t', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('imagenes/grafica_delta.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfica guardada: grafica_delta.png")
    
    # Gráfica 2: α(t)
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, np.rad2deg(alpha_vals), 'r-', linewidth=2.5)
    plt.xlabel('Tiempo (s)', fontsize=13)
    plt.ylabel('α (°)', fontsize=13)
    plt.title('Ángulo α(δ(t))', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('imagenes/grafica_alpha.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfica guardada: grafica_alpha.png")
    
    # Gráfica 3: L(t)
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, L_vals, 'g-', linewidth=2.5)
    plt.xlabel('Tiempo (s)', fontsize=13)
    plt.ylabel('L (m)', fontsize=13)
    plt.title('Longitud L(δ(t))', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('imagenes/grafica_L.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfica guardada: grafica_L.png")
    
    # Gráfica 4: α vs δ (Relación paramétrica)
    plt.figure(figsize=(10, 6))
    plt.plot(np.rad2deg(delta_vals), np.rad2deg(alpha_vals), 'purple', linewidth=2.5)
    plt.xlabel('δ (°)', fontsize=13)
    plt.ylabel('α (°)', fontsize=13)
    plt.title('Relación α(δ)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('imagenes/grafica_alpha_vs_delta.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfica guardada: grafica_alpha_vs_delta.png")
    
    # Gráfica 5: β(t)
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, np.rad2deg(beta_vals), 'orange', linewidth=2.5)
    plt.xlabel('Tiempo (s)', fontsize=13)
    plt.ylabel('β (°)', fontsize=13)
    plt.title('Ángulo β(t)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('imagenes/grafica_beta.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfica guardada: grafica_beta.png")
    
    # Gráfica 6: X_E(t) - Posición del martillo
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, X_E_vals, 'brown', linewidth=2.5)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Tiempo (s)', fontsize=13)
    plt.ylabel('X_E (m)', fontsize=13)
    plt.title('Posición del Martillo X_E(t)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('imagenes/grafica_X_E.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfica guardada: grafica_X_E.png")
    
    # ========== GRÁFICAS INDIVIDUALES DE VELOCIDADES ==========
    
    # Gráfica 7: δ̇(t) - Velocidad angular del motor
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, delta_dot_vals, 'b-', linewidth=2.5)
    plt.xlabel('Tiempo (s)', fontsize=13)
    plt.ylabel('δ̇ (rad/s)', fontsize=13)
    plt.title('Velocidad Angular del Motor δ̇ = ω₂ (constante)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([omega_2 - 1, omega_2 + 1])
    plt.tight_layout()
    plt.savefig('imagenes/grafica_delta_dot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfica guardada: grafica_delta_dot.png")
    
    # Gráfica 8: α̇(t) = ω₁(t)
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, alpha_dot_vals, 'r-', linewidth=2.5, label='α̇ = ω₁')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Tiempo (s)', fontsize=13)
    plt.ylabel('α̇ = ω₁ (rad/s)', fontsize=13)
    plt.title('Velocidad Angular α̇(t) = ω₁(t)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('imagenes/grafica_alpha_dot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfica guardada: grafica_alpha_dot.png")
    
    # Gráfica 9: V_a/f(t)
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, V_af_vals, 'orange', linewidth=2.5)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Tiempo (s)', fontsize=13)
    plt.ylabel('V_a/f (m/s)', fontsize=13)
    plt.title('Velocidad Relativa V_a/f(δ(t))', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('imagenes/grafica_V_af.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfica guardada: grafica_V_af.png")
    
    # ========== GRÁFICAS INDIVIDUALES DE ACELERACIONES ==========
    
    # Gráfica 10: δ̈(t) - Aceleración del motor
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, delta_ddot_vals, 'b-', linewidth=2.5)
    plt.xlabel('Tiempo (s)', fontsize=13)
    plt.ylabel('δ̈ (rad/s²)', fontsize=13)
    plt.title('Aceleración Angular del Motor δ̈ = 0 (velocidad constante)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([-0.5, 0.5])
    plt.tight_layout()
    plt.savefig('imagenes/grafica_delta_ddot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfica guardada: grafica_delta_ddot.png")
    
    # Gráfica 11: α̈(t)
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, alpha_ddot_vals, 'r-', linewidth=2.5)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Tiempo (s)', fontsize=13)
    plt.ylabel('α̈ (rad/s²)', fontsize=13)
    plt.title('Aceleración Angular α̈(t) = dω₁/dt', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('imagenes/grafica_alpha_ddot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfica guardada: grafica_alpha_ddot.png")
    
    # Gráfica 12: Comparación de velocidades angulares
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, delta_dot_vals, 'b-', linewidth=2.5, label='δ̇ (motor)')
    plt.plot(t_vals, alpha_dot_vals, 'r-', linewidth=2.5, label='α̇ = ω₁')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Tiempo (s)', fontsize=13)
    plt.ylabel('Velocidad Angular (rad/s)', fontsize=13)
    plt.title('Comparación de Velocidades Angulares', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('imagenes/grafica_comparacion_velocidades.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfica guardada: grafica_comparacion_velocidades.png")
    
    # Gráfica 13: Magnitud de α̈
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, np.abs(alpha_ddot_vals), 'darkred', linewidth=2.5)
    plt.fill_between(t_vals, 0, np.abs(alpha_ddot_vals), alpha=0.3, color='darkred')
    plt.xlabel('Tiempo (s)', fontsize=13)
    plt.ylabel('|α̈| (rad/s²)', fontsize=13)
    plt.title('Magnitud de la Aceleración Angular |α̈(t)|', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('imagenes/grafica_alpha_ddot_magnitud.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfica guardada: grafica_alpha_ddot_magnitud.png")
    
    # ========== GRÁFICAS DEL MARTILLO ==========
    
    # Gráfica 14: Velocidad del martillo Ẋ_E(t)
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, X_E_dot_vals, 'purple', linewidth=2.5)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Tiempo (s)', fontsize=13)
    plt.ylabel('Ẋ_E (m/s)', fontsize=13)
    plt.title('Velocidad del Martillo Ẋ_E(t)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('imagenes/grafica_martillo_velocidad.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfica guardada: grafica_martillo_velocidad.png")
    
    # Gráfica 15: Aceleración del martillo Ẍ_E(t)
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, X_E_ddot_vals, 'darkgreen', linewidth=2.5)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Tiempo (s)', fontsize=13)
    plt.ylabel('Ẍ_E (m/s²)', fontsize=13)
    plt.title('Aceleración del Martillo Ẍ_E(t)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('imagenes/grafica_martillo_aceleracion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfica guardada: grafica_martillo_aceleracion.png")
    
    print("\n" + "="*80)

