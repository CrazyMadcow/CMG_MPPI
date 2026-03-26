"""
Spacecraft Attitude Dynamics with Pyramid-type Single-Gimbal CMGs
Reference: Oghim et al., "Deep reinforcement learning-based attitude control
           for spacecraft using control moment gyros", ASR 2025
"""

import numpy as np


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------

def quat_mult(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product of two unit quaternions p ⊗ q.

    Convention: q = [qw, qx, qy, qz] (scalar first).
    """
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate (= inverse for unit quaternion): [qw, -qv]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def quat_kinematics(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Time derivative of quaternion given body angular velocity ω.

    q̇ = (1/2) q ⊗ Ω,   Ω = [0, ωx, ωy, ωz]   (Eq. 1)
    """
    Omega = np.array([0.0, omega[0], omega[1], omega[2]])
    return 0.5 * quat_mult(q, Omega)


def attitude_error_quat(q: np.ndarray, q_d: np.ndarray):
    """Compute attitude-error quaternion and its vector part.

    qe = qd* ⊗ q   (Eq. 2)
    Returns (qe, qve) where qe = [qwe, qve].
    """
    qe = quat_mult(quat_conj(q_d), q)
    return qe, qe[1:]


# ---------------------------------------------------------------------------
# CMG Jacobian (pyramid-type cluster)
# ---------------------------------------------------------------------------

def cmg_angular_momentum(delta: np.ndarray, h: float, beta: float) -> np.ndarray:
    """Total angular momentum vector of 4-CMG pyramid cluster.

    Eq. 9:
        Hcmg = h * [-cosβ sinδ1 - cosδ2 + cosβ sinδ3 + cosδ4 ]
                    [ cosδ1 - cosβ sinδ2 - cosδ3 + cosβ sinδ4 ]
                    [ sinβ sinδ1 + sinβ sinδ2 + sinβ sinδ3 + sinβ sinδ4]

    Args:
        delta: gimbal angles [δ1, δ2, δ3, δ4] in radians
        h:     angular momentum magnitude of each flywheel [N·m·s]
        beta:  skew angle of the pyramid [rad]
    """
    d1, d2, d3, d4 = delta
    cb, sb = np.cos(beta), np.sin(beta)
    return h * np.array([
        -cb*np.sin(d1) - np.cos(d2) + cb*np.sin(d3) + np.cos(d4),
         np.cos(d1) - cb*np.sin(d2) - np.cos(d3) + cb*np.sin(d4),
         sb*(np.sin(d1) + np.sin(d2) + np.sin(d3) + np.sin(d4)),
    ])


def cmg_jacobian(delta: np.ndarray, h: float, beta: float) -> np.ndarray:
    """Jacobian A(δ) such that Ḣcmg = A(δ) δ̇.

    Eq. 11:
        A(δ) = h * [-cosβ cosδ1   sinδ2    cosβ cosδ3   -sinδ4 ]
                    [-sinδ1       -cosβ cosδ2   sinδ3    cosβ cosδ4]
                    [ sinβ cosδ1   sinβ cosδ2   sinβ cosδ3  sinβ cosδ4]

    Returns shape (3, 4).
    """
    d1, d2, d3, d4 = delta
    cb, sb = np.cos(beta), np.sin(beta)
    return h * np.array([
        [-cb*np.cos(d1),  np.sin(d2),  cb*np.cos(d3), -np.sin(d4)],
        [-np.sin(d1),    -cb*np.cos(d2), np.sin(d3),   cb*np.cos(d4)],
        [ sb*np.cos(d1),  sb*np.cos(d2), sb*np.cos(d3), sb*np.cos(d4)],
    ])


def singularity_measure(A: np.ndarray) -> float:
    """Singularity measure m = det(A A^T).  m → 0 at singularities."""
    return float(np.linalg.det(A @ A.T))


# ---------------------------------------------------------------------------
# Steering laws
# ---------------------------------------------------------------------------

def pseudo_inverse_steering(A: np.ndarray, tau_d: np.ndarray) -> np.ndarray:
    """Pseudo-inverse steering law.

    δ̇ = A^T (A A^T)^{-1} τd   (Eq. 12)
    """
    AAT = A @ A.T
    return A.T @ np.linalg.solve(AAT, tau_d)


def gsri_steering(A: np.ndarray, tau_c: np.ndarray, t: float,
                  k_sr: float = 0.01) -> np.ndarray:
    """Generalised Singularity-Robust Inverse (GSRI) steering law.

    δ̇ = A^T (A A^T + λ E)^{-1} τc   (Eq. 27)

    where:
        λ = 0.01 exp(-10 det(A A^T))
        E has off-diagonal elements εi = 0.01 sin(0.5π t + φi)
        φ2 = π/2, φ3 = π  (1-indexed; ε1, ε2, ε3 → indices (0,1), (0,2), (1,2))
    """
    AAT = A @ A.T
    lam = k_sr * np.exp(-10.0 * np.linalg.det(AAT))
    eps1 = 0.01 * np.sin(0.5 * np.pi * t)
    eps2 = 0.01 * np.sin(0.5 * np.pi * t + np.pi / 2)
    eps3 = 0.01 * np.sin(0.5 * np.pi * t + np.pi)
    E = np.array([
        [1.0,  eps3, eps2],
        [eps3, 1.0,  eps1],
        [eps2, eps1, 1.0],
    ])
    return A.T @ np.linalg.solve(AAT + lam * E, tau_c)


def momentum_recovery_steering(A: np.ndarray, tau_c: np.ndarray,
                                delta: np.ndarray, delta0: np.ndarray,
                                dt: float, eta: float = 0.1) -> np.ndarray:
    """Steering law with null-motion momentum recovery.

    δ̇ = A⁺ τc + η (A⁺ A − I₄) g   (Eq. 26)

    where A⁺ = A^T (A A^T)^{-1} and g = (δ − δ0) / Δt.
    The null-motion term drives gimbal angles back to δ0.
    """
    A_pinv = A.T @ np.linalg.solve(A @ A.T, np.eye(3))   # shape (4, 3)
    g = (delta - delta0) / dt
    null_proj = A_pinv @ A - np.eye(4)                    # (A⁺A − I)
    return A_pinv @ tau_c + eta * (null_proj @ g)


# ---------------------------------------------------------------------------
# Spacecraft dynamics
# ---------------------------------------------------------------------------

class SpacecraftCMG:
    """Rigid spacecraft with a 4-CMG pyramid cluster.

    State vector x = [q(4), ω(3), δ(4)] with total size 11.

    Reference parameters (Table 2, Oghim et al. 2025):
        J    = diag(21400, 20100, 5500)  [kg·m²]
        h    = 1000                       [N·m·s]
        β    = 53.13°
        δ₀   = [45, −45, 45, −45]°
        δ̇max = 1.0                        [rad/s]
    """

    def __init__(
        self,
        J: np.ndarray | None = None,
        h: float = 1000.0,
        beta_deg: float = 53.13,
        delta0_deg: np.ndarray | None = None,
        delta_dot_max: float = 1.0,
    ):
        self.J = J if J is not None else np.diag([21400.0, 20100.0, 5500.0])
        self.J_inv = np.linalg.inv(self.J)
        self.h = h
        self.beta = np.deg2rad(beta_deg)
        self.delta0 = np.deg2rad(
            delta0_deg if delta0_deg is not None
            else np.array([45.0, -45.0, 45.0, -45.0])
        )
        self.delta_dot_max = delta_dot_max

        # State
        self.q = np.array([1.0, 0.0, 0.0, 0.0])   # unit quaternion [qw, qx, qy, qz]
        self.omega = np.zeros(3)                    # body angular velocity [rad/s]
        self.delta = self.delta0.copy()             # gimbal angles [rad]

    # ------------------------------------------------------------------
    def reset(self,
              q: np.ndarray | None = None,
              omega: np.ndarray | None = None,
              delta: np.ndarray | None = None):
        """Reset spacecraft state."""
        self.q = quat_normalize(q) if q is not None else np.array([1.0, 0.0, 0.0, 0.0])
        self.omega = omega.copy() if omega is not None else np.zeros(3)
        self.delta = delta.copy() if delta is not None else self.delta0.copy()

    # ------------------------------------------------------------------
    @property
    def A(self) -> np.ndarray:
        """Current Jacobian A(δ), shape (3, 4)."""
        return cmg_jacobian(self.delta, self.h, self.beta)

    @property
    def H_cmg(self) -> np.ndarray:
        """Current CMG angular momentum vector, shape (3,)."""
        return cmg_angular_momentum(self.delta, self.h, self.beta)

    @property
    def singularity_measure(self) -> float:
        return singularity_measure(self.A)

    # ------------------------------------------------------------------
    def derivatives(self, q, omega, delta, delta_dot, tau_ext=None):
        """Compute time derivatives of the full state.

        Spacecraft EOM  (Eq. 6, 8):
            Jω̇ = τ − ω × Jω
            τ  = −Ḣcmg − ω × Hcmg   (Eq. 7)
               = −A δ̇ − ω × Hcmg

        Returns (q_dot, omega_dot, delta_dot).
        """
        A_mat = cmg_jacobian(delta, self.h, self.beta)
        H = cmg_angular_momentum(delta, self.h, self.beta)

        H_dot = A_mat @ delta_dot
        tau = -H_dot - np.cross(omega, H)
        if tau_ext is not None:
            tau = tau + tau_ext

        q_dot = quat_kinematics(q, omega)
        omega_dot = self.J_inv @ (tau - np.cross(omega, self.J @ omega))

        return q_dot, omega_dot, delta_dot

    # ------------------------------------------------------------------
    def step(self, delta_dot_cmd: np.ndarray, dt: float,
             tau_ext: np.ndarray | None = None):
        """Integrate one time step with RK4.

        Args:
            delta_dot_cmd: commanded gimbal rates [rad/s], clipped to ±δ̇max
            dt:            time step [s]
            tau_ext:       external disturbance torque [N·m] (optional)

        Returns:
            state dict with q, omega, delta, tau, singularity_measure
        """
        delta_dot = np.clip(delta_dot_cmd, -self.delta_dot_max, self.delta_dot_max)

        def f(q, omega, delta):
            return self.derivatives(q, omega, delta, delta_dot, tau_ext)

        # RK4
        q0, w0, d0 = self.q, self.omega, self.delta

        k1q, k1w, k1d = f(q0, w0, d0)
        k2q, k2w, k2d = f(q0 + 0.5*dt*k1q, w0 + 0.5*dt*k1w, d0 + 0.5*dt*k1d)
        k3q, k3w, k3d = f(q0 + 0.5*dt*k2q, w0 + 0.5*dt*k2w, d0 + 0.5*dt*k2d)
        k4q, k4w, k4d = f(q0 + dt*k3q,     w0 + dt*k3w,     d0 + dt*k3d)

        self.q     = quat_normalize(q0 + (dt/6)*(k1q + 2*k2q + 2*k3q + k4q))
        self.omega = w0 + (dt/6)*(k1w + 2*k2w + 2*k3w + k4w)
        self.delta = d0 + (dt/6)*(k1d + 2*k2d + 2*k3d + k4d)

        # torque actually applied (evaluated at current state)
        A_mat = self.A
        H = self.H_cmg
        tau = -(A_mat @ delta_dot) - np.cross(self.omega, H)

        return {
            "q":                  self.q.copy(),
            "omega":              self.omega.copy(),
            "delta":              self.delta.copy(),
            "tau":                tau,
            "singularity_measure": self.singularity_measure,
        }

    # ------------------------------------------------------------------
    def attitude_error(self, q_d: np.ndarray):
        """Return attitude error quaternion qe and its vector part."""
        return attitude_error_quat(self.q, q_d)


# ---------------------------------------------------------------------------
# Conventional controller (Phase 2, Eq. 25)
# ---------------------------------------------------------------------------

def conventional_controller(qe: np.ndarray, omega: np.ndarray,
                              J: np.ndarray,
                              omega_n: float = 3.0,
                              zeta: float = 0.9,
                              T: float = 10.0,
                              torque_limit: float = 3000.0,
                              omega_max: float = np.deg2rad(30.0)) -> np.ndarray:
    """Quaternion-error feedback control law (Eq. 25).

    τc = −J { 2k sat_{Li}(qve) + (1/T)∫qve + c ω }

    Here the integral term is omitted for a simpler PD version;
    pass the returned torque through a saturation limit.

    Args:
        qe:         error quaternion [qwe, qve]
        omega:      current angular velocity [rad/s]
        J:          inertia tensor
        omega_n:    natural frequency [rad/s]
        zeta:       damping ratio
        T:          time constant for integral [s]
        torque_limit: saturation limit [N·m]
        omega_max:  maximum slew rate [rad/s]
    Returns:
        control torque [N·m], shape (3,)
    """
    qve = qe[1:]  # vector part of error quaternion

    k = omega_n**2 + 2*zeta*omega_n / T
    c = 2*zeta*omega_n + 1.0/T

    # Variable limiter per axis
    def sat(e, L):
        return np.clip(e, -L, L)

    tau_c = np.zeros(3)
    for i in range(3):
        ai = np.abs(qve[i])
        Li = (c / (2*k)) * min(np.sqrt(4*k * ai), omega_max)
        tau_c[i] = -J[i, i] * (2*k * sat(qve[i], Li) + c*omega[i])

    return np.clip(tau_c, -torque_limit, torque_limit)


# ---------------------------------------------------------------------------
# Quick simulation demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dt = 0.1          # [s]
    T_max = 30.0      # [s]
    steps = int(T_max / dt)

    sc = SpacecraftCMG()

    # Desired attitude: 70° yaw rotation
    angle = np.deg2rad(70.0)
    q_d = np.array([np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)])

    # Log arrays
    time = np.linspace(0, T_max, steps)
    log_qe   = np.zeros((steps, 4))
    log_omega = np.zeros((steps, 3))
    log_delta_dot = np.zeros((steps, 4))
    log_sm   = np.zeros(steps)

    sc.reset()

    bq = 0.04   # tolerance to switch to phase 2

    for i, t in enumerate(time):
        qe, qve = sc.attitude_error(q_d)

        if np.linalg.norm(qve) > bq:
            # Phase 1: GSRI steering with PD torque (conventional law for demo)
            tau_d = conventional_controller(qe, sc.omega, sc.J)
            A = sc.A
            delta_dot = gsri_steering(A, tau_d, t)
        else:
            # Phase 2: momentum recovery steering
            tau_d = conventional_controller(qe, sc.omega, sc.J)
            A = sc.A
            delta_dot = momentum_recovery_steering(
                A, tau_d, sc.delta, sc.delta0, dt, eta=0.1
            )

        state = sc.step(delta_dot, dt)

        log_qe[i]        = qe
        log_omega[i]     = np.rad2deg(state["omega"])
        log_delta_dot[i] = delta_dot
        log_sm[i]        = state["singularity_measure"]

    # ------------------------------------------------------------------
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].set_title("Attitude Error Quaternion")
    labels_q = ["qe_w", "qe_x", "qe_y", "qe_z"]
    for j, lbl in enumerate(labels_q):
        axes[0].plot(time, log_qe[:, j], label=lbl)
    axes[0].axhline(bq, color="k", linestyle="--", linewidth=0.8, label="bq")
    axes[0].legend(ncol=2); axes[0].set_ylabel("[-]"); axes[0].grid(True)

    axes[1].set_title("Angular Rates")
    for j, lbl in enumerate(["ωx", "ωy", "ωz"]):
        axes[1].plot(time, log_omega[:, j], label=lbl)
    axes[1].legend(); axes[1].set_ylabel("[deg/s]"); axes[1].grid(True)

    axes[2].set_title("Singularity Measure  m = det(A Aᵀ)")
    axes[2].plot(time, log_sm, color="tab:orange")
    axes[2].set_ylabel("m"); axes[2].set_xlabel("time [s]"); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("spacecraft_cmg_demo.png", dpi=150)
    plt.show()
    print("Done. Plot saved to spacecraft_cmg_demo.png")
