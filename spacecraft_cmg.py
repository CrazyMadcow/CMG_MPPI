"""
Spacecraft Attitude Dynamics with Pyramid-type Single-Gimbal CMGs
Reference: Oghim et al., "Deep reinforcement learning-based attitude control
           for spacecraft using control moment gyros", ASR 2025

Module layout (designed for MPPI extension):
  SpacecraftCMG    – dynamics (state: q, ω, δ)
  AttitudeController – conventional quaternion-error feedback (Eq. 25)
  CMGSteering      – base steering interface
    GSRISteering   – Generalised Singularity-Robust Inverse (Eq. 27)
    MomentumRecoverySteering – null-motion momentum recovery (Eq. 26)
    MPPISteering   – placeholder; override compute() for MPPI
"""

from __future__ import annotations
from typing import Optional
import numpy as np


# ============================================================
#  Quaternion utilities
# ============================================================

def quat_mult(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product p ⊗ q.  Convention: q = [qw, qx, qy, qz]."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse) of a unit quaternion: [qw, -qv]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def quat_kinematics(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """q̇ = (1/2) q ⊗ Ω,   Ω = [0, ωx, ωy, ωz]   (Eq. 1)"""
    Omega = np.array([0.0, omega[0], omega[1], omega[2]])
    return 0.5 * quat_mult(q, Omega)


def attitude_error_quat(q: np.ndarray, q_d: np.ndarray):
    """qe = qd* ⊗ q  (Eq. 2).  Returns (qe, qve) where qe=[qwe,qve]."""
    qe = quat_mult(quat_conj(q_d), q)
    return qe, qe[1:]


# ============================================================
#  CMG Jacobian & angular momentum (pyramid cluster)
# ============================================================

def cmg_angular_momentum(delta: np.ndarray, h: float, beta: float) -> np.ndarray:
    """H_cmg vector of 4-CMG pyramid cluster  (Eq. 9).

    Args:
        delta : gimbal angles [δ1,δ2,δ3,δ4]  [rad]
        h     : angular momentum per flywheel  [N·m·s]
        beta  : skew angle                     [rad]
    """
    d1, d2, d3, d4 = delta
    cb, sb = np.cos(beta), np.sin(beta)
    return h * np.array([
        -cb*np.sin(d1) - np.cos(d2) + cb*np.sin(d3) + np.cos(d4),
         np.cos(d1) - cb*np.sin(d2) - np.cos(d3) + cb*np.sin(d4),
         sb*(np.sin(d1) + np.sin(d2) + np.sin(d3) + np.sin(d4)),
    ])


def cmg_jacobian(delta: np.ndarray, h: float, beta: float) -> np.ndarray:
    """Jacobian A(δ) s.t. Ḣ_cmg = A(δ) δ̇  (Eq. 11).  Shape (3, 4)."""
    d1, d2, d3, d4 = delta
    cb, sb = np.cos(beta), np.sin(beta)
    return h * np.array([
        [-cb*np.cos(d1),  np.sin(d2),    cb*np.cos(d3),  -np.sin(d4)],
        [-np.sin(d1),    -cb*np.cos(d2), np.sin(d3),      cb*np.cos(d4)],
        [ sb*np.cos(d1),  sb*np.cos(d2), sb*np.cos(d3),   sb*np.cos(d4)],
    ])


def singularity_measure(A: np.ndarray, h: float = 1.0) -> float:
    """Normalised singularity measure m = det((A/h)(A/h)ᵀ).

    Normalising by h keeps m in the range [0, ~3] for a 4-CMG pyramid,
    matching the scale used in Oghim et al. (2025) Fig. 8.
    m → 0 at a singular gimbal configuration.
    """
    An = A / h
    return float(np.linalg.det(An @ An.T))


# ============================================================
#  Spacecraft + CMG Dynamics
# ============================================================

class SpacecraftCMG:
    """Rigid spacecraft with a 4-SGCMG pyramid cluster.

    State: q (quaternion, scalar-first), ω (body angular velocity), δ (gimbal angles).

    Default parameters match Table 2 in Oghim et al. 2025:
        J    = diag(21400, 20100, 5500) kg·m²
        h    = 1000 N·m·s
        β    = 53.13°
        δ₀   = [45, −45, 45, −45]° → zero net angular momentum
        δ̇max = 1.0 rad/s
    """

    def __init__(
        self,
        J: Optional[np.ndarray] = None,
        h: float = 1000.0,
        beta_deg: float = 53.13,
        delta0_deg: Optional[np.ndarray] = None,
        delta_dot_max: float = 1.0,
        dt: float = 0.1,
    ):
        self.J     = J if J is not None else np.diag([21400.0, 20100.0, 5500.0])
        self.J_inv = np.linalg.inv(self.J)
        self.h     = h
        self.beta  = np.deg2rad(beta_deg)
        self.delta0 = np.deg2rad(
            delta0_deg if delta0_deg is not None
            else np.array([45.0, -45.0, 45.0, -45.0])
        )
        self.delta_dot_max = delta_dot_max
        self.dt = dt

        # state
        self.q     = np.array([1.0, 0.0, 0.0, 0.0])
        self.omega = np.zeros(3)
        self.delta = self.delta0.copy()

    # ------------------------------------------------------------------
    def reset(self, q=None, omega=None, delta=None):
        self.q     = quat_normalize(q) if q is not None else np.array([1.0, 0.0, 0.0, 0.0])
        self.omega = omega.copy()      if omega is not None else np.zeros(3)
        self.delta = delta.copy()      if delta is not None else self.delta0.copy()

    # ------------------------------------------------------------------
    @property
    def A(self) -> np.ndarray:
        return cmg_jacobian(self.delta, self.h, self.beta)

    @property
    def H_cmg(self) -> np.ndarray:
        return cmg_angular_momentum(self.delta, self.h, self.beta)

    @property
    def singularity_measure(self) -> float:
        return singularity_measure(self.A, self.h)

    # ------------------------------------------------------------------
    def _derivatives(self, q, omega, delta, delta_dot, tau_ext=None):
        """EOM: Jω̇ = τ − ω×Jω,  τ = −Ḣcmg − ω×Hcmg  (Eq. 6-8)."""
        A_mat = cmg_jacobian(delta, self.h, self.beta)
        H     = cmg_angular_momentum(delta, self.h, self.beta)

        tau = -(A_mat @ delta_dot) - np.cross(omega, H)
        if tau_ext is not None:
            tau = tau + tau_ext

        q_dot     = quat_kinematics(q, omega)
        omega_dot = self.J_inv @ (tau - np.cross(omega, self.J @ omega))

        return q_dot, omega_dot, delta_dot   # δ̇ is already the "derivative of δ"

    # ------------------------------------------------------------------
    def step(self, delta_dot_cmd: np.ndarray, tau_ext: Optional[np.ndarray] = None) -> dict:
        """Integrate one time step with RK4.

        Args:
            delta_dot_cmd : commanded gimbal rates [rad/s], clipped to ±δ̇max
            tau_ext       : optional external disturbance torque [N·m]

        Returns dict with keys: q, omega, delta, tau, singularity_measure
        """
        dt = self.dt
        dd = np.clip(delta_dot_cmd, -self.delta_dot_max, self.delta_dot_max)

        def f(q, w, d):
            return self._derivatives(q, w, d, dd, tau_ext)

        q0, w0, d0 = self.q.copy(), self.omega.copy(), self.delta.copy()

        k1q, k1w, k1d = f(q0,             w0,             d0)
        k2q, k2w, k2d = f(q0+.5*dt*k1q, w0+.5*dt*k1w, d0+.5*dt*k1d)
        k3q, k3w, k3d = f(q0+.5*dt*k2q, w0+.5*dt*k2w, d0+.5*dt*k2d)
        k4q, k4w, k4d = f(q0+dt*k3q,   w0+dt*k3w,   d0+dt*k3d)

        self.q     = quat_normalize(q0 + (dt/6)*(k1q+2*k2q+2*k3q+k4q))
        self.omega = w0 + (dt/6)*(k1w+2*k2w+2*k3w+k4w)
        self.delta = d0 + (dt/6)*(k1d+2*k2d+2*k3d+k4d)

        A_mat = self.A
        H     = self.H_cmg
        tau   = -(A_mat @ dd) - np.cross(self.omega, H)

        return {
            "q":                   self.q.copy(),
            "omega":               self.omega.copy(),
            "delta":               self.delta.copy(),
            "tau":                 tau,
            "singularity_measure": self.singularity_measure,
        }

    # ------------------------------------------------------------------
    def attitude_error(self, q_d: np.ndarray):
        """Return (qe, qve): error quaternion and its vector part."""
        return attitude_error_quat(self.q, q_d)


# ============================================================
#  Attitude Controller  (Eq. 25, Wie et al. 2001)
# ============================================================

class AttitudeController:
    """Quaternion-error feedback controller with variable limiter.

    τc = −J { 2k sat_{Li}(q_ve + (1/T)∫q_ve) + c ω }

    Variable limiter per axis:
        L_i = (c/2k) · min( √(4 a_i |q_{ve,i}|), ω_max )
        a_i = torque_limit / J_i

    Ref: Wie et al. (2001); used as Phase-2 law in Oghim et al. (2025).
    """

    def __init__(
        self,
        J: np.ndarray,
        omega_n: float = 3.0,       # natural frequency [rad/s]
        zeta: float = 0.9,          # damping ratio
        T_int: float = 10.0,        # integral time constant [s]
        torque_limit: float = 3000.0,  # saturation [N·m]
        omega_max_deg: float = 30.0,   # max slew rate [deg/s]
        dt: float = 0.1,
    ):
        self.J           = J
        self.k           = omega_n**2 + 2*zeta*omega_n / T_int
        self.c           = 2*zeta*omega_n + 1.0 / T_int
        self.T_int       = T_int
        self.torque_lim  = torque_limit
        self.omega_max   = np.deg2rad(omega_max_deg)
        self.dt          = dt
        self._integral   = np.zeros(3)   # ∫q_ve dt

    def reset(self):
        self._integral = np.zeros(3)

    def compute_torque(self, qe: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Compute control torque.

        Args:
            qe    : error quaternion [qwe, qve_x, qve_y, qve_z]
            omega : current body angular velocity [rad/s]

        Returns:
            τc [N·m], shape (3,)
        """
        qve = qe[1:]
        self._integral += qve * self.dt

        k, c  = self.k, self.c
        tau_c = np.zeros(3)

        for i in range(3):
            ai  = self.torque_lim / self.J[i, i]   # max angular accel per axis
            Li  = (c / (2*k)) * min(np.sqrt(4*ai*abs(qve[i])), self.omega_max)

            inner = qve[i] + self._integral[i] / self.T_int
            tau_c[i] = -self.J[i, i] * (2*k * np.clip(inner, -Li, Li) + c*omega[i])

        return np.clip(tau_c, -self.torque_lim, self.torque_lim)


# ============================================================
#  Steering Laws
# ============================================================

class CMGSteering:
    """Abstract base class for CMG steering laws.

    All subclasses expose the same interface:
        delta_dot = steering.compute(sc, tau_sc, t)

    Args:
        sc     : SpacecraftCMG instance  (A, delta, delta0, omega, H_cmg, …)
        tau_sc : desired spacecraft torque [N·m]  (output of AttitudeController)
        t      : current simulation time [s]

    Physics note
    ============
    The CMG torque acting on the spacecraft body is (Eq. 7):
        τ = −Ḣ_cmg − ω × H_cmg = −A δ̇ − ω × H_cmg
    Steering laws invert this relation so that τ = tau_sc:
        A δ̇ = −tau_sc − ω × H_cmg

    This sign handling is done internally in every subclass, so the
    caller always passes the *desired spacecraft torque* (positive value
    means the spacecraft body receives positive angular acceleration).

    Swapping steering laws (e.g. for MPPI) requires zero changes to the
    simulation loop:
        steering = MPPISteering(...)
        delta_dot = steering.compute(sc, tau_sc, t)
    """

    def compute(self, sc: SpacecraftCMG, tau_sc: np.ndarray, t: float = 0.0) -> np.ndarray:
        raise NotImplementedError


class GSRISteering(CMGSteering):
    """Generalised Singularity-Robust Inverse steering law  (Eq. 27).

    Solves  A δ̇ = rhs  via the SR-regularised pseudo-inverse, where
        rhs = −tau_sc − ω × H_cmg
        λ   = k_sr · exp(−10 det(AAᵀ))
    E is a 3×3 matrix with sinusoidally dithered off-diagonal entries.
    """

    def __init__(self, k_sr: float = 0.01):
        self.k_sr = k_sr

    def compute(self, sc: SpacecraftCMG, tau_sc: np.ndarray, t: float = 0.0) -> np.ndarray:
        A   = sc.A
        H   = sc.H_cmg
        rhs = -tau_sc - np.cross(sc.omega, H)   # A δ̇ = −τ_sc − ω×H

        AAT = A @ A.T
        lam = self.k_sr * np.exp(-10.0 * np.linalg.det(AAT))

        eps1 = 0.01 * np.sin(0.5*np.pi*t)
        eps2 = 0.01 * np.sin(0.5*np.pi*t + np.pi/2)
        eps3 = 0.01 * np.sin(0.5*np.pi*t + np.pi)
        E = np.array([
            [1.0,  eps3, eps2],
            [eps3, 1.0,  eps1],
            [eps2, eps1, 1.0],
        ])

        return A.T @ np.linalg.solve(AAT + lam*E, rhs)


class MomentumRecoverySteering(CMGSteering):
    """Null-motion momentum recovery steering law  (Eq. 26).

    δ̇ = A⁺ rhs + η (A⁺A − I₄) g
    rhs = −tau_sc − ω × H_cmg
    g   = (δ − δ₀) / Δt   drives gimbal angles back to δ₀
    """

    def __init__(self, eta: float = 0.1):
        self.eta = eta

    def compute(self, sc: SpacecraftCMG, tau_sc: np.ndarray, t: float = 0.0) -> np.ndarray:
        A   = sc.A
        H   = sc.H_cmg
        rhs = -tau_sc - np.cross(sc.omega, H)   # A δ̇ = −τ_sc − ω×H

        A_pinv   = A.T @ np.linalg.solve(A @ A.T, np.eye(3))   # (4,3)
        g        = (sc.delta - sc.delta0) / sc.dt
        null_prj = A_pinv @ A - np.eye(4)                       # (A⁺A − I)
        return A_pinv @ rhs + self.eta * (null_prj @ g)


class MPPISteering(CMGSteering):
    """Placeholder for MPPI-based steering.

    MPPI samples gimbal-rate trajectories directly and selects the
    optimal one — it does not need a pre-computed desired torque,
    though tau_sc is available for cost-function shaping.

    The method signature is identical to other steering classes so
    the simulation loop requires zero changes when switching laws:
        steering = MPPISteering(...)
        delta_dot = steering.compute(sc, tau_sc, t)
    """

    def compute(self, sc: SpacecraftCMG, tau_sc: np.ndarray, t: float = 0.0) -> np.ndarray:
        # TODO: implement MPPI rollouts here
        raise NotImplementedError("MPPISteering.compute() is not yet implemented.")
