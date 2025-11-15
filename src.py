
import numpy as np
from numpy.linalg import inv, solve, norm
import sys
import math
import time
import random

def make_linear_data(n=100, d=2, noise=0.5, seed=0):
    np.random.seed(seed)
    X = np.random.randn(n, d)
    true_w = np.arange(1, d+1).astype(float)  # [1,2,...]
    y = X @ true_w + 2.0 + noise * np.random.randn(n)
    # Augmented X with bias
    X_aug = np.hstack([np.ones((n,1)), X])
    W_true = np.concatenate([[2.0], true_w])
    return X, X_aug, y, W_true

def make_logistic_data(n=200, d=2, seed=1):
    np.random.seed(seed)
    X = np.random.randn(n, d)
    # Make labels roughly separable
    w = np.array([1.5, -2.0])
    logits = X @ w + 0.5
    p = 1/(1+np.exp(-logits))
    y = (p > 0.5).astype(int)
    X_aug = np.hstack([np.ones((n,1)), X])
    W_true = np.concatenate([[0.5], w])
    return X, X_aug, y, W_true

def show_line():
    print('-'*80)

def mse_loss(W, X_aug, y):
    n = len(y)
    preds = X_aug @ W
    err = preds - y
    loss = 0.5 * np.mean(err**2)
    grad = (X_aug.T @ err) / len(y)
    # Hessian = (X^T X)/N
    H = (X_aug.T @ X_aug) / len(y)
    return loss, grad, H

def logistic_loss(W, X_aug, y):
    n = len(y)
    z = X_aug @ W
    p = 1/(1+np.exp(-z))
    # cross-entropy average
    eps = 1e-12
    loss = -np.mean(y * np.log(p+eps) + (1-y)*np.log(1-p+eps))
    grad = X_aug.T @ (p - y) / n
    # Hessian: X^T * diag(p(1-p)) * X / n
    S = p * (1-p)
    H = (X_aug.T * S) @ X_aug / n
    return loss, grad, H

def lasso_loss(W, X_aug, y, lam):
    n = len(y)
    preds = X_aug @ W
    err = preds - y
    loss = 0.5 * np.mean(err**2) + lam * np.sum(np.abs(W))
    grad_smooth = (X_aug.T @ err) / n
    # subgradient for l1:
    g = np.sign(W)
    # For zero entries, choose 0 in subgradient for pragmatic run
    g[W==0] = 0.0
    grad = grad_smooth + lam * g
    # No true Hessian because |.| not differentiable; use Hessian of smooth part:
    H = (X_aug.T @ X_aug) / n
    return loss, grad, H

# 1) Gradient Descent (constant / diminishing / backtracking)
def gradient_descent(f_loss, x0, X_aug, y, max_iters=200, alpha=0.1, diminishing=False, backtracking=False, verbose=True):
    x = x0.copy()
    history = []
    for k in range(max_iters):
        loss, grad, H = f_loss(x, X_aug, y)
        gnorm = norm(grad)
        history.append((k, loss, gnorm, x.copy()))
        if verbose:
            print(f"[GD] iter={k:3d} loss={loss:.6f} ||grad||={gnorm:.6e} params={x}")
        if gnorm < 1e-6:
            break
        if diminishing:
            a = alpha / (k+1)
        elif backtracking:
            # simple Armijo backtracking
            a = alpha
            c = 1e-4
            rho = 0.5
            fx = loss
            while True:
                x_new = x - a * grad
                fnew, _, _ = f_loss(x_new, X_aug, y)
                if fnew <= fx - c * a * (gnorm**2) or a < 1e-12:
                    break
                a = rho * a
        else:
            a = alpha
        x = x - a * grad
    return x, history

# 2) Exact line search for quadratic (for MSE)
def gd_exact_linesearch_quadratic(W0, X_aug, y, max_iters=100, verbose=True):
    W = W0.copy()
    n = len(y)
    history = []
    for k in range(max_iters):
        loss, grad, H = mse_loss(W, X_aug, y)
        gnorm = norm(grad)
        history.append((k, loss, gnorm, W.copy()))
        if verbose:
            print(f"[GD-exact] iter={k:3d} loss={loss:.6f} ||grad||={gnorm:.6e} params={W}")
        if gnorm < 1e-8:
            break
        # exact alpha for quadratic: alpha = (g^T g) / (g^T H g)
        denom = grad.T @ H @ grad
        if denom <= 0:
            alpha = 1e-6
        else:
            alpha = (grad.T @ grad) / denom
        W = W - alpha * grad
    return W, history

# 3) Adagrad (from Lecture 4)
def adagrad(f_loss, x0, X_aug, y, alpha=0.1, eps=1e-8, max_iters=200, verbose=True):
    x = x0.copy()
    G = np.zeros_like(x)
    history = []
    for k in range(max_iters):
        loss, grad, H = f_loss(x, X_aug, y)
        gnorm = norm(grad)
        history.append((k, loss, gnorm, x.copy()))
        if verbose:
            print(f"[Adagrad] iter={k:3d} loss={loss:.6f} ||grad||={gnorm:.6e} params={x}")
        if gnorm < 1e-8:
            break
        G += grad**2
        x = x - (alpha / (np.sqrt(G) + eps)) * grad
    return x, history

# 4) Newton's method (multi-d)
def newton_method(f_loss, x0, X_aug, y, max_iters=50, alpha=1.0, verbose=True):
    x = x0.copy()
    history = []
    for k in range(max_iters):
        loss, grad, H = f_loss(x, X_aug, y)
        gnorm = norm(grad)
        history.append((k, loss, gnorm, x.copy()))
        if verbose:
            print(f"[Newton] iter={k:3d} loss={loss:.6f} ||grad||={gnorm:.6e}")
            print(" grad:", grad)
            print(" Hessian:\n", H)
        if gnorm < 1e-10:
            break
        # regularize Hessian if singular
        try:
            delta = solve(H + 1e-6*np.eye(len(x)), grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(H + 1e-6*np.eye(len(x))) @ grad
        x = x - alpha * delta
    return x, history

# 5) BFGS (quasi-Newton) updating inverse Hessian approx
def bfgs(f_loss, x0, X_aug, y, max_iters=100, alpha=1.0, verbose=True):
    x = x0.copy()
    n = len(x)
    B_inv = np.eye(n)  # initial inverse Hessian approx
    history = []
    loss, grad, H = f_loss(x, X_aug, y)
    for k in range(max_iters):
        loss, grad, H = f_loss(x, X_aug, y)
        gnorm = norm(grad)
        history.append((k, loss, gnorm, x.copy()))
        if verbose:
            print(f"[BFGS] iter={k:3d} loss={loss:.6f} ||grad||={gnorm:.6e}")
        if gnorm < 1e-8:
            break
        p = - B_inv @ grad
        s = alpha * p
        x_new = x + s
        loss_new, grad_new, H_new = f_loss(x_new, X_aug, y)
        y_vec = grad_new - grad
        rho = 1.0 / (y_vec @ s + 1e-12)
        # B_inv <- (I - rho s y^T) B_inv (I - rho y s^T) + rho s s^T
        I = np.eye(n)
        Bs = (I - rho * np.outer(s, y_vec))
        B_inv = Bs @ B_inv @ Bs.T + rho * np.outer(s, s)
        x = x_new
        grad = grad_new
    return x, history

# 6) Subgradient method for Lasso (Lecture 6)
def subgradient_lasso(X_aug, y, lam=0.1, alpha=0.1, max_iters=200, verbose=True, diminishing=False):
    n_features = X_aug.shape[1]
    W = np.zeros(n_features)
    history = []
    for k in range(max_iters):
        loss, grad_smooth, H = lasso_loss(W, X_aug, y, lam)
        # subgradient g for L1:
        g_l1 = np.sign(W)
        g_l1[W==0] = 0.0  # choose 0 for zero entries (practical)
        grad = grad_smooth  # lasso_loss already included lam*sign in grad, but we call it anyway for clarity
        gnorm = norm(grad)
        history.append((k, loss, gnorm, W.copy()))
        if verbose:
            print(f"[Subgrad-LASSO] iter={k:3d} loss={loss:.6f} ||grad||={gnorm:.6e} W={W}")
        if gnorm < 1e-8:
            break
        if diminishing:
            a = alpha / (k+1)
        else:
            a = alpha
        # compute full subgradient used earlier
        # grad_smooth is gradient of smooth part; add lam * sign(W)
        full_sub = grad_smooth + lam * np.sign(W)
        full_sub[W==0] = grad_smooth[W==0]  # we prefer only smooth grad at zeros (practical)
        W = W - a * full_sub
    return W, history

# 7) EWMA (Lecture 5) - used as smoothing demonstration, not optimization step
def ewma_sequence(x_seq, beta=0.9):
    s = 0.0
    res = []
    for x in x_seq:
        s = beta * x + (1-beta) * s
        res.append(s)
    return np.array(res)

# 8) Lagrange multiplier solver for single equality constraint (Lecture 8)
def lagrange_equality_solver(f, grad_f, h_fun, grad_h, x0, lam0=0.0, max_iters=50, tol=1e-8, verbose=True):
    # Solve stationary system: grad_x (f + lambda h) = 0, and h(x)=0
    x = x0.copy()
    lam = lam0
    for k in range(max_iters):
        G = grad_f(x)
        Hc = grad_h(x)
        Lx = G + lam * Hc
        # Build linear system for Newton on saddle: [H  Hc; Hc^T 0] [dx; dlam] = -[Lx; h(x)]
        # We will approximate H by numerical Hessian of f (finite diff)
        eps = 1e-5
        n = len(x)
        H = np.zeros((n,n))
        fx, _, _ = mse_loss(x, np.eye(n), np.zeros(n))  # placeholder; we'll compute numeric Hessian of f
        # numeric Hessian of f via grad differences
        for i in range(n):
            ei = np.zeros(n); ei[i]=eps
            H[:,i] = (grad_f(x+ei) - grad_f(x-ei)) / (2*eps)
        hval = h_fun(x)
        # system:
        K = np.block([[H, Hc.reshape(-1,1)], [Hc.reshape(1,-1), np.zeros((1,1))]])
        rhs = -np.concatenate([Lx, [hval]])
        try:
            sol = solve(K, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
        dx = sol[:-1]
        dlam = sol[-1]
        x = x + dx
        lam = lam + dlam
        if verbose:
            print(f"[Lagrange] iter={k} x={x} lambda={lam} h={hval} ||dx||={norm(dx)}")
        if norm(dx) < tol and abs(hval) < tol:
            break
    return x, lam

# 9) KKT check util (Lecture 9)
def kkt_check(f_grad, g_list, h_list, x, lambdas, mus, tol=1e-6):
    # Stationarity:
    grad = f_grad(x)
    # sum multipliers * grads
    for i, g in enumerate(g_list):
        grad = grad + lambdas[i] * g(x)
    for j, h in enumerate(h_list):
        grad = grad + mus[j] * h(x)
    stationarity = norm(grad) < tol
    # primal feasibility:
    primal_ok = True
    for g in g_list:
        g_val = g(x)
        # Handle both scalar and array returns
        if isinstance(g_val, np.ndarray):
            if np.any(g_val > tol):
                primal_ok = False
        else:
            if g_val > tol:
                primal_ok = False
    for h in h_list:
        if abs(h(x)) > tol:
            primal_ok = False
    # dual feasibility:
    dual_ok = all([l >= -tol for l in lambdas])
    # complementary slackness approx:
    cs_ok = True
    for i, g in enumerate(g_list):
        if abs(lambdas[i]*g(x)) > tol:
            cs_ok = False
    return stationarity, primal_ok, dual_ok, cs_ok

# -------------------------------
# Terminal UI
# -------------------------------
def main_menu():
    print("Mini-project: Optimization algorithms (Lectures 1..9)")
    show_line()
    print("Pick an algorithm to run on default dataset:")
    print(" 1) Linear regression: closed-form (normal eqn) + GD (constant/diminishing/exact line-search).")
    print(" 2) Logistic regression: gradient descent + Adagrad.")
    print(" 3) Newton method on quadratic (MSE) and Newton general (logistic).")
    print(" 4) Quasi-Newton (BFGS).")
    print(" 5) LASSO via subgradient method.")
    print(" 6) EWMA demo.")
    print(" 7) Lagrange multiplier example (equality constraint).")
    print(" 8) KKT condition demo (simple 2D inequality case).")
    print(" 0) Exit.")
    show_line()

def run_option(choice):
    if choice == '1':
        # Linear regression
        X, X_aug, y, W_true = make_linear_data(n=200, d=3, noise=0.3)
        print("Closed-form solution (normal eqn):")
        W_closed = np.linalg.pinv(X_aug.T @ X_aug) @ (X_aug.T @ y)
        print(" W_closed:", W_closed, " true W:", W_true)
        print("\nNow run Gradient Descent (constant alpha=0.05):")
        W0 = np.zeros(X_aug.shape[1])
        W_gd, hist = gradient_descent(mse_loss, W0, X_aug, y, max_iters=200, alpha=0.05, diminishing=False, verbose=True)
        print("Final W:", W_gd)
        print("\nNow run GD with exact line search (quadratic):")
        W0b = np.zeros(X_aug.shape[1])
        W_exact, h2 = gd_exact_linesearch_quadratic(W0b, X_aug, y, max_iters=100, verbose=True)
        print("Exact-line-search final W:", W_exact)
        return

    if choice == '2':
        X, X_aug, y, W_true = make_logistic_data()
        W0 = np.zeros(X_aug.shape[1])
        print("Logistic regression via GD (alpha=0.5):")
        W_gd, h = gradient_descent(logistic_loss, W0, X_aug, y, max_iters=200, alpha=0.5, diminishing=False, verbose=True)
        print("Finished GD, final weights:", W_gd)
        print("\nAdagrad (alpha=0.5):")
        W0b = np.zeros(X_aug.shape[1])
        W_adg, h2 = adagrad(logistic_loss, W0b, X_aug, y, alpha=0.5, max_iters=200, verbose=True)
        print("Adagrad final weights:", W_adg)
        return

    if choice == '3':
        print("Newton method on MSE (quadratic):")
        X, X_aug, y, W_true = make_linear_data(n=100, d=2, noise=0.2)
        W0 = np.zeros(X_aug.shape[1])
        W_newton, h = newton_method(mse_loss, W0, X_aug, y, max_iters=20, verbose=True)
        print("Newton final:", W_newton)
        print("\nNewton method on logistic (using Hessian of logistic):")
        X2, X_aug2, y2, W_true2 = make_logistic_data()
        W0b = np.zeros(X_aug2.shape[1])
        W_newt2, h2 = newton_method(logistic_loss, W0b, X_aug2, y2, max_iters=20, verbose=True)
        print("Logistic Newton final:", W_newt2)
        return

    if choice == '4':
        print("BFGS demo on MSE loss")
        X, X_aug, y, W_true = make_linear_data(n=150, d=3, noise=0.4)
        W0 = np.zeros(X_aug.shape[1])
        W_bfgs, hist = bfgs(mse_loss, W0, X_aug, y, max_iters=50, verbose=True)
        print("BFGS final W:", W_bfgs)
        return

    if choice == '5':
        print("LASSO via subgradient method (Lecture 6).")
        X, X_aug, y, W_true = make_linear_data(n=80, d=2, noise=0.5)
        lam = 0.5
        W_sub, hist = subgradient_lasso(X_aug, y, lam=lam, alpha=0.1, max_iters=100, verbose=True, diminishing=True)
        print("Final LASSO weights:", W_sub)
        return

    if choice == '6':
        print("EWMA demo (smoothing a noisy loss curve):")
        xs = np.linspace(0,10,100)
        noisy = np.sin(xs) + 0.4*np.random.randn(len(xs))
        sm = ewma_sequence(noisy, beta=0.9)
        for i in range(0,100,10):
            print(f" t={i} raw={noisy[i]:.3f} ewma={sm[i]:.3f}")
        return

    if choice == '7':
        print("Lagrange multiplier example: maximize f(x,y)=x*y subject to x^2/8 + y^2/2 - 1 = 0  (from lecture).")
        # We'll minimize negative of f to turn into minimization
        def f(x):
            return -(x[0]*x[1])
        def grad_f(x):
            return np.array([ -x[1], -x[0] ])
        def h(x):
            return x[0]**2/8 + x[1]**2/2 - 1
        def grad_h(x):
            return np.array([ x[0]/4.0, x[1] ])
        x0 = np.array([1.0, 0.5])
        xopt, lam = lagrange_equality_solver(None, grad_f, h, grad_h, x0, lam0=0.0, verbose=True)
        print("Solution x:", xopt, " lambda:", lam)
        return

    if choice == '8':
        print("KKT demo: simple 2D minimization with g(x)<=0 constraints.")
        # Example: minimize f(x) = x1^2 + x2^2 subject to g1(x) = x1 + x2 - 1 <= 0
        def f(x): return x[0]**2 + x[1]**2
        def f_grad(x): return np.array([2*x[0], 2*x[1]])
        def g1(x): return x[0] + x[1] - 1
        def g1_grad(x): return np.array([1.0, 1.0])
        x = np.array([0.2, 0.2])
        # Solve unconstrained -> (0,0) but infeasible for g1? here g1(0,0) = -1 <=0 feasible
        # compute KKT multipliers by solving stationarity grad + lambda*g' = 0 with lambda>=0
        # For simplicity check KKT at candidate x=(0,0)
        lambdas = [0.0]
        mus = []
        station, primal_ok, dual_ok, cs_ok = kkt_check(f_grad, [g1_grad], [], np.array([0.0,0.0]), lambdas, mus)
        print("KKT check at x=(0,0): stationarity, primal_ok, dual_ok, complementary_slackness:", station, primal_ok, dual_ok, cs_ok)
        return

    print("Invalid option")

# -------------------------------
# Run interactive loop (only if script is run directly)
# -------------------------------
def run_interactive_menu():
    while True:
        main_menu()
        choice = input("Enter choice number: ").strip()
        if choice == '0' or choice.lower() in ['q', 'quit', 'exit']:
            print("Exiting. Good luck with your mini-project!")
            break
        try:
            run_option(choice)
        except Exception as e:
            print("Error while running option:", e)
        input("\nPress Enter to return to menu...")

if __name__ == '__main__':
    run_interactive_menu()
