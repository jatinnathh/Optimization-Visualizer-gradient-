import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import time
from src import (
    make_linear_data, make_logistic_data,
    mse_loss, logistic_loss, lasso_loss,
    gradient_descent, adagrad, newton_method, bfgs, subgradient_lasso,
    gd_exact_linesearch_quadratic, ewma_sequence, lagrange_equality_solver, kkt_check
)
st.set_page_config(layout="wide", page_title="Optimization Visualizer")

# Custom CSS to fix sidebar width
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            min-width: 450;
            max-width: 450;
        }
    </style>
""", unsafe_allow_html=True)

# Dynamic title based on selected algorithm and problem
st.title("Gradient Descent Optimizer Visualizer")

# Sidebar controls
st.sidebar.header("Configuration")

# Algorithm selection
algo = st.sidebar.selectbox(
    "Select Algorithm:",
    ["Closed-Form (Normal Equation)", "Gradient Descent (Constant)", "Gradient Descent (Diminishing)", 
     "Gradient Descent (Backtracking)", "GD (Exact Line Search)", "Adagrad", "Newton's Method", 
     "BFGS", "Subgradient (Lasso)", "EWMA Smoothing", "Lagrange Multiplier", "KKT Conditions"]
)

# Problem type
problem = st.sidebar.selectbox(
    "Select Problem Type:",
    ["Linear Regression (MSE)", "Logistic Regression"]
)

# Display dynamic subtitle
st.markdown(f"**Algorithm:** {algo} | **Problem:** {problem}")

# Initialize session state
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_iteration' not in st.session_state:
    st.session_state.current_iteration = 0

# Data size
data_size = st.sidebar.slider(
    "Data Size (n):",
    min_value=50,
    max_value=500,
    value=150,
    step=50
)

# Dimension
dimension = st.sidebar.slider(
    "Feature Dimension (d):",
    min_value=2,
    max_value=10,
    value=3,
    step=1
)

# Max epochs
max_epochs = st.sidebar.slider(
    "Max Epochs:",
    min_value=10,
    max_value=500,
    value=100,
    step=10
)

# Learning rate (for methods that use it)
if algo in ["Gradient Descent (Constant)", "Gradient Descent (Diminishing)", "Gradient Descent (Backtracking)", "Adagrad"]:
    learning_rate = st.sidebar.slider(
        "Learning Rate (α):",
        min_value=0.001,
        max_value=1.0,
        value=0.1,
        step=0.01
    )
else:
    learning_rate = 0.1

# Lasso lambda parameter
if algo == "Subgradient (Lasso)":
    lasso_lambda = st.sidebar.slider(
        "Lasso λ:",
        min_value=0.01,
        max_value=1.0,
        value=0.1,
        step=0.01
    )
else:
    lasso_lambda = 0.1

# Generate data
np.random.seed(42)

if problem == "Linear Regression (MSE)":
    X, X_aug, y, W_true = make_linear_data(n=data_size, d=dimension, noise=0.5)
    loss_fn = mse_loss
else:
    X, X_aug, y, W_true = make_logistic_data(n=data_size, d=dimension)
    loss_fn = logistic_loss

W0 = np.zeros(X_aug.shape[1])

# Create placeholders for real-time updates
st.sidebar.markdown("---")
col_start, col_stop = st.sidebar.columns(2)

with col_start:
    if st.button("Start Optimization", key="start_btn"):
        st.session_state.is_running = True
        st.session_state.history = []
        st.session_state.current_iteration = 0
        st.rerun()

with col_stop:
    if st.button("Stop", key="stop_btn"):
        st.session_state.is_running = False
        st.rerun()

# Main content area with placeholders
progress_placeholder = st.empty()
chart_col1, chart_col2 = st.columns(2)
loss_chart = chart_col1.empty()
grad_chart = chart_col2.empty()

stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
metric1 = stats_col1.empty()
metric2 = stats_col2.empty()
metric3 = stats_col3.empty()
metric4 = stats_col4.empty()

traj_col = st.empty()
table_col = st.empty()

# Real-time algorithm execution
if st.session_state.is_running:
    try:
        # Create a custom runner for real-time updates
        history = []
        W_current = W0.copy()
        
        if algo == "Closed-Form (Normal Equation)":
            if problem == "Linear Regression (MSE)":
                W_final = np.linalg.pinv(X_aug.T @ X_aug) @ (X_aug.T @ y)
                loss_final, grad_final, _ = mse_loss(W_final, X_aug, y)
                loss_init, grad_init, _ = mse_loss(W0, X_aug, y)
                history = [
                    (0, loss_init, grad_init, W0.copy()),
                    (1, loss_final, grad_final, W_final.copy())
                ]
                st.session_state.history = history
                st.session_state.is_running = False
            else:
                st.error("Closed-form solution only works for Linear Regression (MSE)")
        
        elif algo == "GD (Exact Line Search)":
            if problem == "Linear Regression (MSE)":
                W_current, history = gd_exact_linesearch_quadratic(
                    W0, X_aug, y, max_iters=max_epochs, verbose=False
                )
                st.session_state.history = history
                st.session_state.is_running = False
            else:
                st.error("Exact line search only works for Linear Regression (MSE)")
        
        elif algo == "Gradient Descent (Constant)":
            # Real-time GD implementation
            history = []
            W = W0.copy()
            for k in range(max_epochs):
                loss, grad, H = loss_fn(W, X_aug, y)
                gnorm = np.linalg.norm(grad)
                history.append((k, loss, gnorm, W.copy()))
                st.session_state.history = history
                st.session_state.current_iteration = k + 1
                
                # Update visualizations
                iterations = [h[0] for h in history]
                losses = [h[1] for h in history]
                grad_norms = [h[2] for h in history]
                
                with progress_placeholder.container():
                    st.progress(min((k + 1) / max_epochs, 1.0), f"Epoch {k+1}/{max_epochs}")
                
                with loss_chart.container():
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(iterations, losses, linewidth=2, color='#1f77b4', marker='o', markersize=4)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.set_title(f"Loss Convergence - {algo}")
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log')
                    st.pyplot(fig, use_container_width=True)
                
                with grad_chart.container():
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(iterations, grad_norms, linewidth=2, color='#ff7f0e', marker='s', markersize=4)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("||Gradient||")
                    ax.set_title(f"Gradient Norm - {algo}")
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log')
                    st.pyplot(fig, use_container_width=True)
                
                with metric1.container():
                    st.metric("Initial Loss", f"{losses[0]:.6f}")
                with metric2.container():
                    st.metric("Current Loss", f"{loss:.6f}")
                with metric3.container():
                    st.metric("Loss Reduction", f"{((losses[0] - loss) / losses[0] * 100):.2f}%")
                with metric4.container():
                    st.metric("Gradient Norm", f"{gnorm:.6e}")
                
                # Check for convergence
                if gnorm < 1e-6:
                    progress_placeholder.info("Converged!")
                    break
                
                W = W - learning_rate * grad
            
            W_current = W
            st.session_state.is_running = False
        
        elif algo == "Gradient Descent (Diminishing)":
            history = []
            W = W0.copy()
            for k in range(max_epochs):
                loss, grad, H = loss_fn(W, X_aug, y)
                gnorm = np.linalg.norm(grad)
                history.append((k, loss, gnorm, W.copy()))
                st.session_state.history = history
                st.session_state.current_iteration = k + 1
                
                iterations = [h[0] for h in history]
                losses = [h[1] for h in history]
                grad_norms = [h[2] for h in history]
                
                with progress_placeholder.container():
                    st.progress(min((k + 1) / max_epochs, 1.0), f"Epoch {k+1}/{max_epochs}")
                
                with loss_chart.container():
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(iterations, losses, linewidth=2, color='#1f77b4', marker='o', markersize=4)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.set_title(f"Loss Convergence - {algo}")
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log')
                    st.pyplot(fig, use_container_width=True)
                
                with grad_chart.container():
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(iterations, grad_norms, linewidth=2, color='#ff7f0e', marker='s', markersize=4)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("||Gradient||")
                    ax.set_title(f"Gradient Norm - {algo}")
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log')
                    st.pyplot(fig, use_container_width=True)
                
                with metric1.container():
                    st.metric("Initial Loss", f"{losses[0]:.6f}")
                with metric2.container():
                    st.metric("Current Loss", f"{loss:.6f}")
                with metric3.container():
                    st.metric("Loss Reduction", f"{((losses[0] - loss) / losses[0] * 100):.2f}%")
                with metric4.container():
                    st.metric("Gradient Norm", f"{gnorm:.6e}")
                
                if gnorm < 1e-6:
                    progress_placeholder.info("Converged!")
                    break
                
                alpha = learning_rate / (k + 1)
                W = W - alpha * grad
            
            W_current = W
            st.session_state.is_running = False
        
        elif algo == "Gradient Descent (Backtracking)":
            history = []
            W = W0.copy()
            for k in range(max_epochs):
                loss, grad, H = loss_fn(W, X_aug, y)
                gnorm = np.linalg.norm(grad)
                history.append((k, loss, gnorm, W.copy()))
                st.session_state.history = history
                st.session_state.current_iteration = k + 1
                
                iterations = [h[0] for h in history]
                losses = [h[1] for h in history]
                grad_norms = [h[2] for h in history]
                
                with progress_placeholder.container():
                    st.progress(min((k + 1) / max_epochs, 1.0), f"Epoch {k+1}/{max_epochs}")
                
                with loss_chart.container():
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(iterations, losses, linewidth=2, color='#1f77b4', marker='o', markersize=4)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.set_title(f"Loss Convergence - {algo}")
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log')
                    st.pyplot(fig, use_container_width=True)
                
                with grad_chart.container():
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(iterations, grad_norms, linewidth=2, color='#ff7f0e', marker='s', markersize=4)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("||Gradient||")
                    ax.set_title(f"Gradient Norm - {algo}")
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log')
                    st.pyplot(fig, use_container_width=True)
                
                with metric1.container():
                    st.metric("Initial Loss", f"{losses[0]:.6f}")
                with metric2.container():
                    st.metric("Current Loss", f"{loss:.6f}")
                with metric3.container():
                    st.metric("Loss Reduction", f"{((losses[0] - loss) / losses[0] * 100):.2f}%")
                with metric4.container():
                    st.metric("Gradient Norm", f"{gnorm:.6e}")
                
                if gnorm < 1e-6:
                    progress_placeholder.info("Converged!")
                    break
                
                # Backtracking line search
                alpha = learning_rate
                c = 1e-4
                rho = 0.5
                fx = loss
                for _ in range(20):
                    W_new = W - alpha * grad
                    fnew, _, _ = loss_fn(W_new, X_aug, y)
                    if fnew <= fx - c * alpha * (gnorm**2) or alpha < 1e-12:
                        break
                    alpha = rho * alpha
                
                W = W - alpha * grad
            
            W_current = W
            st.session_state.is_running = False
        
        elif algo == "Adagrad":
            history = []
            W = W0.copy()
            G = np.zeros_like(W)
            for k in range(max_epochs):
                loss, grad, H = loss_fn(W, X_aug, y)
                gnorm = np.linalg.norm(grad)
                history.append((k, loss, gnorm, W.copy()))
                st.session_state.history = history
                st.session_state.current_iteration = k + 1
                
                iterations = [h[0] for h in history]
                losses = [h[1] for h in history]
                grad_norms = [h[2] for h in history]
                
                with progress_placeholder.container():
                    st.progress(min((k + 1) / max_epochs, 1.0), f"Epoch {k+1}/{max_epochs}")
                
                with loss_chart.container():
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(iterations, losses, linewidth=2, color='#1f77b4', marker='o', markersize=4)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.set_title(f"Loss Convergence - {algo}")
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log')
                    st.pyplot(fig, use_container_width=True)
                
                with grad_chart.container():
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(iterations, grad_norms, linewidth=2, color='#ff7f0e', marker='s', markersize=4)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("||Gradient||")
                    ax.set_title(f"Gradient Norm - {algo}")
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log')
                    st.pyplot(fig, use_container_width=True)
                
                with metric1.container():
                    st.metric("Initial Loss", f"{losses[0]:.6f}")
                with metric2.container():
                    st.metric("Current Loss", f"{loss:.6f}")
                with metric3.container():
                    st.metric("Loss Reduction", f"{((losses[0] - loss) / losses[0] * 100):.2f}%")
                with metric4.container():
                    st.metric("Gradient Norm", f"{gnorm:.6e}")
                
                if gnorm < 1e-8:
                    progress_placeholder.info("Converged!")
                    break
                
                G += grad**2
                W = W - (learning_rate / (np.sqrt(G) + 1e-8)) * grad
            
            W_current = W
            st.session_state.is_running = False
        
        elif algo == "Newton's Method":
            history = []
            W = W0.copy()
            for k in range(max_epochs):
                loss, grad, H = loss_fn(W, X_aug, y)
                gnorm = np.linalg.norm(grad)
                history.append((k, loss, gnorm, W.copy()))
                st.session_state.history = history
                st.session_state.current_iteration = k + 1
                
                iterations = [h[0] for h in history]
                losses = [h[1] for h in history]
                grad_norms = [h[2] for h in history]
                
                with progress_placeholder.container():
                    st.progress(min((k + 1) / max_epochs, 1.0), f"Epoch {k+1}/{max_epochs}")
                
                with loss_chart.container():
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(iterations, losses, linewidth=2, color='#1f77b4', marker='o', markersize=4)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.set_title(f"Loss Convergence - {algo}")
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log')
                    st.pyplot(fig, use_container_width=True)
                
                with grad_chart.container():
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(iterations, grad_norms, linewidth=2, color='#ff7f0e', marker='s', markersize=4)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("||Gradient||")
                    ax.set_title(f"Gradient Norm - {algo}")
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log')
                    st.pyplot(fig, use_container_width=True)
                
                with metric1.container():
                    st.metric("Initial Loss", f"{losses[0]:.6f}")
                with metric2.container():
                    st.metric("Current Loss", f"{loss:.6f}")
                with metric3.container():
                    st.metric("Loss Reduction", f"{((losses[0] - loss) / losses[0] * 100):.2f}%")
                with metric4.container():
                    st.metric("Gradient Norm", f"{gnorm:.6e}")
                
                if gnorm < 1e-10:
                    progress_placeholder.info("Converged!")
                    break
                
                try:
                    from numpy.linalg import solve
                    delta = solve(H + 1e-6*np.eye(len(W)), grad)
                except:
                    delta = np.linalg.pinv(H + 1e-6*np.eye(len(W))) @ grad
                
                W = W - delta
            
            W_current = W
            st.session_state.is_running = False
        
        elif algo == "BFGS":
            history = []
            W = W0.copy()
            n = len(W)
            B_inv = np.eye(n)
            loss, grad, H = loss_fn(W, X_aug, y)
            
            for k in range(max_epochs):
                loss, grad, H = loss_fn(W, X_aug, y)
                gnorm = np.linalg.norm(grad)
                history.append((k, loss, gnorm, W.copy()))
                st.session_state.history = history
                st.session_state.current_iteration = k + 1
                
                iterations = [h[0] for h in history]
                losses = [h[1] for h in history]
                grad_norms = [h[2] for h in history]
                
                with progress_placeholder.container():
                    st.progress(min((k + 1) / max_epochs, 1.0), f"Epoch {k+1}/{max_epochs}")
                
                with loss_chart.container():
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(iterations, losses, linewidth=2, color='#1f77b4', marker='o', markersize=4)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.set_title(f"Loss Convergence - {algo}")
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log')
                    st.pyplot(fig, use_container_width=True)
                
                with grad_chart.container():
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(iterations, grad_norms, linewidth=2, color='#ff7f0e', marker='s', markersize=4)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("||Gradient||")
                    ax.set_title(f"Gradient Norm - {algo}")
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log')
                    st.pyplot(fig, use_container_width=True)
                
                with metric1.container():
                    st.metric("Initial Loss", f"{losses[0]:.6f}")
                with metric2.container():
                    st.metric("Current Loss", f"{loss:.6f}")
                with metric3.container():
                    st.metric("Loss Reduction", f"{((losses[0] - loss) / losses[0] * 100):.2f}%")
                with metric4.container():
                    st.metric("Gradient Norm", f"{gnorm:.6e}")
                
                if gnorm < 1e-8:
                    progress_placeholder.info("Converged!")
                    break
                
                p = - B_inv @ grad
                s = learning_rate * p
                x_new = W + s
                loss_new, grad_new, H_new = loss_fn(x_new, X_aug, y)
                y_vec = grad_new - grad
                rho = 1.0 / (y_vec @ s + 1e-12)
                
                I = np.eye(n)
                Bs = (I - rho * np.outer(s, y_vec))
                B_inv = Bs @ B_inv @ Bs.T + rho * np.outer(s, s)
                
                W = x_new
                grad = grad_new
            
            W_current = W
            st.session_state.is_running = False
        
        elif algo == "Subgradient (Lasso)":
            history = []
            W = np.zeros(X_aug.shape[1])
            
            for k in range(max_epochs):
                loss, grad_smooth, H = lasso_loss(W, X_aug, y, lasso_lambda)
                g_l1 = np.sign(W)
                g_l1[W==0] = 0.0
                full_sub = grad_smooth + lasso_lambda * np.sign(W)
                full_sub[W==0] = grad_smooth[W==0]
                gnorm = np.linalg.norm(full_sub)
                history.append((k, loss, gnorm, W.copy()))
                st.session_state.history = history
                st.session_state.current_iteration = k + 1
                
                iterations = [h[0] for h in history]
                losses = [h[1] for h in history]
                grad_norms = [h[2] for h in history]
                
                with progress_placeholder.container():
                    st.progress(min((k + 1) / max_epochs, 1.0), f"Epoch {k+1}/{max_epochs}")
                
                with loss_chart.container():
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(iterations, losses, linewidth=2, color='#1f77b4', marker='o', markersize=4)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.set_title(f"Loss Convergence - {algo}")
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log')
                    st.pyplot(fig, use_container_width=True)
                
                with grad_chart.container():
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(iterations, grad_norms, linewidth=2, color='#ff7f0e', marker='s', markersize=4)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("||Gradient||")
                    ax.set_title(f"Gradient Norm - {algo}")
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log')
                    st.pyplot(fig, use_container_width=True)
                
                with metric1.container():
                    st.metric("Initial Loss", f"{losses[0]:.6f}")
                with metric2.container():
                    st.metric("Current Loss", f"{loss:.6f}")
                with metric3.container():
                    st.metric("Loss Reduction", f"{((losses[0] - loss) / losses[0] * 100):.2f}%")
                with metric4.container():
                    st.metric("Gradient Norm", f"{gnorm:.6e}")
                
                if gnorm < 1e-8:
                    progress_placeholder.info("Converged!")
                    break
                
                W = W - learning_rate * full_sub
            
            W_current = W
            st.session_state.is_running = False
        
        elif algo == "EWMA Smoothing":
            xs = np.linspace(0, 10, max_epochs)
            noisy_loss = np.sin(xs) + 0.4 * np.random.randn(len(xs))
            smoothed = ewma_sequence(noisy_loss, beta=0.9)
            
            history = []
            for i in range(len(xs)):
                history.append((i, noisy_loss[i], smoothed[i], np.array([noisy_loss[i]])))
                st.session_state.history = history
                st.session_state.current_iteration = i + 1
                
                with progress_placeholder.container():
                    st.progress(min((i + 1) / len(xs), 1.0), f"Sample {i+1}/{len(xs)}")
                
                with loss_chart.container():
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(range(i+1), noisy_loss[:i+1], linewidth=2, color='#d62728', marker='o', markersize=4, label='Noisy Loss')
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Loss")
                    ax.set_title("Original Noisy Loss Signal")
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    st.pyplot(fig, use_container_width=True)
                
                with grad_chart.container():
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(range(i+1), smoothed[:i+1], linewidth=2, color='#2ca02c', marker='s', markersize=4, label='EWMA Smoothed')
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Smoothed Value")
                    ax.set_title("EWMA Smoothed Signal (β=0.9)")
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    st.pyplot(fig, use_container_width=True)
            
            st.session_state.is_running = False
        
        else:
            st.info("Select an algorithm and click 'Start Optimization' to begin!")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.session_state.is_running = False

else:
    # Display stored results if available
    if st.session_state.history:
        history = st.session_state.history
        iterations = [h[0] for h in history]
        losses = [h[1] for h in history]
        grad_norms = [h[2] for h in history]
        
        with loss_chart.container():
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(iterations, losses, linewidth=2, color='#1f77b4', marker='o', markersize=4)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"Loss Convergence - {algo}")
            ax.grid(True, alpha=0.3)
            if max(losses) > 0:
                ax.set_yscale('log')
            st.pyplot(fig, use_container_width=True)
        
        with grad_chart.container():
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(iterations, grad_norms, linewidth=2, color='#ff7f0e', marker='s', markersize=4)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("||Gradient||")
            ax.set_title(f"Gradient Norm - {algo}")
            ax.grid(True, alpha=0.3)
            if max(grad_norms) > 0:
                ax.set_yscale('log')
            st.pyplot(fig, use_container_width=True)
    else:
        with loss_chart.container():
            st.info("Click 'Start Optimization' to begin training")

# Footer
st.markdown("---")
st.markdown("""
**About this visualizer:**
- Visualizes different gradient descent variants and advanced optimization algorithms
- Shows real-time convergence visualization with live updating graphs
- Supports multiple problem types (linear regression, logistic regression)
- Interactive parameter tuning for algorithm configuration
- Click "Start Optimization" to begin training and watch gradients descend in real-time

**Algorithms Implemented:**
- **Closed-Form (Normal Equation)**: Direct solution for linear regression
- **Gradient Descent Variants**:
  - Constant step size
  - Diminishing step size (1/(k+1))
  - Backtracking line search
  - Exact line search (quadratic)
- **Adagrad**: Adaptive learning rate based on historical gradients
- **Newton's Method**: Second-order optimization using Hessian
- **BFGS**: Quasi-Newton method with inverse Hessian approximation
- **Subgradient (Lasso)**: For non-smooth L1-regularized regression
- **EWMA Smoothing**: Exponentially Weighted Moving Average for signal smoothing
- **Lagrange Multiplier**: Solves equality-constrained optimization
- **KKT Conditions**: Verifies optimality conditions for constrained problems
            

             Jatin Nath      Optimization visualizer
""")
