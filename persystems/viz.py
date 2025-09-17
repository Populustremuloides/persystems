"""
Plotting helpers for the ring-world demos.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_efe_trace(ts, G, risk, ambiguity, title="Expected Free Energy (1-step)"):
    plt.figure(figsize=(8, 4))
    plt.plot(ts, G, label="EFE (chosen)")
    plt.plot(ts, risk, label="Risk")
    plt.plot(ts, ambiguity, label="Ambiguity")
    plt.xlabel("time")
    plt.ylabel("nats")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cost_info(ts, exp_cost, info_gain):
    plt.figure(figsize=(8, 4))
    plt.plot(ts, exp_cost, label="Expected cost")
    plt.plot(ts, info_gain, label="Information gain")
    plt.xlabel("time")
    plt.ylabel("nats")
    plt.title("Cost − Information gain decomposition")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_entropy(ts, H_q):
    plt.figure(figsize=(8, 4))
    plt.plot(ts, H_q)
    plt.xlabel("time")
    plt.ylabel("entropy of belief")
    plt.title("Belief entropy over time")
    plt.tight_layout()
    plt.show()


def plot_final_posterior(qs):
    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(qs.size), qs)
    plt.xlabel("state")
    plt.ylabel("posterior probability")
    plt.title("Final posterior over states")
    plt.tight_layout()
    plt.show()

