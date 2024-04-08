import tkinter as tk
from tkinter import ttk
import numpy as np

from bandits import StationaryBandit
from agents import EpsilonGreedyAgent, GreedyAgent, OptimisticGreedyAgent, RandomAgent, UpperConfidenceBoundAgent

class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Agent Comparison")
        self.geometry("2880x1620")  # Set window size to 2880 x 1620

        # Variables
        self.k = tk.IntVar(value=1)
        self.min_val = tk.IntVar(value=0)
        self.max_val = tk.IntVar(value=1000)
        self.n = tk.IntVar(value=10)
        self.variance = tk.IntVar(value=1)
        self.epsilon = tk.DoubleVar(value=0.1)
        self.c = tk.DoubleVar(value = 0.1)

        # UI Elements
        self.k_frame, self.k_slider, self.k_entry = self.create_slider_with_entry("k", 1, 25, self.k)
        self.min_frame, self.min_slider, self.min_entry = self.create_slider_with_entry("min", 0, 999, self.min_val)
        self.max_frame, self.max_slider, self.max_entry = self.create_slider_with_entry("max", 1, 1000, self.max_val)
        self.n_frame, self.n_slider, self.n_entry = self.create_slider_with_entry("n", 10, 5000, self.n)
        self.variance_frame, self.variance_slider, self.variance_entry = self.create_slider_with_entry("variance", 1, 100, self.variance)
        self.epsilon_frame, self.epsilon_slider, self.epsilon_entry = self.create_slider_with_entry("epsilon", 0.01, 1.0, self.epsilon, is_int=False)
        self.c_frame, self.c_slider, self.c_entry = self.create_slider_with_entry("c", 0.01, 1.0, self.c, is_int=False)

        self.output_text = tk.Text(self, height=20, width=100)  # Larger text box
        self.output_text.pack(pady=10)

        # Bindings for slider movements
        self.k_slider.bind("<ButtonRelease-1>", self.generate_output)
        self.min_slider.bind("<ButtonRelease-1>", self.generate_output)
        self.max_slider.bind("<ButtonRelease-1>", self.generate_output)
        self.n_slider.bind("<ButtonRelease-1>", self.generate_output)
        self.variance_slider.bind("<ButtonRelease-1>", self.generate_output)
        self.epsilon_slider.bind("<ButtonRelease-1>", self.generate_output)
        self.c_slider.bind("<ButtonRelease-1>", self.generate_output)

    def create_slider_with_entry(self, label, min_val, max_val, var, is_int=True):
        frame = ttk.Frame(self)
        frame.pack(fill="x", pady=5)
        ttk.Label(frame, text=label).pack(side="left", padx=5)

        slider_frame = ttk.Frame(frame)
        slider_frame.pack(side="left", padx=5)
        if is_int:
            slider = ttk.Scale(slider_frame, from_=min_val, to=max_val, variable=var, orient="horizontal", command=lambda value, var=var: self.update_var(value, var))
        else:
            slider = ttk.Scale(slider_frame, from_=min_val, to=max_val, variable=var, orient="horizontal", command=lambda value, var=var: self.update_float_var(value, var))
        slider.pack(side="left")
        
        entry = ttk.Entry(frame, textvariable=var, validate="key", validatecommand=(self.register(lambda s: self.validate_entry(s, min_val, max_val)), '%P'))
        entry.pack(side="left", padx=5)

        return frame, slider, entry

    def update_var(self, value, var):
        try:
            var.set(int(float(value)))
        except ValueError:
            pass

    def update_float_var(self, value, var):
        try:
            # Check if the value has more than 2 decimal places
            if len(value.split('.')[1]) > 2:
                # Round to 2 decimal places
                rounded_value = "{:.2f}".format(float(value))
                var.set(float(rounded_value))
            else:
                var.set(float(value))
        except (ValueError, IndexError):
            pass

    def validate_entry(self, value, min_val, max_val):
        try:
            return min_val <= int(value) <= max_val
        except ValueError:
            return False

    def update_max_min(self, event=None):
        min_value = self.min_val.get()
        max_value = self.max_val.get()
        if min_value > max_value:
            self.max_val.set(min_value)

    def update_min_max(self, event=None):
        min_value = self.min_val.get()
        max_value = self.max_val.get()
        if min_value > max_value:
            self.max_val.set(min_value)

    def generate_output(self, event=None):
        min_value = self.min_val.get()
        max_value = self.max_val.get()
        if min_value > max_value:
            self.max_val.set(min_value)
            return

        k_value = self.k.get()
        n_value = self.n.get()
        variance_value = self.variance.get()
        epsilon_value = self.epsilon.get()
        c_value = self.c.get()

        self.output_text.delete(1.0, tk.END)  # Clear previous output
        self.output_text.insert(tk.END, "LOADING . . .")

        curr_bandit = StationaryBandit(k_value, min_value, max_value, variance_value)

        greedy = GreedyAgent(curr_bandit)
        opt_greedy = OptimisticGreedyAgent(curr_bandit, max_value*2)
        eps_greedy = EpsilonGreedyAgent(curr_bandit, epsilon_value)
        random = RandomAgent(curr_bandit)
        ucb = UpperConfidenceBoundAgent(curr_bandit)

        greedy_agent = greedy.runSequence(n_value)
        optimistic_greedy_agent = opt_greedy.runSequence(n_value)
        epsilon_greedy_agent = eps_greedy.runSequence(n_value)
        random_agent = random.runSequence(n_value)
        ucb_agent = ucb.runSequence(n_value, c_value)      

        output_text = f"Greedy Agent:\n {greedy_agent}\n\n" \
                     f"Optimistic Greedy Agent:\n {optimistic_greedy_agent}\n\n" \
                     f"Epsilon Greedy Agent:\n {epsilon_greedy_agent}\n\n" \
                     f"Upper Confidence Bound Agent:\n {ucb_agent}\n\n" \
                     f"Random Agent:\n {random_agent}"

        self.output_text.delete(1.0, tk.END)  # Clear previous output
        self.output_text.insert(tk.END, output_text)

if __name__ == "__main__":
    app = Application()
    app.mainloop()
