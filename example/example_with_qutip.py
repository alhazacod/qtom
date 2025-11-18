#!/usr/bin/env python3

import numpy as np
import qutip as qt
from qtom.cli.generate_data import generate_povms, generate_training_data
from qtom.models.training import train_model
from qtom.core.povm import load_fixed_povm
from qtom.core.data_generation import get_probabilities, sample_measurements
from qtom.core.state_representation import cholesky_to_density

# Train the model
print("Training model...")
generate_povms([2], num_states=20000)
generate_training_data([2], num_states=20000)
model, _ = train_model(2, 'data/quantum_training_data_d2.npz', 'models/demo_model.h5', epochs=10000)

# Test with QuTiP states
povm = load_fixed_povm(2)

# Create QuTiP states and reconstruct
states = {
    "|0⟩": qt.basis(2, 0),
    "|1⟩": qt.basis(2, 1),
    "|+⟩": (qt.basis(2,0) + qt.basis(2,1)).unit(),
}

print("\nReconstruction results:")
for name, state in states.items():
    rho_true = (state * state.dag()).full()
    probs = get_probabilities(rho_true, povm)
    counts = sample_measurements(probs, 1000)
    measured_probs = counts / 1000
    
    pred = cholesky_to_density(model.predict(measured_probs.reshape(1,-1), verbose=0)[0], 2)
    
    from qtom.evaluation.metrics import hilbert_schmidt_distance
    error = hilbert_schmidt_distance(rho_true, pred)
    
    print(f"{name}: error = {error:.4f}")
