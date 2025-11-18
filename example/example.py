#!/usr/bin/env python3
"""
Proper minimal example using library functions correctly.
"""

import numpy as np
import os
from qtom.cli.generate_data import generate_povms, generate_training_data
from qtom.models.training import train_model
from qtom.core.random_states import random_HS_state
from qtom.evaluation.metrics import hilbert_schmidt_distance

# Generate POVM and data
generate_povms([2], num_states=1000)
generate_training_data([2], num_states=1000)

# Train the model
model, history = train_model(2, 'data/quantum_training_data_d2.npz', 'models/demo_model.h5')

# Test
from qtom.core.data_generation import get_probabilities, sample_measurements
from qtom.core.state_representation import cholesky_to_density
import keras

#model = keras.models.load_model('models/demo_model.h5', compile=False)
povm = np.load('data/SRMpom2.npz')['povm']

test_rho = random_HS_state(2)
probs = sample_measurements(get_probabilities(test_rho, povm), 1000) / 1000
pred = cholesky_to_density(model.predict(probs.reshape(1,-1), verbose=0)[0], 2)

print(f"Reconstruction error: {hilbert_schmidt_distance(test_rho, pred):.4f}")
print("True state:\n", np.round(test_rho, 3))
print("Predicted state:\n", np.round(pred, 3))
