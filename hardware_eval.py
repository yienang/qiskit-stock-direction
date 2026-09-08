# NOTE: running this script submits a real job to IBM Quantum hardware
# and consumes minutes from your monthly Open Plan budget. Not free to re-run casually.

import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_aer.primitives import SamplerV2 as AerSampler

from data_pull import read_csv_file
from features import build_feature_set
from preprocess import chronological_split, scale_for_quantum
from quantum_model import build_feature_map, build_ansatz

df = read_csv_file("AAPL", "5y", "1d")
result = build_feature_set(df)
train, test = chronological_split(result)
feature_cols = [col for col in result.columns if col != "label"]
X_train, X_test, scaler = scale_for_quantum(train, test, feature_cols)

loaded_weights = np.load("vqc_weights.npy")

feature_map = build_feature_map(6)
ansatz = build_ansatz(6)
combined = feature_map.compose(ansatz)

bound_circuits = []
for i in range(10):
    test_row = X_test[i]
    combined_values = np.concatenate([test_row, loaded_weights])
    circuit = combined.assign_parameters(combined_values)
    circuit.measure_all()
    bound_circuits.append(circuit)

print(f"Built {len(bound_circuits)} bound circuits")

service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=6)
print(f"Using backend: {backend.name}")

pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
transpiled_circuits = pm.run(bound_circuits)
print(f"Transpiled {len(transpiled_circuits)} circuits")

sampler = Sampler(mode=backend)
job = sampler.run(transpiled_circuits, shots=256)
print(f"Job ID: {job.job_id()}")
print("Waiting for job to complete...")
result = job.result()
print(result)

hw_counts = result[0].data.meas.get_counts()
print("\nReal hardware counts (first test row):")
print(hw_counts)

aer_sampler = AerSampler()
sim_job = aer_sampler.run([transpiled_circuits[0]], shots=128)
sim_result = sim_job.result()
sim_counts = sim_result[0].data.meas.get_counts()
print("\nSimulator counts (same circuit):")
print(sim_counts)