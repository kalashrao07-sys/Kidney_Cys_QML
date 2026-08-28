import pennylane as qml
import numpy as np

N_QUBITS = 7
N_CYCLES = 3

dev = qml.device("default.qubit", wires=N_QUBITS)

def hea1(params, wires):
    n = len(wires)

    for i in range(n):
        qml.Hadamard(wires=wires[i])
        qml.RX(params[i, 0], wires=wires[i])
        qml.RY(params[i, 1], wires=wires[i])

    for i in range(n):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])

    for i in range(n):
        qml.Toffoli(
            wires=[wires[i], wires[(i + 1) % n], wires[(i + 2) % n]]
        )

def hea2(params, wires):
    n = len(wires)

    for i in range(n):
        qml.Hadamard(wires=wires[i])
        qml.RY(params[i, 0], wires=wires[i])
        qml.RZ(params[i, 1], wires=wires[i])

    for i in range(n):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])

    for i in range(n):
        qml.Toffoli(
            wires=[wires[i], wires[(i + 1) % n], wires[(i + 2) % n]]
        )

@qml.qnode(dev)
def circuit(x, weights1, weights2):

    qml.AmplitudeEmbedding(
        x,
        wires=range(N_QUBITS),
        normalize=True,
        pad_with=0.0
    )

    for l in range(N_CYCLES):
        hea1(weights1[l], wires=range(N_QUBITS))
        hea2(weights2[l], wires=range(N_QUBITS))

    return [qml.expval(qml.PauliZ(i)) for i in range(5)]


x = np.ones(128) / np.sqrt(128)

w1 = np.zeros((N_CYCLES, N_QUBITS, 2))
w2 = np.zeros((N_CYCLES, N_QUBITS, 2))

print(qml.specs(circuit)(x, w1, w2))