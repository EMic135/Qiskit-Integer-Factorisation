print("Starting Program...")

import Defaults
from qiskit.circuit.library import QFTGate
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from math import gcd, ceil, log2
from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from Gates import AdditionalGates as QC

plt.rcParams["figure.figsize"] = (20,5)

print("Imports finished, taking using input...\n")

N=int(input("Enter the number to factorise"))
a=int(input("Enter a value for a"))
if gcd(a, N) > 1:
    print(f"Error: gcd({a},{N}) > 1")
    quit()

print("\nPreparing circuit...")

k=2*ceil(log2(N))
b_list = [pow(a,2**val,N) for val in range(k)]

work_bits = k//2
phase_bits = k

Phase_Register = QuantumRegister(phase_bits,"P")
Work_Register = QuantumRegister(work_bits,"W")
Carry = QuantumRegister(work_bits+1,"C")
Accumulator1 = QuantumRegister(work_bits,"A1")
Flag = QuantumRegister(1,"F")
Ancilla = QuantumRegister(1,"anc")
meas = ClassicalRegister(phase_bits,"meas")
qc = QC(Phase_Register, Work_Register, Carry, Accumulator1, Flag, Ancilla, meas)
qc.h(Phase_Register)
qc.x(Work_Register[0])

for index in range(len(b_list)):
    b = b_list[index]
    if b > 1:
        qc.const_mult_mod_self(Work_Register, b, N, Carry, Accumulator1, Flag, Phase_Register[index], Ancilla)

qc.compose(QFTGate(phase_bits).inverse(),Phase_Register,inplace=True)
qc.measure(Phase_Register, meas)

print("Circuit ready, loading simulator...")

sim = Defaults.Simulation()

print("Simulator ready, decomposing circuit...")

decomp = [qc]
gates = decomp[-1].count_ops()
for i in range(0,10):
    decomp.append((decomp[-1] if len(decomp) > 0 else qc).decompose())
    new_gates = decomp[-1].count_ops()
    if new_gates == gates:
        del(decomp[-1])
        break
    else:
        gates = new_gates
smallest = -1
for i, circ in enumerate(decomp):
    sim.update_isa_circuit(circ)
    gates = sim.isa_circuit.count_ops()
    if smallest == -1 or sum(gates.values()) < sum(decomp[smallest].count_ops().values()) :
        smallest = i

print("Circuit decomposed, loading circuit to simulator...")
sim.update_isa_circuit(decomp[smallest])
print("Circuit loaded, running simulator...")
while True:
    job_name = f"Shor({a},{N}) gates sample"
    result = sim.get_result(job_name)
    if result:
        if result == True:
            print("Waiting on result...")
        else:
            print("Simulation finished, finding factors...")
            break
    else:
        if result == False:
            print("Job was cancelled or errored")
            quit()
        else:
            #sim.update_isa_circuit(qc)
            sim.save_job(sim.sampler.run([sim.isa_circuit],shots=1024),job_name)
            print("Job Started...")
    sleep(1)


counts = result[0].data.meas.get_counts()
counts_keep = {}
threshold = np.max(list(counts.values())) / 5

for key, value in counts.items():
    if value > threshold:
        counts_keep[key] = value

found = False
attempt = 0
while not found:
    bitstring = list(counts_keep.keys())[attempt]
    attempt += 1
    decimal = int(bitstring, 2)
    if decimal != 0:
        phase = decimal / (2**phase_bits)
        frac = Fraction(phase).limit_denominator(N)
        r = frac.denominator
        if r % 2 == 0:
            x = pow(a, r // 2, N) - 1
            d = gcd(x, N)
            if d > 1:
                found = True
                print(f"*** Non-trivial factor found: {x}, GCD({x},{N}): {gcd(x,N)} ***")