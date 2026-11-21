from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.circuit.library import UnitaryGate
import numpy as np
from math import gcd as gcd


class AdditionalGates(QuantumCircuit):
    def and_gate(self, a, b, r, ctrl=None, anc=None):
        if ctrl is None:
            if type(a) is Qubit:
                if type(b) is Qubit:
                    self.ccx(a, b, r)
                elif type(b) is bool or type(b) is int:
                    if b: self.cx(a, r)
                else:
                    raise TypeError('b must be a Qubit, bool or int')
            elif type(a) is bool or type(a) is int:
                if type(b) is Qubit:
                    if a: self.cx(b, r)
                elif type(b) is bool or type(b) is int:
                    if a and b: self.x(r)
                else:
                    raise TypeError('b must be a Qubit, bool or int')
            else:
                raise TypeError('a must be a Qubit, bool or int')
        else:
            if type(a) is Qubit:
                if type(b) is Qubit:
                    self.mcx([ctrl, a, b], r, anc)
                elif type(b) is bool or type(b) is int:
                    if b: self.ccx(ctrl, a, r)
                else:
                    raise TypeError('b must be a Qubit, bool or int')
            elif type(a) is bool or type(a) is int:
                if type(b) is Qubit:
                    if a: self.ccx(ctrl, b, r)
                elif type(b) is bool or type(b) is int:
                    if a and b: self.cx(ctrl, r)
                else:
                    raise TypeError('b must be a Qubit, bool or int')
            else:
                raise TypeError('a must be a Qubit, bool or int')

    def xor_gate(self, a, b, r, ctrl=None):
        if ctrl is None:
            if type(a) is Qubit:
                self.cx(a, r)
            elif type(a) is bool or type(a) is int:
                if a: self.x(r)
            else:
                raise TypeError('a must be a Qubit, bool or int')
            if type(b) is Qubit:
                self.cx(b, r)
            elif type(b) is bool or type(b) is int:
                if b: self.x(r)
            else:
                raise TypeError('b must be a Qubit, bool or int')
        else:
            if type(a) is Qubit:
                self.ccx(ctrl, a, r)
            elif type(a) is bool or type(a) is int:
                if a: self.cx(ctrl, r)
            else:
                raise TypeError('a must be a Qubit, bool or int')
            if type(b) is Qubit:
                self.ccx(ctrl, b, r)
            elif type(b) is bool or type(b) is int:
                if b: self.cx(ctrl, r)
            else:
                raise TypeError('b must be a Qubit, bool or int')

    def or_gate(self, a, b, r, ctrl=None):
        self.and_gate(a, b, r, ctrl)
        self.xor_gate(a, b, r, ctrl)

    def copy_gate(self, a, b, ctrl=None):
        if ctrl is None:
            for i in range(len(a)):
                if type(a[i]) is Qubit:
                    self.cx(a[i], b[i])
                elif type(a[i]) is bool or type(a[i]) is int:
                    if a[i]: self.x(b[i])
        else:
            for i in range(len(a)):
                if type(a[i]) is Qubit:
                    self.ccx(ctrl, a[i], b[i])
                elif type(a[i]) is bool or type(a[i]) is int:
                    if a[i]: self.cx(ctrl, b[i])

    def invert(self, a):
        if type(a) is QuantumRegister:
            self.x(a)
        else:
            a = [1 - val for val in a]
        return a

    def carry(self, a, b, c, c_out, ctrl=None, anc=None):
        self.and_gate(a, b, c_out, ctrl, anc)
        self.and_gate(a, c, c_out, ctrl, anc)
        self.and_gate(c, b, c_out, ctrl, anc)

    def add_self(self, a, b, c, ctrl=None, anc=None):
        """a=a+b"""
        if len(c) != len(a) + 1 or len(c) != len(b) + 1:
            raise ValueError('c must be 1 qubit larger than a and b')
        for i in range(len(a)):
            self.carry(a[i], b[i], c[i], c[i + 1], anc = anc)
        for i in range(len(a)-1, -1, -1):
            self.carry(a[i], b[i], c[i], c[i+1], anc = anc)
            self.xor_gate(b[i], c[i], a[i], ctrl)

    def sub_self(self, a, b, c, ctrl=None, anc=None):
        """a=a+b"""
        if len(c) != len(a) + 1 or len(c) != len(b) + 1:
            raise ValueError('c must be 1 qubit larger than a and b')
        b = self.invert(b)
        self.x(c[0])

        self.add_self(a, b, c, ctrl, anc)

        b = self.invert(b)
        self.x(c[0])

    def geq(self, a, b, c, f, ctrl=None, anc=None):
        """f=a>=b"""
        b = self.invert(b)
        self.x(c[0])

        for i in range(len(a)):
            self.carry(a[i], b[i], c[i], c[i + 1], anc=anc)
        if ctrl is None:
            self.cx(c[-1], f)
        else:
            self.ccx(ctrl, c[-1], f)
        for i in range(len(a) - 1, -1, -1):
            self.carry(a[i], b[i], c[i], c[i + 1], anc=anc)

        b = self.invert(b)
        self.x(c[0])

    def add_mod_self(self, a, b, N, c, f, ctrl=None, anc=None):
        self.add_self(a, b, c, ctrl, anc=anc)
        self.geq(a, N, c, f, ctrl, anc=anc)
        self.sub_self(a, N, c, f, anc=anc)
        self.sub_self(a, b, c, ctrl, anc=anc)
        self.geq(a, N, c, f, ctrl, anc=anc)
        self.add_self(a, b, c, ctrl, anc=anc)

    def const_mult_mod(self, a:QuantumRegister, b:int, N:int, c, r, f, ctrl=None, anc=None):
        """r=ab%N"""
        if ctrl is not None:
            if anc is None:
                raise ValueError('controlling mult mod requires 1 anc')
            control=anc[0]
            anc=anc[1:]
        n = len(a)
        N_bits = [(N >> j) & 1 for j in range(n)]
        for i in range(n):
            value = (b * (1 << i)) % N
            if value>0:
                if ctrl is not None:
                    self.ccx(ctrl, a[i], control)
                else:
                    control = a[i]

                b_bits = [(value >> j) & 1 for j in range(n)]
                self.add_mod_self(r, b_bits, N_bits, c, f, ctrl=control, anc=anc)

                if ctrl is not None:
                    self.ccx(ctrl, a[i], control)

    def const_mult_mod_self(self, a:QuantumRegister, b:int, N:int, c, acu1, f, ctrl=None, anc=None):
        """a=ab%N"""
        self.const_mult_mod(a, b, N, c, acu1, f, ctrl, anc)
        self.const_mult_mod(acu1, pow(b, -1, N), N, c, a, f, ctrl, anc)
        self.copy_gate(acu1, a, ctrl)
        self.copy_gate(a, acu1, ctrl)
        self.copy_gate(acu1, a, ctrl)

    def perm_matrix_multmod(self, a:QuantumRegister, b:int, N:int, ctrl=None):
        """a=ab%N"""
        n = len(a)
        dim = 2 ** n
        U = np.identity(dim, np.int32)
        for x in range(N):
            U[x, x] = 0
            U[(b * x % N), x] = 1
        G = UnitaryGate(U)
        G.name = "\\ket{x}\\rightarrow\\ket{x\\cdot" + str(b) + " mod " + str(N) + "}"
        if ctrl is None:
            self.append(G,a)
        else:
            self.append(G.control(),list(ctrl)+list(a))
        return G




if __name__ == "__main__" and True: #Change to True to run tests
    from qiskit import ClassicalRegister, transpile
    from qiskit_aer import AerSimulator
    import matplotlib.pyplot as plt
    sim = AerSimulator()

    def test_addself(n):
        for x in range(2 ** (n - 1)):
            for y in range(2 ** (n-1)):
                a = QuantumRegister(n, 'a')
                b = QuantumRegister(n, 'b')
                c = QuantumRegister(n + 1, 'c')
                a_out = ClassicalRegister(len(a), 'a_o')
                b_out = ClassicalRegister(len(b), 'b_o')
                c_out = ClassicalRegister(len(c), 'c_o')
                qc = AdditionalGates(a, b, c, a_out, b_out, c_out)

                for i in range(n):
                    if (x >> i) & 1:
                        qc.x(a[i])
                    if (y >> i) & 1:
                        qc.x(b[i])

                qc.add_self(a, b, c)

                qc.measure(a, a_out)
                qc.measure(b, b_out)
                qc.measure(c, c_out)

                # qc.draw(output='mpl')
                # plt.show()

                res = sim.run(transpile(qc, sim), shots=1).result()
                raw = list(res.get_counts())[0].split(" ")
                output = [int(number, 2) for number in raw]
                goal = x+y
                goal -= goal*(goal>15)
                a=output[2]
                b=output[1]
                c=output[0]
                print(f"ADD: {x}+{y}={goal}    a={a}:{a==goal}\n    b={b}:{b==y},c=0:{c==0}")

    def test_subself(n):
        for x in range(2 ** (n - 1)):
            for y in range(2 ** (n - 1)):
                a = QuantumRegister(n, 'a')
                b = QuantumRegister(n, 'b')
                c = QuantumRegister(n + 1, 'c')
                a_out = ClassicalRegister(len(a), 'a_o')
                b_out = ClassicalRegister(len(b), 'b_o')
                c_out = ClassicalRegister(len(c), 'c_o')
                qc = AdditionalGates(a, b, c, a_out, b_out, c_out)

                for i in range(n):
                    if (x >> i) & 1:
                        qc.x(a[i])
                    if (y >> i) & 1:
                        qc.x(b[i])

                qc.sub_self(a, b, c)

                qc.measure(a, a_out)
                qc.measure(b, b_out)
                qc.measure(c, c_out)

                # qc.draw(output='mpl')
                # plt.show()

                res = sim.run(transpile(qc, sim), shots=1).result()
                raw = list(res.get_counts())[0].split(" ")
                output = [int(number, 2) for number in raw]
                goal = x-y+(16*((x-y)<0))
                a=output[2]
                b=output[1]
                c=output[0]
                print(f"SUB: {x}-{y}={goal}    a={a}:{a==goal}\n    b={y}:{b==y},c=0:{c==0}")

    def test_geq(n):
        for x in range(2 ** (n - 1)):
            for y in range(2 ** (n - 1)):
                a = QuantumRegister(n, 'a')
                b = QuantumRegister(n, 'b')
                c = QuantumRegister(n + 1, 'c')
                f = QuantumRegister(1, 'r')
                a_out = ClassicalRegister(len(a), 'a_o')
                b_out = ClassicalRegister(len(b), 'b_o')
                c_out = ClassicalRegister(len(c), 'c_o')
                f_out = ClassicalRegister(len(f), 'r_o')
                qc = AdditionalGates(a, b, c, f, a_out, b_out, c_out, f_out)

                for i in range(n):
                    if (x >> i) & 1:
                        qc.x(a[i])
                    if (y >> i) & 1:
                        qc.x(b[i])

                qc.geq(a, b, c, f)

                qc.measure(a, a_out)
                qc.measure(b, b_out)
                qc.measure(c, c_out)
                qc.measure(f, f_out)

                # qc.draw(output='mpl')
                # plt.show()

                res = sim.run(transpile(qc, sim), shots=1).result()
                raw = list(res.get_counts())[0].split(" ")
                output = [int(number, 2) for number in raw]
                goal = int(x>=y)
                a=output[3]
                b=output[2]
                c=output[1]
                f=output[0]
                print(f"GEQ: {x}>={y}={goal}    f={f}:{f==goal}\n    a={a}:{a==x},b={b}:{b==y},c=0:{c==0}")

    def test_addmodself(n):
        N_val=2 ** (n - 1)
        for x in range(N_val//3,2 ** (n - 1)):
            for y in range(1,2 ** (n - 1)):
                a = QuantumRegister(n, 'a')
                b = QuantumRegister(n, 'b')
                N = QuantumRegister(n, 'N')
                c = QuantumRegister(n + 1, 'c')
                f = QuantumRegister(1, 'f')
                anc = QuantumRegister(1, 'anc')
                a_out = ClassicalRegister(len(a), 'a_o')
                b_out = ClassicalRegister(len(b), 'b_o')
                c_out = ClassicalRegister(len(c), 'c_o')
                f_out = ClassicalRegister(len(f), 'f_o')
                qc = AdditionalGates(a, b, N, c, f, anc, a_out, b_out, c_out, f_out)

                for i in range(n):
                    if (x >> i) & 1:
                        qc.x(a[i])
                    if (y >> i) & 1:
                        qc.x(b[i])
                    if (N_val >> i) & 1:
                        qc.x(N[i])

                qc.addmod_self(a, b, N, c, f, anc=anc)

                qc.measure(a, a_out)
                qc.measure(b, b_out)
                qc.measure(c, c_out)
                qc.measure(f, f_out)

                # qc.draw(output='mpl')
                # plt.show()

                res = sim.run(transpile(qc, sim), shots=1).result()
                raw = list(res.get_counts())[0].split(" ")
                output = [int(number, 2) for number in raw]
                goal = (x+y)%N_val
                a = output[3]
                c = output[1]
                f = output[0]
                print(f"ACCMOD: {x}+{y}%{N_val}={goal}    a={a}:{a == goal}\n    c=0:{c==0}, c={c}\n    f=0:{f==0}, f={f}")

    def test_multmod(n):
        for x in range(1,2 ** (n - 1)):
            for y in range(1,2 ** (n - 1)):
                a = QuantumRegister(n, 'a')
                b = y
                N = 2 ** (n - 1)
                c = QuantumRegister(n + 1, 'c')
                r = QuantumRegister(n, 'r')
                f = QuantumRegister(1, 'f')
                a_out = ClassicalRegister(len(a), 'a_o')
                c_out = ClassicalRegister(len(c), 'c_o')
                r_out = ClassicalRegister(len(r), 'r_o')
                f_out = ClassicalRegister(len(f), 'f_o')
                qc = AdditionalGates(a, c, r, f, a_out, c_out, r_out, f_out)

                for i in range(n):
                    if (x >> i) & 1:
                        qc.x(a[i])

                qc.const_mult_mod(a, b, N, c, r, f)

                qc.measure(a, a_out)
                qc.measure(c, c_out)
                qc.measure(r, r_out)
                qc.measure(f, f_out)

                # qc.draw(output='mpl')
                # plt.show()

                res = sim.run(transpile(qc, sim), shots=1).result()
                raw = list(res.get_counts())[0].split(" ")
                output = [int(number, 2) for number in raw]
                goal = (x*y)%N
                a = output[3]
                c = output[2]
                r = output[1]
                f = output[0]
                print(f"MULTMOD: {x}*{y}%{N}={goal}    r={r}:{r == goal}\n    c=0:{c},f=0:{f}")
                if x==y:
                    break

    def test_multmod_self(n):
        for x in range(1,2 ** (n - 1)):
            for y in range(1,2 ** (n - 1)):
                b = y
                N = 7
                if gcd(b, N) == 1:
                    a = QuantumRegister(n, 'a')
                    c = QuantumRegister(n + 1, 'c')
                    r1 = QuantumRegister(n, 'r1')
                    f = QuantumRegister(1, 'f')
                    a_out = ClassicalRegister(len(a), 'a_o')
                    c_out = ClassicalRegister(len(c), 'c_o')
                    f_out = ClassicalRegister(len(f), 'f_o')
                    r1_out = ClassicalRegister(len(r1), 'r1_o')
                    qc = AdditionalGates(a, c, r1, f, a_out, c_out, f_out, r1_out)

                    for i in range(n):
                        if (x >> i) & 1:
                            qc.x(a[i])

                    qc.const_mult_mod_self(a, b, N, c, r1, f)


                    qc.measure(a, a_out)
                    qc.measure(c, c_out)
                    qc.measure(f, f_out)

                    # qc.draw(output='mpl')
                    # plt.show()

                    res = sim.run(transpile(qc, sim), shots=1).result()
                    raw = list(res.get_counts())[0].split(" ")
                    output = [int(number, 2) for number in raw]
                    goal = (x*y)%N
                    a = output[3]
                    c = output[2]
                    f = output[1]
                    r1 = output[0]
                    print(f"MULTMOD: {x}*{y}%{N}={goal}    a={a}:{a == goal}\n    c={c}, f={f}, r1={r1}")
                    if x==y:
                        break

    n=4
    #test_addself(n)
    #test_subself(n)
    #test_geq(n)
    #test_addmodself(n)
    #test_multmod(n)
    test_multmod_self(n)

if __name__ == "__main__" and True:
    from math import ceil, log2
    from qiskit.circuit.library import QFTGate

    a = 2
    N = 15
    if gcd(a, N) > 1:
        print(f"Error: gcd({a},{N}) > 1")
        quit()

    k = 2 * ceil(log2(N))
    b_list = [pow(a, 2 ** val, N) for val in range(k)]

    work_bits = k // 2
    phase_bits = k

    Phase_Register = QuantumRegister(phase_bits, "P")
    Work_Register = QuantumRegister(work_bits, "W")
    Carry = QuantumRegister(work_bits + 1, "C")
    Accumulator1 = QuantumRegister(work_bits, "A1")
    Flag = QuantumRegister(1, "F")
    Ancilla = QuantumRegister(1, "anc")
    meas = ClassicalRegister(phase_bits, "meas")
    qc = AdditionalGates(Phase_Register, Work_Register, Carry, Accumulator1, Flag, Ancilla, meas)
    qc.h(Phase_Register)
    qc.x(Work_Register[0])

    for index in range(len(b_list)):
        b = b_list[index]
        if b > 1:
            qc.const_mult_mod_self(Work_Register, b, N, Carry, Accumulator1, Flag, Phase_Register[index], Ancilla)

    qc.compose(QFTGate(phase_bits).inverse(), Phase_Register, inplace=True)
    qc.measure(Phase_Register, meas)