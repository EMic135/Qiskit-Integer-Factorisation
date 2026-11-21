from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.circuit.library import UnitaryGate
import numpy as np
from math import gcd as gcd


class AdditionalGates(QuantumCircuit):
    def and_gate(self, a, b, r, ctrl=None, anc=None):
        """applies "r = a AND b" to the current circuit

        :param a: Qubit | Bool | Int. the "a" in "r = a AND b"
        :param b: Qubit | Bool | Int. the "b" in "r = a AND b"
        :param r: Qubit. the "r" in "r = a AND b"
        :param ctrl: Qubit | None. controls if "a AND b" is written to "r"
        :param anc: Qubit | QubitRegister | None. Ancilla qubit(s) if required/provided.

        :raise TypeError: if "a" or "b" are not a Qubit|Bool|Int.

        """
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
        """
        applies "r = a XOR b" to the current circuit

        :param a: Qubit | Bool | Int. the "a" in "r = a XOR b"
        :param b: Qubit | Bool | Int. the "b" in "r = a XOR b"
        :param r: Qubit. the "r" in "r = a XOR b"
        :param ctrl: Qubit | None. controls if "a XOR b" is written to "r"

        :raise TypeError: if "a" or "b" are not a Qubit|Bool|Int.
        """
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
        """
        applies "r = a OR b" to the current circuit through an "and" and "xor" gate

        :param a: Qubit | Bool | Int. the "a" in "r = a OR b"
        :param b: Qubit | Bool | Int. the "b" in "r = a OR b"
        :param r: Qubit. the "r" in "r = a OR b"
        :param ctrl: Qubit | None. controls if "a OR b" is written to "r"

        :raise TypeError: if "a" or "b" are not a Qubit|Bool|Int.
        """
        self.and_gate(a, b, r, ctrl)
        self.xor_gate(a, b, r, ctrl)

    def copy_gate(self, a, b, ctrl=None):
        """
        applies "b = a XOR b" to the current circuit, which when "b=0" results in "b=a"

        :param a: Qubit | Bool | Int. the "a" in "b = a XOR b"
        :param b: Qubit. the "b" in "b = a XOR b"
        :param ctrl: Qubit | None. controls if "a XOR b" is written to "b"

        :raise TypeError: if "a" is not a Qubit|Bool|Int or "b" is not a Qubit.
        """
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
        """
        inverts "a" such that all 1s go to 0s and 0s go to 1s

        :param a: QubitRegister | Qubit | [Bool] | [Int]. the "a" in "b = a XOR b"
        """
        if type(a) is QuantumRegister or type(a) is Qubit:
            self.x(a)
        else:
            a = [1 - val for val in a]
        return a

    def carry(self, a, b, c, c_out, ctrl=None, anc=None):
        """computes "c_out = (a+b+c>1)"

        :param a: Qubit | Bool | Int. the "a" in "c_out = (a+b+c>1)"
        :param b: Qubit | Bool | Int. the "b" in "c_out = (a+b+c>1)"
        :param c: Qubit | Bool | Int. the "c" in "c_out = (a+b+c>1)"
        :param c_out: Qubit. the "c_out" in "c_out = (a+b+c>1)"
        :param ctrl: Qubit | None. controls if "(a+b+c>1)" is written to "c_out"
        :param anc: Qubit | QubitRegister | None. Ancilla qubit(s) if required/provided.

        :raise TypeError: if "a" or "b" are not a Qubit|Bool|Int.
        """
        self.and_gate(a, b, c_out, ctrl, anc)
        self.and_gate(a, c, c_out, ctrl, anc)
        self.and_gate(c, b, c_out, ctrl, anc)

    def add_self(self, a, b, c, ctrl=None, anc=None):
        """computes "a=a+b"

        :param a: QubitRegister | [Qubit]. the "a" in "a=a+b"
        :param b: QubitRegister | [Qubit] | [Bool] | [Int]. the "b" in "a=a+b"
        :param c: QubitRegister | [Qubit]. the carry qubits required whilst computing the operation
        :param ctrl: Qubit | None. controls if "a+b" is written to "a"
        :param anc: QubitRegister | [Qubit] | Qubit | None. Ancilla qubit(s) if required/provided.

        :raise ValueError: if "c" does not have at least 1 more qubit than "a" and "b"
        :raise ValueError: if "a" and "b" are not equal sizes
        """
        if len(c) <= len(a) or len(c) <= len(b):
            raise ValueError('c must be larger than a and b')
        if len(a) != len(b):
            raise ValueError('a must be the same size as b')

        for i in range(len(a)):
            self.carry(a[i], b[i], c[i], c[i + 1], anc = anc)
        for i in range(len(a)-1, -1, -1):
            self.carry(a[i], b[i], c[i], c[i+1], anc = anc)
            self.xor_gate(b[i], c[i], a[i], ctrl)

    def sub_self(self, a, b, c, ctrl=None, anc=None):
        """computes "a=a-b"

        :param a: QubitRegister | [Qubit]. the "a" in "a=a-b"
        :param b: QubitRegister | [Qubit] | [Bool] | [Int]. the "b" in "a=a-b"
        :param c: QubitRegister | [Qubit]. the carry qubits required whilst computing the operation
        :param ctrl: Qubit | None. controls if "a-b" is written to "a"
        :param anc: QubitRegister | [Qubit] | Qubit | None. Ancilla qubit(s) if required/provided.

        :raise ValueError: if "c" does not have at least 1 more qubit than "a" and "b"
        :raise ValueError: if "a" and "b" are not equal sizes
        """
        if len(c) <= len(a) or len(c) <= len(b):
            raise ValueError('c must be larger than a and b')
        if len(a) != len(b):
            raise ValueError('a must be the same size as b')

        b = self.invert(b)
        self.x(c[0])

        self.add_self(a, b, c, ctrl, anc)

        b = self.invert(b)
        self.x(c[0])

    def geq(self, a, b, c, f, ctrl=None, anc=None):
        """computes "f = a>=b"

        :param a: QubitRegister | [Qubit]. the "a" in "f = a>=b"
        :param b: QubitRegister | [Qubit] | [Bool] | [Int]. the "b" in "f = a>=b"
        :param c: QubitRegister | [Qubit]. the carry qubits required whilst computing the operation
        :param f: Qubit. the "f" in "f = a>=b"
        :param ctrl: Qubit | None. controls if "a>=b" is written to "f"
        :param anc: QubitRegister | [Qubit] | Qubit | None. Ancilla qubit(s) if required/provided.

        :raise ValueError: if "c" does not have at least 1 more qubit than "a" and "b"
        :raise ValueError: if "a" and "b" are not equal sizes
        """
        if len(c) <= len(a) or len(c) <= len(b):
            raise ValueError('c must be larger than a and b')
        if len(a) != len(b):
            raise ValueError('a must be the same size as b')

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
        """computes "a = a+b % N"

        :param a: QubitRegister | [Qubit]. the "a" in "a = a+b % N"
        :param b: QubitRegister | [Qubit] | [Bool] | [Int]. the "b" in "a = a+b % N"
        :param N: QubitRegister | [Qubit] | [Bool] | [Int]. the "N" in "a = a+b % N"
        :param c: QubitRegister | [Qubit]. the carry qubits required whilst computing the operation
        :param f: Qubit. the flag qubit required whilst computing the operation
        :param ctrl: Qubit | None. controls if "a+b % N" is written to "a"
        :param anc: QubitRegister | [Qubit] | Qubit | None. Ancilla qubit(s) if required/provided.

        :raise ValueError: if "c" does not have at least 1 more qubit than "a" and "b"
        :raise ValueError: if "a" and "b" are not equal sizes
        """
        if len(c) <= len(a) or len(c) <= len(b):
            raise ValueError('c must be larger than a and b')
        if len(a) != len(b):
            raise ValueError('a must be the same size as b')

        self.add_self(a, b, c, ctrl, anc=anc)
        self.geq(a, N, c, f, ctrl, anc=anc)
        self.sub_self(a, N, c, f, anc=anc)
        self.sub_self(a, b, c, ctrl, anc=anc)
        self.geq(a, N, c, f, ctrl, anc=anc)
        self.add_self(a, b, c, ctrl, anc=anc)

    def const_mult_mod(self, a:QuantumRegister, b:int, N:int, c, r, f, ctrl=None, anc=None):
        """computes "r = ab % N"

        :param a: QubitRegister | [Qubit]. the "a" in "r = ab % N"
        :param b: Int. the "b" in "r = ab % N"
        :param N: Int. the "N" in "r = ab % N"
        :param c: QubitRegister | [Qubit]. the carry qubits required whilst computing the operation
        :param a: QubitRegister | [Qubit]. the "r" in "r = ab % N"
        :param f: Qubit. the flag qubit required whilst computing the operation
        :param ctrl: Qubit | None. controls if "ab % N" is written to "r"
        :param anc: QubitRegister | [Qubit] | Qubit | None. Ancilla qubit(s) if required/provided.

        :raise ValueError: if "ctrl" is not none and "anc" is None, there are not enough qubits
        """
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
        """computes "a = ab % N"

        :param a: QubitRegister | [Qubit]. the "a" in "a = ab % N"
        :param b: Int. the "b" in "a = ab % N"
        :param N: Int. the "N" in "a = ab % N"
        :param c: QubitRegister | [Qubit]. the carry qubits required whilst computing the operation
        :param acu1: QubitRegister | [Qubit]. the accumulator qubits required whilst computing the operation
        :param f: Qubit. the flag qubit required whilst computing the operation
        :param ctrl: Qubit | None. controls if "ab % N" is written to "a"
        :param anc: QubitRegister | [Qubit] | Qubit | None. Ancilla qubit(s) if required/provided.
        """
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




if __name__ == "__main__":
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
                #print(f"ADD: {x}+{y}={goal}    a={a}:{a==goal}\n    b={b}:{b==y},c=0:{c==0}")
                if a!=goal or b != y or c != 0:
                    print(f"ADD FAILED: {x}+{y}={goal}    a={a}:{a==goal}\n    b={b}:{b==y},c=0:{c==0}")
                    return None
        print("ADD PASSED")

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
                #print(f"SUB: {x}-{y}={goal}    a={a}:{a==goal}\n    b={y}:{b==y},c=0:{c==0}")
                if a != goal or b != y or c != 0:
                    print(f"SUB FAILED: {x}-{y}={goal}    a={a}:{a==goal}\n    b={y}:{b==y},c=0:{c==0}")
                    return None
        print("SUB PASSED")

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
                #print(f"GEQ: {x}>={y}={goal}    f={f}:{f==goal}\n    a={a}:{a==x},b={b}:{b==y},c=0:{c==0}")
                if f != goal or a != x or b != y or c != 0:
                    print(f"GEQ FAILED: {x}>={y}={goal}    f={f}:{f==goal}\n    a={a}:{a==x},b={b}:{b==y},c=0:{c==0}")
                    return None
        print("GEQ PASSED")

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

                qc.add_mod_self(a, b, N, c, f, anc=anc)

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
                #print(f"ADDMOD: {x}+{y}%{N_val}={goal}    a={a}:{a == goal}\n    c=0:{c==0}, c={c}\n    f=0:{f==0}, f={f}")
                if a != goal or c != 0 or f!=0:
                    print(f"ADDMOD FAILED: {x}+{y}%{N_val}={goal}    a={a}:{a == goal}\n    c=0:{c==0}, c={c}\n    f=0:{f==0}, f={f}")
                    return None
        print("ADDMOD PASSED")

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
                #print(f"MULTMOD: {x}*{y}%{N}={goal}    r={r}:{r == goal}\n    c=0:{c==0}, c={c}\n    f=0:{f==0}, f={f}")
                if r != goal or c != 0 or f != 0:
                    print(f"MULTMOD FAILED: {x}*{y}%{N}={goal}    r={r}:{r == goal}\n    c=0:{c==0}, c={c}\n    f=0:{f==0}, f={f}")
                    return None
                if x==y:
                    break
        print("MULTMOD PASSED")

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
                    #print(f"MULTMOD: {x}*{y}%{N}={goal}    a={a}:{a == goal}\n    c={c}, f={f}, r1={r1}")
                    if a != goal or c != 0 or f != 0:
                        print( f"MULTMOD FAILED: {x}*{y}%{N}={goal}    a={a}:{a == goal}\n    c=0:{c==0}, c={c}\n    f=0:{f==0}, f={f}\n    r1=0:{r1==0}, r1={r1}")
                        return None
                    if x == y:
                        break
        print("MULTMOD PASSED")

    print("BEGINNING TESTS")
    n=4
    test_addself(n)
    test_subself(n)
    test_geq(n)
    test_addmodself(n)
    test_multmod(n)
    test_multmod_self(n)