from typing import List, Tuple, Union
from sympy import Eq, Matrix, Symbol, latex, sympify, zeros
# from sympy import Rational, symbols
# from IPython.display import Math, display

from core import CommonMatrixCalculator

EquationList = List[Eq]
SymbolList = List[Symbol]


class LinearSystemConverter(CommonMatrixCalculator):

    @staticmethod
    def str_to_Eq(expressions: List[str], get_unknowns: bool = False) -> Union[EquationList, Tuple[EquationList, SymbolList]]:
        """Convert string expressions to SymPy equations.

        Args:
            expressions: List of string expressions representing equations
            get_unknowns: Boolean flag to indicate whether to extract unknowns from the last line

        Returns:
            List of SymPy equations or tuple of equations and unknowns
        """
        unknowns_str = ''
        expressions = expressions.split('\n')
        if '=' not in expressions[-1]:
            unknowns_str = expressions[-1]
            expressions = expressions[:-1]
        elif get_unknowns:
            raise ValueError("Please specify unknowns in the last line")
        eq_list = []
        for i, expr in enumerate(expressions):
            if '=' not in expr:
                raise ValueError(f"Equation format error in line {i+1}")
            left, right = expr.split('=')
            eq = Eq(sympify(left), sympify(right))
            eq_list.append(eq)
        if get_unknowns:
            unkowns = [sympify(sym) for sym in unknowns_str.split(' ')]
            return eq_list, unkowns
        return eq_list

    def equations_to_matrix(self, equations: EquationList, unknowns: SymbolList = None, parameters: SymbolList = None) -> Tuple[Matrix, Matrix, Matrix, SymbolList]:
        """Convert linear system of equations to matrix form Ax = b.
        Supports parameters in both A and b matrices.

        Args:
            equations: List of equations
            unknowns: List of unknown variables
            parameters: List of parameters (optional). If not specified,
                       all symbols that are not unknowns are treated as parameters

        Returns:
            Tuple containing coefficient matrix A, unknown vector x,
            constant vector b, and parameter list
        """
        # Process input equations
        processed_eqs = []
        for eq in equations:
            processed_eqs.append(eq.lhs - eq.rhs)

        # Extract all symbols
        all_symbols = set()
        for eq in processed_eqs:
            all_symbols.update(eq.free_symbols)
        all_symbols = sorted(list(all_symbols), key=str)

        # Determine unknowns and parameters
        if unknowns is None:
            # If unknowns not specified, use all symbols
            unknowns = all_symbols
            parameters = []
        else:
            # If unknowns specified, determine parameters
            if parameters is None:
                parameters = [
                    sym for sym in all_symbols if sym not in unknowns]

        # Build coefficient matrix A and constant vector b
        n_eq = len(processed_eqs)
        n_unknowns = len(unknowns)

        A = zeros(n_eq, n_unknowns)
        b = zeros(n_eq, 1)

        for i, eq in enumerate(processed_eqs):
            # For each equation, extract coefficients of unknowns
            for j, unknown in enumerate(unknowns):
                # Extract coefficient of unknown (may contain parameters)
                coeff = eq.coeff(unknown)
                A[i, j] = coeff
                # Subtract this term from equation
                eq = eq - coeff * unknown

            # Remaining part (with parameters and constants) goes to b with sign change
            b[i, 0] = -eq

        # Create unknown vector
        x = Matrix(unknowns)

        return A, x, b, parameters

    def matrix_to_equations(self, A: Matrix, x: Matrix, b: Matrix) -> EquationList:
        """
        Convert matrix form Ax = b back to system of linear equations.
        Supports parameters in A and b.

        Args:
            A: Coefficient matrix
            x: Unknown vector
            b: Constant vector

        Returns:
            List of equations
        """
        equations = []
        for i in range(A.rows):
            lhs = 0
            for j in range(A.cols):
                if A[i, j] != 0:
                    if lhs == 0:
                        lhs = A[i, j] * x[j]
                    else:
                        lhs += A[i, j] * x[j]
            equations.append(Eq(lhs, b[i]))

        return equations

    def show_equations_to_matrix(self, equations: str, unknowns: SymbolList = None, parameters: List = None, show_steps: bool = True) -> Tuple[Matrix, Matrix, Matrix, SymbolList]:
        """
        Display the conversion process from equations to matrix form, supports parameters.

        Args:
            equations: str of system of equations
            unknowns: List of unknown variables
            parameters: List of parameters
            show_steps: Flag to show step-by-step process

        Returns:
            Tuple containing coefficient matrix A, unknown vector x,
            constant vector b, and parameter list
        """
        if isinstance(equations, str):
            equations = self.str_to_Eq(equations)

        if show_steps:
            self.step_generator.clear()
            self.add_step("方程组到矩阵的转换")

        A, x, b, params = self.equations_to_matrix(
            equations, unknowns, parameters)

        if show_steps:
            # Display original system of equations
            self.add_step("原始线性方程组")
            eq_latex = r"\begin{cases}"
            for eq in equations:
                if isinstance(eq, Eq):
                    eq_latex += latex(eq.lhs) + " = " + latex(eq.rhs) + r" \\ "
                else:
                    eq_latex += latex(eq) + " = 0" + r" \\ "
            eq_latex += r"\end{cases}"
            self.step_generator.add_step(eq_latex)

            # Display unknowns and parameters information
            if unknowns:
                u_str = ''
                for u in unknowns:
                    u_str += latex(u) + r"\;\;"
                self.step_generator.add_step(
                    r"\text{未知数: }" + u_str)
            if params:
                p_str = ''
                for p in params:
                    p_str += latex(p) + r"\;\;"
                self.step_generator.add_step(
                    r"\text{参数: }" + p_str)

            # Display matrix form
            self.add_step("矩阵形式")
            self.step_generator.add_step(rf"A = {latex(A)}")
            self.step_generator.add_step(rf"\boldsymbol{{x}} = {latex(x)}")
            self.step_generator.add_step(rf"\boldsymbol{{b}} = {latex(b)}")

            # Display complete matrix equation
            self.add_step("完整矩阵方程")
            self.step_generator.add_step(f"{latex(A)} {latex(x)} = {latex(b)}")

        return A, x, b, params

    def show_matrix_to_equations(self, A: Matrix, x: Matrix, b: Matrix, show_steps: bool = True) -> EquationList:
        """
        Display the conversion process from matrix to equations, supports parameters.

        Args:
            A: Coefficient matrix
            x: Unknown vector
            b: Constant vector
            show_steps: Flag to show step-by-step process

        Returns:
            List of equations
        """
        if show_steps:
            self.step_generator.clear()
            self.add_step("矩阵到方程组的转换")

        equations = self.matrix_to_equations(A, x, b)

        if show_steps:
            # Display matrix form
            self.add_step("矩阵形式")
            self.step_generator.add_step(rf"A = {latex(A)}")
            self.step_generator.add_step(rf"\boldsymbol{{x}} = {latex(x)}")
            self.step_generator.add_step(rf"\boldsymbol{{b}} = {latex(b)}")
            self.step_generator.add_step(f"{latex(A)} {latex(x)} = {latex(b)}")

            # Display converted system of equations
            self.add_step("对应的线性方程组")
            eq_latex = r"\begin{cases}"
            for eq in equations:
                eq_latex += latex(eq) + r" \\ "
            eq_latex += r"\end{cases}"
            self.step_generator.add_step(eq_latex)

        return equations


# def demo():
#     converter = LinearSystemConverter()

#     # Define symbols
#     x, y, z = symbols('x y z')
#     a, b, c, d, k, m, n = symbols('a b c d k m n')

#     # Example 1: Simple parametric system

#     converter.step_generator.add_step(r"\textbf{示例1: 简单的参数系统}")
#     equations1 = [
#         Eq(a*x + b*y, c),
#         Eq(d*x - b*y, k)
#     ]
#     converter.show_equations_to_matrix(
#         equations1, unknowns=[x, y])
#     display(Math(converter.get_steps_latex()))

#     # Example 2: Mixed case - partially specified unknowns

#     converter.step_generator.add_step(r"\textbf{示例2: 混合情况}")
#     equations2 = [
#         Eq(2*x + a*y, b + 1),
#         Eq(b*x - 3*y, c)
#     ]
#     converter.show_equations_to_matrix(
#         equations2, unknowns=[x, y])
#     display(Math(converter.get_steps_latex()))

#     # Example 3: 3-variable system with multiple parameters

#     converter.step_generator.add_step(r"\textbf{示例3: 三变量多参数系统}")
#     equations3 = [
#         Eq(a*x + b*y + c*z, d),
#         Eq(2*x + k*y - z, m),
#         Eq(x + y + z, n)
#     ]
#     A3, x3, b3, _ = converter.show_equations_to_matrix(
#         equations3, unknowns=[x, y, z])
#     display(Math(converter.get_steps_latex()))

#     # Example 4: Convert matrix form back to equations (with parameters)

#     converter.step_generator.add_step(rf"\textbf{{示例4: 矩阵形式转回方程组(带参数)}}")
#     converter.show_matrix_to_equations(A3, x3, b3)
#     display(Math(converter.get_steps_latex()))

#     # Example 5: Explicitly specify parameters

#     converter.step_generator.add_step(r"\textbf{示例5: 明确指定参数}")
#     equations5 = [
#         Eq(k*x + m*y, n),
#         Eq(2*k*x - y, 3*m)
#     ]
#     converter.show_equations_to_matrix(
#         equations5, unknowns=[x, y], parameters=[k, m, n])
#     display(Math(converter.get_steps_latex()))

#     # Example 6: Physics system example - spring-mass system

#     converter.step_generator.add_step(r"\textbf{示例6: 物理系统示例}")
#     F1, F2, k1, k2, k3 = symbols('F1 F2 k1 k2 k3')
#     x1_sym, x2_sym = symbols('x1 x2')

#     equations6 = [
#         Eq((k1 + k2)*x1_sym - k2*x2_sym, F1),
#         Eq(-k2*x1_sym + (k2 + k3)*x2_sym, F2)
#     ]
#     converter.show_equations_to_matrix(
#         equations6, unknowns=[x1_sym, x2_sym])
#     display(Math(converter.get_steps_latex()))

#     converter.step_generator.add_step(r"\textbf{示例7: 数值示例}")
#     x, y = symbols('x y')

#     equations7 = [
#         Eq(8*x - 3*y, 6),
#         Eq(10*x + 2*y, 8)
#     ]
#     converter.show_equations_to_matrix(
#         equations7, unknowns=[x, y])
#     display(Math(converter.get_steps_latex()))

#     converter.step_generator.add_step(r"\textbf{示例8: 分数示例}")
#     x, y = symbols('x y')

#     equations8 = [
#         Eq(8*x - Rational(5, 6)*y - 6*a, 0),
#         Eq(10*x + 2*y, 8)
#     ]
#     converter.show_equations_to_matrix(
#         equations8, unknowns=[x, y])
#     display(Math(converter.get_steps_latex()))


# if __name__ == "__main__":
#     demo()
