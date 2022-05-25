import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from tabulate import tabulate
from ADS import ABM4
from ES import FE
from MLNS import MLN
from RKS import RK4


class Solver(object):
    def __init__(self, problem, dx, solver_type=1):
        """
        Инициализация начальных параметров.
        """
        self.problem, self.dx = problem, dx
        self.solver_type = solver_type
        self.solver = self._choose_sch(solver_type)

    @staticmethod
    def _choose_sch(solver_type):
        methods = {1: FE, 2: RK4, 3: ABM4, 4: MLN}
        if solver_type in methods:
            return methods[solver_type]
        else:
            raise ValueError('not choose numerical scheme!')

    def dsolve(self):
        """
        Функция запсука решателя задачи Коши в соответствии с выбранной разностной схемой.
        Возвращает массив nd.array значений узлов одномерной сетки и численного решения.
        :return: self.u, self.x
        """
        solver = self.solver(self.problem)
        solver.set_initial_condition(self.problem.u0)
        n = int(round(self.problem.End / self.dx))
        x_points = np.linspace(0, self.problem.End, n + 1)
        self.u, self.x = solver.solve(x_points)


    def table_dsol(self):
        """
        Функция вывода таблицы значений решения и погрешности численного моделирования.
        Возвращает максимальное значение погрешности вычислений.
        """
        type_slt = ['FE', 'RK4', 'ABM4', 'MLN']
        res = [list(self.x), list(self.u[:, 0]), list(self.exact()), list(self.u[:, 0] - self.exact())]
        print(tabulate(list(zip(*res)), headers=['iter', 'x', 'y(x)_%3s ', ' y_exact(x)', 'Accuracy'],
                       floatfmt=(".2f",".4f",".8f",".8f",".10f"),
                       tablefmt='github', showindex="always", colalign=("center",))
              % (type_slt[self.solver_type - 1]))
        print(end='\n')
        print('Максимальное значение погрешности вычислений: {0:f}'
              .format(np.round(np.max(self.u[:, 0] - self.exact()), 6)), end='\n')
        return np.round(np.max(self.u[:, 0] - self.exact()), 6)

    def exact(self):
        """
        Функция возвращает массив точек точного аналитического решения задачи Коши: y(x).
        """
        return (np.exp(-3 * self.x) * (
                -129 * (self.x ** 4) - 16 * (self.x ** 3) + 54 * (self.x ** 2) + 36 * self.x)) / 12

    def plot_sol(self):
        """
        Функция графической визуализации точного решения и результатов численного моделирования.
        """
        plt.plot(self.x, self.u[:, 0], 'b--o', linewidth=1, markersize=4)
        plt.plot(self.x, self.exact(), '-r', linewidth=2)
        plt.title('Численное решение задачи Коши: dx=%g'
                  % (self.x[1] - self.x[0]))
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y(x)$', rotation='horizontal')
        plt.legend(["Численное решение ОДУ", "Точное решение"])
        plt.grid(color='grey', linestyle='--', linewidth=0.7)
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='grey', linestyle='--', alpha=0.2)
        plt.show()

    def plot_accuracy(self):
        """
        Функция графической визуализации динамики погрешности вычислений.
        """
        acc_err = self.u[:, 0] - self.exact()
        type_slt = ['FE', 'RK4', 'ABM4', 'MLN']
        plt.plot(self.x, acc_err, 'k--o', linewidth=1, markersize=4)
        plt.title('Погрешность численного решения: dx=%g ' 'метода %s'
                  % ((self.x[1] - self.x[0]), type_slt[self.solver_type - 1]))
        plt.xlabel(r'$x$')
        plt.ylabel(r'$e(x)$', rotation='vertical')
        plt.grid(color='grey', linestyle='--', linewidth=0.7)
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='grey', linestyle='--', alpha=0.2)
        plt.show()

    def norm_accuracy(self):
        """
        Функция оценки нормы погрешности вычислений. Возвращает значение нормы погрешности
        численного моделирования. Оценивается стандартная номра L2.
        """
        acc_err = self.u[:, 0] - self.exact()
        norm = np.round((LA.norm(acc_err, 2)), 8)
        print('Норма погрешности вычислений: {0:f}'.format(norm), end='\n')
        return norm

    def err_sch(self):
        """
        Функция оценки погрешности дискретизации выбранной численной схемы.
        Возвращает значение оценки "сверху" погрешности дискретизации метода
        err_sch. Данная погрешность уже входит в погрешность расчета.
        Значение оценки существенно зависит от выбранного шага дискретизации сетки узлов.
        :return: err
        """
        dx_err = (self.x[5] - self.x[4])
        if self.solver_type == 1:
            err = np.round((dx_err ** 2), 6)
        elif self.solver_type == 2:
            err = np.round((dx_err ** 4), 6)
        elif self.solver_type == 3:
            err = np.round((dx_err ** 4), 6)
        elif self.solver_type == 4:
            err = np.round((dx_err ** 3), 6)
        else:
            raise ValueError('Невозможно оценить погрешность. Численная схема не выбрана!')
        print('Погрешность дискретизации (оценка "сверху"): {0:f}'.format(err), end='\n')
        return err

    def accuracy_approx(self,dx):
        """
        Функция оценки динамики нормы в зависимости от выбранного шага интегрирования.
        Возвращает значения нормы для каждого шага. Также позволяет визуализировать
        графически динамику нормы погрешности расчета, в зависимости от выбранного шага и
        метода (разностной схемы).
        :param dx:
        :return: norm_list
        """
        type_slt = ['FE', 'RK4', 'ABM4', 'MLN']
        norm_list = []
        for curr_dx in dx:
            solver = self.solver(self.problem)
            n = int(round(self.problem.End / curr_dx))
            solver.set_initial_condition(self.problem.u0)
            x_points = np.linspace(0, self.problem.End, n + 1)
            self.u, self.x = solver.solve(x_points)
            norm = np.round((LA.norm(self.u[:, 0] - self.exact(), 2)), 8)
            norm_list.append(norm)
        print(norm_list)
        plt.plot(dx, norm_list,'k--o', linewidth=1, markersize=4)
        plt.title('Динамика нормы погрешности при различных dx для метода %s'
                  % (type_slt[self.solver_type - 1]))
        plt.xlabel(r'$dx$')
        plt.ylabel(r'$E_{norm}(x)$', rotation='vertical')
        plt.grid(color='grey', linestyle='--', linewidth=0.7)
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='grey', linestyle='--', alpha=0.2)
        plt.show()
        return norm_list