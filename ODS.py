import numpy as np


class ODE_Solver(object):
    """
    Суперкласс для решения ОДУ или системы ОДУ нормированной и приведенной к виду: du/dx = f(u, x)

    Атрибуты класса:
    x: массив узлов точек координаты x
    u: массив решения ОДУ в точках узла x
    k: число шагов вычисления значения в узлах
    f: функция правой части ОДУ, реализованная в виде: f(u, x)

    Погрешность вычислений явлется суммой погрешности дискретизации
    и погрешностей округления.
    """

    def __init__(self, f):
        if not callable(f):
            # проверка корректности типа передаваемой функции f(u, x)
            raise TypeError('f не %s, является функцией' % type(f))
        # инициализация функции правой части ОДУ
        self.f = lambda u, x: np.asarray(f(u, x), float)
        self.k = None

    def solver_step(self):
        """
        Реализация конкретной численной схемы решателя.
        Вычисление значения функции решения ОДУ на одном шаге
        Функция возвращает значение решения на шаге k.
        """

        raise NotImplementedError("Метод не имеет реализации или неверно инициализирован параметр solver_type.")

    def set_initial_condition(self, u0):
        """
        Функция  инициализации начальных условий задачи Коши.
        По умолчанию функция возвращает False, если нет элементов.
        """

        if isinstance(u0, (float, int)):  # ОДУ является одномерным
            self.neq = 1
            u0 = float(u0)
        else:  # система ОДУ, ОДУ порядка выше 1-го
            u0 = np.asarray(u0)  # (начальные условия вектор-функция - порядок 2+)
            self.neq = u0.size
        self.u0 = u0

        # Проверка, что функция возвращает вектор f корректной длины:
        try:
            f0 = self.f(self.u0, 0)
        except IndexError:
            raise IndexError(
                'Индекс вектора u выходит за границы f(u,x). Допустимые индексы %s' % (str(range(self.neq))))
        if f0.size != self.neq:
            raise ValueError('f(u,x) возвращено %d элементов, вектор u имеет %d элементов' % (f0.size, self.neq))

    def solve(self, coord_points, terminate=None):
        """
        Решение ОДУ и  запись полученных значений в виде массива или списка.
        По умолчанию функция возвращает False, если нет элементов.
        """

        # используется для контроля шага - в случае реализации адаптивных методов позволяет
        # вести контроль за выборо шага на сетке узлов решения. Не используется для методов:
        # FE, RK4, ABM4, MLN.
        if terminate is None:
            terminate = lambda u, x, step_no: False

        if isinstance(coord_points, (float, int)):
            raise TypeError('solve: массив точек x не содержит чисел.')
        self.x = np.asarray(coord_points)
        if self.x.size <= 1:
            raise ValueError('ODESolver.solve требует на вход массив координат'
                             ' точек поиска решения. Число точек меньше 2!')

        n = self.x.size
        if self.neq == 1:  # ОДУ 1-го порядка
            self.u = np.zeros(n)
        else:  # ОДУ порядка 2+ или система ОДУ
            self.u = np.zeros((n, self.neq))

        # Присвоить self.u[0] значение начального условия  self.u0
        self.u[0] = self.u0

        # Проход по сетке координат узлов, в которых отыскивается решение.
        # Используется векторизация посрдеством хранения данных в  numpy array.
        for k in range(n - 1):
            self.k = k
            self.u[k + 1] = self.solver_step()
            if terminate(self.u, self.x, self.k + 1):
                break
        return self.u[: k + 2], self.x[: k + 2]
