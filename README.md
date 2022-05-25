# ODE solver Final CODEMIKA Project
<br>
Программная имплементация проблемно-ориентированного ПО для решения
задачи Коши представленного ОДУ с начальными условиями:

$$\frac{d^{5}y}{dx^{5}} + 15 \frac{d^{4}y}{dx^{4}} + 90 \frac{d^{3}y}{dx^{3}} + 270 \frac{d^{2}y}{dx^{2}} +
405 \frac{dy}{dx} + 243y = 0; \quad x \in [0;5]$$


$$ y\Big|_{x = 0} = 0; \quad \frac{dy}{dx}\Big| _ {x = 0} = 3;  \quad \frac{d^{2}y}{dx^{2}}\Big| _ {x = 0} = -9;
\quad \frac{d^{3}y}{dx^{3}}\Big| _ {x = 0} = -8; \quad \frac{d^{4}y}{dx^{4}}\Big| _ {x = 0} = 0. $$

<br>
**Точное аналитическое решение задачи Коши** $$ y(x) $$:

$$ \displaystyle y{\left(x \right)} = \frac{e^{- 3 x} \left(- 129 x^{4} - 16 x^{3} + 54 x^{2} + 36x\right)}{12} $$

<br>
В рамках данного проекта реализованы **4 численные схемы** численной аппроксимации полученного
точного аналитического решения. 

- Прямой метод Эйлера 2 порядка точности (явная схема);
- Метод Рунге-Кутта 4 порядка точности (явная схема);
- Метод Адамса-Башфорта-Моултона 4 порядка точности (явная схема);
- Метод Милнса 4 порядка точности.

Программой предусмотрены _следующие основные возможности_:
1. Получение численного решения задачи Коши 
2. Форматированный вывод результатов расчетов в виде таблицы
3. Расчет нормы погрешности численной схемы
4. Расчет оценки погрешности дсикретизации выбранной численной схемы
5. Вывод графика точного решения и результатов численного моделирования
6. Вывод графика погрешности расчета
7. Расчет и вывод графика динамики нормы в зависимости от задаваемого шага дискретизации
узлов сетки.

### Диаграмма классов Class Diagram:
Диаграмма классов, реализованный в рамках данного проекта представлена в файле `Class_Hierarhy.png`

### Список реализованных методов:
Каждая функция снабжена минимально необходимым описанием результата работы и входных
аргуементов. Подробное описание каждой функции и ее параметров, атрибутов и методов 
классов смотреть в `Docstring`. Ниже представлен список методов, реализованных в программе:

* `ode_sol_prob ()` - начальная функция запуска программы
* `dsolve ()` - запуск решателя ОДУ в соответствии с выбранной схемой
* `table_dsol ()` - получение форматированной таблицы результатов моделирования
* `plot_sol ()` - графическая визуализация полученного решения
* `plot_accuracy ()` - графическая визуализация динамики погрешности вычислений
* `norm_accuracy ()` - функция оценки нормы погрешности вычислений
* `err_sch ()` - функция оценки погрешности дискретизации выбранной численной схемы
* `accuracy_approx ()` - функция оценки динамики нормы в зависимости от выбранного шага интегрирования
* `solver_step ()` - имплементация конкретной численной схемы решателя
* `set_initial_condition ()` - функция  инициализации начальных условий задачи Коши
* `solve ()` - решение ОДУ и расчет значений на каждом шаге узла сетки


### Верификация и валидация решения:
Точное аналитическое решение, а также его график представлены в файле-скрипте 
`Analytic_Solve.ipynb`. 

### Подготовка к запуску:
_Запуск и тестирование программы_: Python 3.10, PyCharm 2021.3. 
```
git clone {ссылка на репозиторий}
virtualenv env
pip install -r requirements.txt
python main.py
```

### Примеры описания команд:
```python
# Выбор типа численной схемы аппроксимации результата:
    # 1. Прямой метод Эйлера 2 порядка точности FE (solver_type = 1)
    # 2. Метод Рунге-Кутта 4 порядка (явная схема) RK4 (solver_type = 2)
    # 3. Метод Адамса-Башфорта-Моултона 4 порядка ABM4 (solver_type = 3)
    # 4. Метод Милнса 4 порядка точности  MLN (solver_type = 4)

# формирование начальной задачи Коши u0 - вектор начальных условий, End  - конец интервала
# интегрирования [0; End]
problem = Problem(u0=[0, 3, -9, -8, 0], End=5)

# вызов решателя ОДУ: передаем исходную задачу, шаг интегрирования 
# и тип численной схемы solver_type
solver = Solver(problem, dx=0.05, solver_type=1)

# вызов требуемых функций - функция dsolve() выполняется обязательно
# остальные методы вызываются по мере необходимости
solver.dsolve()

solver.table_dsol()
solver.plot_sol()
solver.plot_accuracy()
solver.norm_accuracy()
solver.err_sch()

# Если необходимо исследовать эффективность схемы в зависимости от выбранного шага интегрирования.
# Раскомментировать, если необходимо
# dxx = [0.005, 0.01, 0.015, 0.02, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25]
# problem1 = Problem(u0=[0, 3, -9, -8, 0], End=5)
# solver1 = Solver(problem1, dxx, solver_type=4)
# solver1.accuracy_approx(dxx)
```

### Дополнительные замечания:
Проект представляет собой первую версию реализации решателя задачи Коши, в дальнейшем
планируется расширить функционал:

- Реализация метода Рунге-Кутта 3 порядка и модфицированного метода Эйлера
- Реализация адаптивного метода Рунге-Кутта-Фельберга 
- Реализация функции контроля точности
- Реализация консольного интерфеса или Telegram-бота
- Реализация функции записи данных расчета в dat/csv файл

_В программной реализации не предусмотрена возможность логгирования._