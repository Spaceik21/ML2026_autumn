import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        # Постоянный шаг
        if self._method == 'Constant':
            return self.c

        # Сильные условия Вульфа
        if self._method == 'Wolfe':
            result = scipy.optimize.line_search(oracle.func,oracle.grad,x_k,d_k,c1=self.c1,c2=self.c2)
            alpha = result.alpha if hasattr(result, 'alpha') else result[0]

            # Если шаг найден, возвращаем его
            if alpha is not None:
                return alpha

            # Если метод Вульфа не нашел шаг, используем Армихо
            alpha = self.alpha_0

        # Метод Армихо
        elif self._method == 'Armijo':
            alpha = (previous_alpha if previous_alpha is not None else self.alpha_0)

        else:
            raise ValueError('Unknown method {}'.format(self._method))

        # Значение функции при alpha = 0
        phi_0 = oracle.func_directional(x_k, d_k, 0.0)

        # Производная функции по направлению при alpha = 0
        dphi_0 = oracle.grad_directional(x_k, d_k, 0.0)

        # Дробление шага по условию Армихо
        while (oracle.func_directional(x_k, d_k, alpha) > phi_0 + self.c1 * alpha * dphi_0):
            alpha = alpha / 2.0
        return alpha


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict or None
        Dictionary with line search options, e.g. {'method': 'Armijo', 'c1': 1e-4}.
        See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    if line_search_options is None:
        line_search_options = {}
    line_search_tool = LineSearchTool(**line_search_options)
    x_k = np.copy(x_0)

    start_time = datetime.now()
    previous_alpha = None
    initial_grad_norm_sq = None

    for k in range(max_iter + 1):
        # Вычисляем функцию и градиент
        grad = oracle.grad(x_k)
        func_value = oracle.func(x_k)

        # Проверяем ошибки вычислений
        if (grad is None or func_value is None or
            not np.all(np.isfinite(grad)) or
            not np.all(np.isfinite(func_value))):
            return x_k, 'computational_error', history

        grad_norm_sq = np.dot(grad, grad)

        if initial_grad_norm_sq is None:
            initial_grad_norm_sq = grad_norm_sq

        if not np.isfinite(grad_norm_sq):
            return x_k, 'computational_error', history

        # Сохраняем историю
        if trace:
            elapsed = (datetime.now() - start_time).total_seconds()
            history['time'].append(elapsed if k > 0 else 0.0)
            history['func'].append(func_value)
            history['grad_norm'].append(np.sqrt(grad_norm_sq))

            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

        # Вывод информации
        if display:
            print(f'Iteration {k}: f={np.squeeze(func_value)}, '
                  f'grad_norm={np.sqrt(grad_norm_sq)}')

        # Проверяем критерий остановки
        if grad_norm_sq <= tolerance * initial_grad_norm_sq:
            return x_k, 'success', history

        if k == max_iter:
            break

        # Выбираем направление спуска
        d_k = -grad

        # Подбираем длину шага
        alpha_start = None if previous_alpha is None else 2 * previous_alpha
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=alpha_start)

        if alpha is None or not np.all(np.isfinite(alpha)) or alpha <= 0:
            return x_k, 'computational_error', history

        # Переходим в новую точку
        x_k = x_k + alpha * d_k

        if not np.all(np.isfinite(x_k)):
            return x_k, 'computational_error', history

        previous_alpha = alpha
    return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict or None
        Dictionary with line search options, e.g. {'method': 'Armijo', 'c1': 1e-4}.
        See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    if line_search_options is None:
        line_search_options = {}
    line_search_tool = LineSearchTool(**line_search_options)
    x_k = np.copy(x_0)
    start_time = datetime.now()
    initial_grad_norm_sq = None

    for k in range(max_iter + 1):
        # Вычисляем функцию и градиент
        raw_grad = oracle.grad(x_k)
        func_value = oracle.func(x_k)

        # Проверяем ошибки вычислений
        if raw_grad is None or func_value is None:
            return x_k, 'computational_error', history

        grad = np.asarray(raw_grad, dtype=float).reshape(-1)

        if (not np.all(np.isfinite(grad)) or
                not np.all(np.isfinite(func_value))):
            return x_k, 'computational_error', history

        grad_norm_sq = np.dot(grad, grad)

        if not np.isfinite(grad_norm_sq):
            return x_k, 'computational_error', history

        # Запоминаем начальную норму градиента
        if initial_grad_norm_sq is None:
            initial_grad_norm_sq = grad_norm_sq

        # Сохраняем историю
        if trace:
            elapsed = (datetime.now() - start_time).total_seconds()
            history['time'].append(0.0 if k == 0 else elapsed)
            history['func'].append(func_value)
            history['grad_norm'].append(np.sqrt(grad_norm_sq))

            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

        # Выводим информацию об итерации
        if display:
            print(f'Iteration {k}: f={np.squeeze(func_value)}, 'f'grad_norm={np.sqrt(grad_norm_sq)}')

        # Проверяем критерий остановки
        if grad_norm_sq <= tolerance * initial_grad_norm_sq:
            return x_k, 'success', history

        if k == max_iter:
            break

        # Вычисляем гессиан
        raw_hess = oracle.hess(x_k)

        if raw_hess is None:
            return x_k, 'computational_error', history

        # Преобразуем гессиан в обычную двумерную матрицу
        if scipy.sparse.issparse(raw_hess):
            hess = raw_hess.toarray()
        else:
            hess = np.atleast_2d(np.asarray(raw_hess, dtype=float))

        if not np.all(np.isfinite(hess)):
            return x_k, 'computational_error', history

        # Находим направление Ньютона методом Холецкого
        try:
            factor = scipy.linalg.cho_factor(hess, lower=True)
            d_k = scipy.linalg.cho_solve(factor, -grad)

        except (LinAlgError, ValueError):
            return x_k, 'newton_direction_error', history

        if not np.all(np.isfinite(d_k)):
            return x_k, 'computational_error', history

        # Подбираем шаг, начиная с alpha = 1
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1.0)

        if (alpha is None or
                not np.all(np.isfinite(alpha)) or alpha <= 0):
            return x_k, 'computational_error', history

        # Переходим в новую точку
        x_k = x_k + alpha * d_k

        if not np.all(np.isfinite(x_k)):
            return x_k, 'computational_error', history
    return x_k, 'iterations_exceeded', history

