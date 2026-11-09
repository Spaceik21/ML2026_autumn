import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Parameters
    ----------
        A : np.array or scipy.sparse.csr_matrix of shape (m, n)
            Feature matrix (training sample).
        b : np.array of shape (m,)
            Labels vector with values in {-1, 1}.
        regcoef : float
            Regularization coefficient.
    """
    def __init__(self, A, b, regcoef):
        self.A = A
        self.b = b
        self.regcoef = regcoef
        self._m = A.shape[0]

    def func(self, x):
        # Вычисляем z = b * (A @ x)
        z = self.b * self.A.dot(x)

        # Средняя логистическая потеря
        loss = np.mean(np.logaddexp(0, -z))

        # L2-регуляризация
        regularization = 0.5 * self.regcoef * np.dot(x, x)
        
        return loss + regularization   

    def grad(self, x):
        # Вычисляем z
        z = self.b * self.A.dot(x)

        # Сигмоида от -z
        q = expit(-z)

        # Градиент логистической потери и регуляризации
        g = -self.A.T.dot(self.b * q) / self._m + self.regcoef * x
        
        return np.asarray(g).reshape(-1)

    def hess(self, x):
        # Вычисляем z и сигмоиду
        z = self.b * self.A.dot(x)
        q = expit(-z)

        # Коэффициенты для диагональной матрицы
        s = q * (1 - q)

        # Вычисляем A.T @ Diag(s) @ A
        if scipy.sparse.issparse(self.A):
            weighted_A = self.A.multiply(s[:, None])
            H = self.A.T.dot(weighted_A).toarray()
        else:
            weighted_A = self.A * s[:, None]
            H = self.A.T.dot(weighted_A)

        # Добавляем регуляризацию
        n = self.A.shape[1]

        return H / self._m + self.regcoef * np.eye(n)


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # Преобразуем x в массив
    x = np.asarray(x, dtype=float)
    n = x.size
    grad = np.zeros(n)

    # Значение функции в исходной точке
    f0 = func(x)

    # Вычисляем каждую компоненту градиента
    for i in range(n):
        x_i = x.copy()
        x_i[i] += eps
        grad[i] = (func(x_i) - f0) / eps

    return grad


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # Преобразуем x в массив
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n))

    # Значение функции в исходной точке
    f0 = func(x)

    # Значения функции при изменении одной координаты
    f_shift = np.zeros(n)

    for i in range(n):
        x_i = x.copy()
        x_i[i] += eps
        f_shift[i] = func(x_i)

    # Вычисляем вторые производные
    for i in range(n):
        for j in range(i, n):
            x_ij = x.copy()
            x_ij[i] += eps
            x_ij[j] += eps
            value = (func(x_ij)- f_shift[i]- f_shift[j]+ f0) / eps**2
            H[i, j] = value
            H[j, i] = value
    return H
