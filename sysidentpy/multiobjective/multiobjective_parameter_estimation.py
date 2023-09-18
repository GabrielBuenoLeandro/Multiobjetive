""" Affine Information Least Squares for NARMAX models"""

import numpy as np
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.narmax_base import RegressorDictionary


class AILS:
    """Affine Information Least Squares (AILS) for NARMAX Parameter Estimation.

    AILS is a non-iterative multiobjective Least Squares technique used for finding
    Pareto-set solutions in NARMAX (Nonlinear AutoRegressive Moving Average with
    eXogenous inputs) model parameter estimation. This method is suitable for
    linear-in-the-parameter model structures.

    Two types of auxiliary information can be incorporated: static function and
    steady-state gain.

    Parameters
    ----------
    static_gain : bool, default=True
        Flag indicating the presence of data related to steady-state gain.
    static_function : bool, default=True
        Flag indicating the presence of data concerning static function.
    final_model : ndarray, default=[[0], [0]]
        Model code representation.

    References
    ----------
    1. Nepomuceno, E. G., Takahashi, R. H. C., & Aguirre, L. A. (2007).
    "Multiobjective parameter estimation for nonlinear systems: Affine information and
    least-squares formulation."
    International Journal of Control, 80, 863-871.
    """

    def __init__(
        self,
        static_gain=True,
        static_function=True,
        final_model=np.zeros((1, 1)),
        normalize=True,
    ):
        self.n_inputs = np.max(final_model // 1000) - 1
        self.degree = np.shape(final_model)[1]
        self.final_model = final_model
        self.static_gain = static_gain
        self.static_function = static_function
        self.normalize = normalize

    def get_term_clustering(self, qit):
        """
        Get the term clustering of the model.

        This function takes a matrix `qit` and compute the term clustering based
        on their values. It calculates the number of occurrences of each value
        for each row in the matrix.

        Parameters
        ----------
        qit : ndarray
            Input matrix containing terms clustering to be sorted.

        Returns
        -------
        N_aux : ndarray
            A new matrix with rows representing the number of occurrences of each value
            for each row in the input matrix `qit`. The columns correspond to different
            values.

        Examples
        --------
        >>> qit = np.array([[1, 2, 2],
        ...                 [1, 3, 1],
        ...                 [2, 2, 3]])
        >>> sorter = Sorter()
        >>> result = sorter.get_term_clustering(qit)
        >>> print(result)
        [[1. 2. 0. 0.]
        [2. 0. 1. 0.]
        [0. 2. 1. 0.]]

        Notes
        -----
        The function calculates the number of occurrences of each value (from 1 to
        the maximum value in the input matrix `qit`) for each row and returns a matrix
        where rows represent rows of the input matrix `qit`, and columns represent
        different values.

        """
        max_value = int(np.max(qit))
        counts_matrix = np.zeros((qit.shape[0], max_value))

        for k in range(1, max_value + 1):
            counts_matrix[:, k - 1] = np.sum(qit == k, axis=1)

        return counts_matrix

    def build_linear_mapping(self):
        """
        Assemble the linear mapping matrix R using the regressor-space method.

        This function constructs the linear mapping matrix R, which plays a key role in
        mapping the parameter vector to the cluster coefficients. It also generates a
        row matrix qit that assists in locating terms within the linear mapping matrix.
        This qit matrix is later used in creating the static regressor matrix (Q).

        Returns
        -------
        R : ndarray of int
            A constant matrix of ones and zeros that maps the parameter vector to
            cluster coefficients.
        qit : ndarray of int
            A row matrix that helps locate terms within the linear mapping matrix R and
            is used in the creation of the static regressor matrix (Q).

        Notes
        -----
        The linear mapping matrix R is constructed using the regressor-space method.
        It plays a crucial role in the parameter estimation process, facilitating the
        mapping of parameter values to cluster coefficients. The qit matrix aids in
        term localization within the linear mapping matrix R and is subsequently used
        to build the static regressor matrix (Q).

        """
        xlag = [1] * self.n_inputs

        object_qit = RegressorDictionary(xlag=xlag, ylag=[1])
        # Given xlag and ylag equal to 1, there is no repetition of terms, which is
        # ideal for building qit.
        qit = object_qit.regressor_space(n_inputs=self.n_inputs) // 1000
        model = self.final_model // 1000
        R = np.all(qit[:, None, :] == model, axis=2).astype(int)
        # Find rows with all zeros in R (sum of row elements is 0)
        null_rows = list(np.where(np.sum(R, axis=1) == 0)[0])

        R = np.delete(R, null_rows, axis=0)
        qit = np.delete(qit, null_rows, axis=0)
        return R, self.get_term_clustering(qit)

    def build_static_function_information(self, x_static, y_static):
        """
        Construct a matrix of static regressors for a NARMAX model.

        Parameters
        ----------
        y_static : array-like, shape (n_samples_static_function,)
            Output of the static function.
        x_static : array-like, shape (n_samples_static_function,)
            Static function input.

        Returns
        -------
        Q_dot_R : ndarray of floats, shape (n_samples_static_function, n_parameters)
            The result of multiplying the matrix of static regressors (Q) with the
            linear mapping matrix (R), where n_parameters is the number of model
            parameters.
        static_covariance: ndarray of floats, shape (n_parameters, n_parameters)
            The covariance QR'QR
        static_response: ndarray of floats, shape (n_parameters,)
            The response QR'y

        Notes
        -----
        This function constructs a matrix of static regressors (Q) based on the provided
        static function outputs (y_static) and inputs (x_static). The linear mapping
        matrix (R) should be precomputed before calling this function. The result
        Q_dot_R represents the static regressors for the NARMAX model.

        """
        R, qit = self.build_linear_mapping()
        Q = y_static ** qit[:, 0]
        for k in range(self.n_inputs):
            Q *= x_static ** qit[:, 1 + k]

        Q = Q.reshape(len(y_static), len(qit))

        QR = Q.dot(R)
        static_covariance = (QR.T).dot(QR)
        static_response = (QR.T).dot(y_static)
        return QR, static_covariance, static_response

    def build_static_gain_information(self, x_static, y_static, gain):
        """
        Construct a matrix of static regressors referring to the derivative (gain).

        Parameters
        ----------
        y_static : array-like, shape (n_samples_static_function,)
            Output of the static function.
        x_static : array-like, shape (n_samples_static_function,)
            Static function input.
        gain : array-like, shape (n_samples_static_gain,)
            Static gain input.

        Returns
        -------
        HR : ndarray of floats, shape (n_samples_static_function, n_parameters)
            The matrix of static regressors for the derivative (gain) multiplied by the
            linear mapping matrix R.
        gain_covariance : ndarray of floats, shape (n_parameters, n_parameters)
            The covariance matrix (HR'HR) for the gain-related regressors.
        gain_response : ndarray of floats, shape (n_parameters,)
            The response vector (HR'y) for the gain-related regressors.

        Notes
        -----
        This function constructs a matrix of static regressors (G+H) for the derivative
        (gain) based on the provided static function outputs (y_static), inputs
        (x_static), and gain values. The linear mapping matrix (R) should be
        precomputed before calling this function.

        """
        R, qit = self.build_linear_mapping()
        H = np.zeros((len(y_static), len(qit)))
        G = np.zeros((len(y_static), len(qit)))
        for i in range(0, len(y_static)):
            for j in range(1, len(qit)):
                if y_static[i, 0] == 0:
                    if (qit[j, 0]) == 1:
                        H[i, j] = gain[i]
                    else:
                        H[i, j] = 0
                else:
                    H[i, j] = gain[i] * qit[j, 0] * y_static[i, 0] ** (qit[j, 0] - 1)
                for k in range(0, self.n_inputs):
                    if x_static[i, k] == 0:
                        if (qit[j, 1 + k]) == 1:
                            G[i, j] = 1
                        else:
                            G[i, j] = 0
                    else:
                        G[i, j] = qit[j, 1 + k] * x_static[i, k] ** (qit[j, 1 + k] - 1)

        HR = (G + H).dot(R)
        gain_covariance = (HR.T).dot(HR)
        gain_response = (HR.T).dot(gain)
        return HR, gain_covariance, gain_response

    def set_weights(self):
        """
        Set weights assigned to each objective in the multi-objective optimization.

        Returns
        -------
        weights : ndarray of floats
            An array containing the weights for each objective.

        Notes
        -----
        This method calculates the weights to be assigned to different objectives in
        multi-objective optimization. The choice of weights depends on the presence
        of static function and static gain data. If both are present, a set of weights
        for dynamic, gain, and static objectives is computed. If either static function
        or static gain is absent, a simplified set of weights is generated.

        """
        w1 = np.logspace(-0.01, -5, num=50, base=2.71)
        if self.static_function is False or self.static_gain is False:
            w2 = 1 - w1
            return np.vstack([w1, w2])

        w2 = w1[::-1]
        w1_grid, w2_grid = np.meshgrid(w1, w2)
        w3_grid = 1 - (w1_grid + w2_grid)
        mask = w1_grid + w2_grid <= 1
        dynamic_weight = np.flip(w1_grid[mask])
        gain_weight = np.flip(w2_grid[mask])
        static_weight = np.flip(w3_grid[mask])
        return np.vstack([dynamic_weight, gain_weight, static_weight])
    
    def get_cost_function(self, y, psi, theta):
        """
        Calculate the cost function based on residuals.

        Parameters
        ----------
        y : ndarray of floats
            The target data used in the identification process.
        psi : ndarray of floats, shape (n_samples, n_parameters)
            The matrix of regressors.
        theta : ndarray of floats
            The parameter vector.

        Returns
        -------
        cost_function : float
            The calculated cost function value.

        Notes
        -----
        This method computes the cost function value based on the residuals between
        the target data (y) and the predicted values using the regressors (dynamic
        and static) and parameter vector (theta). It quantifies the error in the
        model's predictions.

        """
        residuals = y - psi.dot(theta)
        return residuals.T.dot(residuals)
    
    def build_system_data(self, y, static_gain, static_function):
        """
        Construct a list of output data components for the NARMAX system.

        Parameters
        ----------
        y : ndarray of floats
            The target data used in the identification process.
        static_gain : ndarray of floats
            Static gain output data.
        static_function : ndarray of floats
            Static function output data.

        Returns
        -------
        system_data : list of ndarrays
            A list containing data components, including the target data (y),
            static gain data (if present), and static function data (if present).

        Notes
        -----
        This method constructs a list of data components that are used in the NARMAX
        system identification process. The components may include the target data (y),
        static gain data (if enabled), and static function data (if enabled).

        """
        if not self.static_gain:
            return [y] + [static_function]

        if not self.static_function:
            return [y] + [static_gain]

        return [y] + [static_gain] + [static_function]


class IM(FROLS):
    """Multiobjective parameter estimation using technique proposed by Nepomuceno et. al.

    Reference:
    NEPOMUCENO, E. G.; TAKAHASHI, R. H. C. ; AGUIRRE, L. A. . Multiobjective parameter
    estimation for nonlinear systems: Affine information and least-squares formulation.
    International Journal of Control (Print), v. 80, p. 863-871, 2007.

    Parameters
    ----------
    sg : bool, default=True
        Presence of data referring to static gain.
    sf : bool, default=True
        Presence of data regarding static function.
    model_type : string, default='NARMAX'
        Model type.
    final_model : ndarray, default = ([[0],[0]])
        Template code.
    """

    def __init__(
        self,
        sg=True,
        sf=True,
        model_type="NARMAX",
        final_model=np.zeros((1, 1)),
        norm=True,
    ):
        self.sg = sg
        self.sf = sf
        self.n_inputs = np.max(final_model // 1000) - 1
        self.non_degree = np.shape(final_model)[1]
        self.model_type = model_type
        self.final_model = final_model
        self.norm = norm
        # self.basis_function = Polynomial(degree=non_degree)

    def Exp(self, qit):
        N_aux = np.zeros((int(np.shape(qit)[0]), int(np.max(qit))))
        for k in range(0, int(np.max(qit))):
            for i in range(0, np.shape(qit)[0]):
                for j in range(0, np.shape(qit)[1]):
                    if k + 1 == qit[i, j]:
                        N_aux[i, k] = 1 + N_aux[i, k]
        return N_aux

    def R_qit(self):
        """Assembly of the matrix of the linear mapping R, where to locate the terms uses the regressor-space method

        Returns:
        --------
            R : ndarray of int
            Matrix of the linear mapping composed by zeros and ones.
            qit : ndarray of int
            Row matrix that helps in locating the terms of the linear mapping matrix
            and will later be used in the making of the static regressor matrix (Q).
        """
        # 65 to 68 => Construction of the generic qit matrix.
        xlag = []
        for i in range(0, self.n_inputs):
            xlag.append(1)
        object_qit = RegressorDictionary(xlag=xlag, ylag=[1])
        # With xlag and ylag equal to 1 there is no repetition of terms, being ideal for qit assembly.
        qit = object_qit.regressor_space(n_inputs=self.n_inputs) // 1000
        model = self.final_model // 1000
        # 73 to 78 => Construction of the generic R matrix.
        R = np.zeros((np.shape(qit)[0], np.shape(model)[0]))
        b = []
        for i in range(0, np.shape(qit)[0]):
            for j in range(0, np.shape(model)[0]):
                if (qit[i, :] == model[j, :]).all():
                    R[i, j] = 1
            if sum(R[i, :]) == 0:
                b.append(i)  # Identification of null rows of the R matrix.
        R = np.delete(
            R, b, axis=0
        )  # Eliminating the null rows from the generic R matrix.
        qit = np.delete(
            qit, b, axis=0
        )  # Eliminating the null rows from the generic qit matrix.
        return R, self.Exp(qit)

    def static_function(self, x_static, y_static):
        """Matrix of static regressors.

        Parameters
        ----------
        y_static : array-like of shape = n_samples_static_function, default = ([0])
            Output of static function.
        x_static : array-like of shape = n_samples_static_function, default = ([0])
            Static function input.

        Returns:
        -------
            Q.dot(R) : ndarray of floats
            Returns the multiplication of the matrix of static regressors (Q) and linear mapping (R).
        """
        R, qit = self.R_qit()
        # 102 to 107 => Assembly of the matrix Q.
        Q = np.zeros((len(y_static), len(qit)))
        for i in range(0, len(y_static)):
            for j in range(0, len(qit)):
                Q[i, j] = y_static[i, 0] ** (qit[j, 0])
                for k in range(0, self.n_inputs):
                    Q[i, j] = Q[i, j] * x_static[i, k] ** (qit[j, 1 + k])
        return Q.dot(R), Q

    def static_gain(self, x_static, y_static, gain):
        """Matrix of static regressors referring to derivative.

        Parameters
        ----------
        y_static : array-like of shape = n_samples_static_function, default = ([0])
            Output of static function.
        x_static : array-like of shape = n_samples_static_function, default = ([0])
            Static function input.
        gain : array-like of shape = n_samples_static_gain, default = ([0])
            Static gain input.

        Returns:
        --------
        (G+H).dot(R) : ndarray of floats
            Matrix of static regressors for the derivative (gain) multiplied
            he matrix of the linear mapping R.
        """
        R, qit = self.R_qit()
        # 130 to 151 => Construction of the matrix H and G (Static gain).
        H = np.zeros((len(y_static), len(qit)))
        G = np.zeros((len(y_static), len(qit)))
        for i in range(0, len(y_static)):
            for j in range(1, len(qit)):
                if y_static[i, 0] == 0:
                    if (qit[j, k]) == 1:
                        H[i, j] = gain[i]
                    else:
                        H[i, j] = 0
                else:
                    H[i, j] = gain[i] * qit[j, 0] * y_static[i, 0] ** (qit[j, 0] - 1)
                for k in range(0, self.n_inputs):
                    if x_static[i, k] == 0:
                        if (qit[j, 1 + k]) == 1:
                            G[i, j] = 1
                        else:
                            G[i, j] = 0
                    else:
                        G[i, j] = qit[j, 1 + k] * x_static[i, k] ** (qit[j, 1 + k] - 1)
        return (G + H).dot(R), H + G

    def weights(self):
        """Weights givenwith each objective.

        Returns:
        -------
        w : ndarray of floats
           Matrix with the weights.
        """
        w1 = np.logspace(-0.01, -5, num=50, base=2.71)
        w2 = w1[::-1]
        a1 = []
        a2 = []
        a3 = []
        for i in range(0, len(w1)):
            for j in range(0, len(w2)):
                if w1[i] + w2[j] <= 1:
                    a1.append(w1[i])
                    a2.append(w2[j])
                    a3.append(1 - (w1[i] + w2[j]))
        if self.sg != False and self.sf != False:
            W = np.zeros((3, len(a1)))
            W[0, :] = a1
            W[1, :] = a2
            W[2, :] = a3
        else:
            W = np.zeros((2, len(a1)))
            W[0, :] = a2
            W[1, :] = np.ones(len(a1)) - a2
        return W

    def affine_information_least_squares(
        self,
        y_static=np.zeros(1),
        x_static=np.zeros(1),
        gain=np.zeros(1),
        y_train=np.zeros(1),
        psi=np.zeros((1, 1)),
        W=np.zeros((1, 1)),
    ):
        """Calculation of parameters via multi-objective techniques.

        Parameters
        ----------
        y_static : array-like of shape = n_samples_static_function, default = ([0])
            Output of static function.
        x_static : array-like of shape = n_samples_static_function, default = ([0])
            Static function input.
        gain : array-like of shape = n_samples_static_gain, default = ([0])
            Static gain input.
        y_train : array-like of shape = n_samples, defalult = ([0])
            The target data used in the identification process.
        psi : ndarray of floats, default = ([[0],[0]])
            Matrix of static regressors.

        Returns
        -------
        J : ndarray
            Matrix referring to the objectives.
        W : ndarray
            Matrix referring to weights.
        E : ndarray
            Matrix of the Euclidean norm.
        Array_theta : ndarray
            Matrix with parameters for each weight.
        HR : ndarray
            H matrix multiplied by R.
        QR : ndarray
            Q matrix multiplied by R.
        w : ndarray, default = ([[0],[0]])
            Matrix with weights.
        """
        HR = None
        QR = None
        # 217 to 218 => Checking if the weights add up to 1.
        if np.round(sum(W[:, 0]), 5) != 1:
            W = self.weights()
        E = np.zeros(np.shape(W)[1])
        Array_theta = np.zeros((np.shape(W)[1], np.shape(self.final_model)[0]))
        #  222 to 257 => Calculation of the Parameters as a result of the input data.
        PSI_aux1 = (psi).T.dot(psi)
        PSI_aux2 = (psi.T).dot(y_train)
        for i in range(0, np.shape(W)[1]):
            theta1 = W[0, i] * PSI_aux1
            theta2 = W[0, i] * PSI_aux2
            w = 1
            if self.sf == True:
                if i == 0:
                    QR = self.static_function(x_static, y_static)[0]
                    QR_aux1 = (QR.T).dot(QR)
                    QR_aux2 = (QR.T).dot(y_static)
                theta1 = W[w, i] * QR_aux1 + theta1
                theta2 = theta2 + (W[w, i] * QR_aux2).reshape(-1, 1)
                w = w + 1
            if self.sg == True:
                if i == 0:
                    HR = self.static_gain(x_static, y_static, gain)[0]
                    HR_aux1 = (HR.T).dot(HR)
                    HR_aux2 = (HR.T).dot(gain)
                theta1 = W[w, i] * HR_aux1 + theta1
                theta2 = theta2 + (W[w, i] * HR_aux2).reshape(-1, 1)
                w = w + 1
            if i == 0:
                J = np.zeros((w, np.shape(W)[1]))
            Theta = ((np.linalg.inv(theta1)).dot(theta2)).reshape(-1, 1)
            Array_theta[i, :] = Theta.T
            J[0, i] = (((y_train) - (psi.dot(Theta))).T).dot(
                (y_train) - (psi.dot(Theta))
            )
            w = 1
            if self.sg == True:
                J[w, i] = (((gain) - (HR.dot(Theta))).T).dot((gain) - (HR.dot(Theta)))
                w = w + 1
            if self.sf == True:
                J[w, i] = (((y_static) - (QR.dot(Theta))).T).dot(
                    (y_static) - (QR.dot(Theta))
                )
        for i in range(0, np.shape(W)[1]):
            E[i] = np.sqrt(np.sum(J[:, i] ** 2))  # Normalizing quadratic errors.
        if self.norm == True:
            for i in range(0, np.shape(J)[0]):
                J[i, :] = J[i, :] / np.max(J[i, :])
            E = E / np.max(E)
        # Finding the smallest squared error in relation to the three objectives.
        min_value = min(E)
        position = list(E).index(min_value)
        return J, W, E, Array_theta, HR, QR, position
