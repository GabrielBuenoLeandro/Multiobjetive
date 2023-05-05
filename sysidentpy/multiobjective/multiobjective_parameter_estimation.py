import numpy as np
from sysidentpy.model_structure_selection import FROLS


class IM(FROLS):
    """Multiobjective parameter estimation using technique proposed by Nepomuceno et. al.

    Reference:
    NEPOMUCENO, E. G.; TAKAHASHI, R. H. C. ; AGUIRRE, L. A. . Multiobjective parameter
    estimation for nonlinear systems: Affine information and least-squares formulation.
    International Journal of Control (Print), v. 80, p. 863-871, 2007.

    Parameters
    ----------
    static_gain : bool, default=True
        Presence of data referring to static gain.
    static_function : bool, default=True
        Presence of data regarding static function.
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
    n_inputs : int, default=1
        Number of entries.
    non_degree : int, default=2
        Degree of nonlinearity.
    model_type : string, default='NARMAX'
        Model type.
    final_model : ndarray, default = ([[0],[0]])
        Template code.
    w : ndarray, default = ([[0],[0]])
        Matrix with weights.
    """

    def __init__(
        self,
        static_gain=True,
        static_function=True,
        y_static=np.zeros(1),
        x_static=np.zeros(1),
        gain=np.zeros(1),
        y_train=np.zeros(1),
        psi=np.zeros((1, 1)),
        n_inputs=1,
        non_degree=2,
        model_type="NARMAX",
        final_model=np.zeros((1, 1)),
        w=np.zeros((1, 1)),
    ):
        self.static_gain = static_gain
        self.static_function = static_function
        self.psi = psi
        self._n_inputs = n_inputs
        self.non_degree = non_degree
        self.model_type = model_type
        self.y_static = y_static
        self.x_static = x_static
        self.final_model = final_model
        self.gain = gain
        self.y_train = y_train
        self.w = w

    def r_qit(self):
        """Assembly of the R matrix and ordering of the q_it."""
        N = []
        for i in range(0, self._n_inputs):
            N.append(1)
        qit = (
            self.regressor_space(self.non_degree, N, 1, self._n_inputs, self.model_type)
            // 1000
        )
        # (41, 50) -> Gerando a matriz de mapeamento linear
        model = self.final_model // 1000
        R = np.zeros((np.shape(qit)[0], np.shape(model)[0]))
        b = []
        for i in range(0, np.shape(qit)[0]):
            for j in range(0, np.shape(model)[0]):
                if (qit[i, :] == model[j, :]).all():
                    R[i, j] = 1
            if sum(R[i, :]) == 0:
                b.append(i)
        R = np.delete(R, b, axis=0)
        # (52, 67) -> é montada a matriz dos regressores estáticos
        qit = np.delete(qit, b, axis=0)
        return R, qit

    def static_function(self):
        """Matrix of static regressors."""
        R, qit = self.r_qit()
        a = np.shape(qit)[0]
        N_aux = np.zeros((a, int(np.max(qit))))
        for k in range(0, int(np.max(qit))):
            for i in range(0, np.shape(qit)[0]):
                for j in range(0, np.shape(qit)[1]):
                    if k + 1 == qit[i, j]:
                        N_aux[i, k] = 1 + N_aux[i, k]
        qit = N_aux
        Q = np.zeros((len(self.y_static), len(qit)))
        for i in range(0, len(self.y_static)):
            for j in range(0, len(qit)):
                Q[i, j] = self.y_static[i, 0] ** (qit[j, 0])
                for k in range(0, self._n_inputs):
                    Q[i, j] = Q[i, j] * self.x_static[i, k] ** (qit[j, 1 + k])
        return Q.dot(R)

    def static_gain(self):
        """Matrix of static regressors referring to derivative."""
        R, qit = self.r_qit()
        H = np.zeros((len(self.y_static), len(qit)))
        G = np.zeros((len(self.y_static), len(qit)))
        for i in range(0, len(self.y_static)):
            for j in range(1, len(qit)):
                if self.y_static[i, 0] == 0:
                    H[i, j] = 0
                else:
                    H[i, j] = (
                        self.gain[i]
                        * qit[j, 0]
                        * self.y_static[i, 0] ** (qit[j, 0] - 1)
                    )
                for k in range(0, self._n_inputs):
                    if self.x_static[i, k] == 0:
                        G[i, j] = 0
                    else:
                        G[i, j] = qit[j, 1 + k] * self.x_static[i, k] ** (
                            qit[j, 1 + k] - 1
                        )
        return (G + H).dot(R)

    def weights(self):
        """Weights givenwith each goal."""
        w1 = np.arange(0.01, 1.00, 0.05)
        w2 = np.arange(1.00, 0.01, -0.05)
        a1 = []
        a2 = []
        a3 = []
        for i in range(0, len(w1)):
            for j in range(0, len(w2)):
                if w1[i] + w2[j] <= 1:
                    a1.append(w1[i])
                    a2.append(w2[j])
                    a3.append(1 - (w1[i] + w2[j]))
        if self.static_gain != False and self.static_function != False:
            w = np.zeros((3, len(a1)))
            w[0, :] = a1
            w[1, :] = a2
            w[2, :] = a3
        else:
            w = np.zeros((2, len(a1)))
            w[0, :] = a2
            w[1, :] = np.ones(len(a1)) - a2
        return w

    def multio(self):
        """Calculation of parameters via multi-objective techniques.
        Returns
        -------
        J : ndarray
            Matrix referring to the objectives.
        w : ndarray
            Matrix referring to weights.
        E : ndarray
            Matrix of the Euclidean norm.
        Array_theta : ndarray
            Matrix with parameters for each weight.
        HR : ndarray
            H matrix multiplied by R.
        QR : ndarray
            Q matrix multiplied by R.
        """
        if sum(self.w[:, 0]) != 1:
            w = self.weights()
        else:
            w = self.w
        E = np.zeros(np.shape(w)[1])
        Array_theta = np.zeros((np.shape(w)[1], np.shape(self.final_model)[0]))
        for i in range(0, np.shape(w)[1]):
            part1 = w[0, i] * (self.psi).T.dot(self.psi)
            part2 = w[0, i] * (self.psi.T).dot(self.y_train)
            w = 1
            if self.static_function == True:
                QR = self.static_function()
                part1 = w[w, i] * (QR.T).dot(QR) + part1
                part2 = part2 + (w[w, i] * (QR.T).dot(self.y_static)).reshape(-1, 1)
                w = w + 1
            if self.static_function == True:
                HR = self.static_gain()
                part1 = w[w, i] * (HR.T).dot(HR) + part1
                part2 = part2 + (w[w, i] * (HR.T).dot(self.gain)).reshape(-1, 1)
                w = w + 1
            if i == 0:
                J = np.zeros((w, np.shape(w)[1]))
            Theta = ((np.linalg.inv(part1)).dot(part2)).reshape(-1, 1)
            Array_theta[i, :] = Theta.T
            J[0, i] = (((self.y_train) - (self.psi.dot(Theta))).T).dot(
                (self.y_train) - (self.psi.dot(Theta))
            )
            w = 1
            if self.static_gain == True:
                J[w, i] = (((self.gain) - (HR.dot(Theta))).T).dot(
                    (self.gain) - (HR.dot(Theta))
                )
                w = w + 1
            if self.static_function == True:
                J[w, i] = (((self.y_static) - (QR.dot(Theta))).T).dot(
                    (self.y_static) - (QR.dot(Theta))
                )
        for i in range(0, np.shape(w)[1]):
            E[i] = np.linalg.norm(J[:, i] / np.max(J))
        return J / np.max(J), w, E, Array_theta, HR, QR
