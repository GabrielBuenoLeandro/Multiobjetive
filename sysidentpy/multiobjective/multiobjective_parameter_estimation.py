import numpy as np
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from itertools import product
#from sysidentpy import basis_function


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
    def __init__(self,
                 sg=True,
                 sf=True,
                 y_static=np.zeros(1),
                 x_static=np.zeros(1),
                 gain=np.zeros(1),
                 y_train=np.zeros(1),
                 psi=np.zeros((1, 1)),
                 n_inputs=1,
                 non_degree=2,
                 model_type='NARMAX',
                 final_model=np.zeros((1, 1)),
                 W=np.zeros((1, 1)),
                 ):
        self.sg = sg
        self.sf = sf
        self.psi = psi
        self.n_inputs = n_inputs
        self.non_degree = non_degree
        self.model_type = model_type
        self.Y_static = y_static
        self.X_static = x_static
        self.final_model = final_model
        self.gain = gain
        self.y_train = y_train
        self.W = W
        #self.basis_function = Polynomial(degree=non_degree)

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
        # 83 to 90 => Construction of the generic qit matrix.
        model = self.final_model//1000
        out = np.max(model)
        N = np.arange(0, out+1)
        b = (product(N, repeat=out))
        possibilities = [] 
        for i in b:
            possibilities.append(i)
        qit = (np.array(possibilities))
        # 92 to 97 => Construction of the generic R matrix.
        R = np.zeros((np.shape(qit)[0], np.shape(model)[0]))
        b = []
        for i in range(0, np.shape(qit)[0]):
            for j in range(0, np.shape(model)[0]):
                if (qit[i, :] == model[j, :]).all():
                    R[i, j] = 1
            if sum(R[i, :]) == 0:
                b.append(i) # Identification of null rows of the R matrix.
        R = np.delete(R, b, axis=0) # Eliminating the null rows from the generic R matrix.
        qit = np.delete(qit, b, axis=0) # Eliminating the null rows from the generic qit matrix.
        return R, qit
               
    def static_function(self):
        """Matrix of static regressors.

        Returns:
        -------
            Q.dot(R) : ndarray of floats
            Returns the multiplication of the matrix of static regressors (Q) and linear mapping (R).
        """
        R, qit = self.R_qit()
        #  115 to 121 => Converting the qit into a matrix of exponents, where the first column indicates the output, 
        # the second column the first input, the third column the second input and so on.
        a = np.shape(qit)[0]
        N_aux = np.zeros((a, int(np.max(qit))))
        for k in range(0, int(np.max(qit))):
            for i in range(0, np.shape(qit)[0]):
                for j in range(0, np.shape(qit)[1]):
                    if k + 1 == qit[i, j]:
                        N_aux[i, k] = 1 + N_aux[i, k]
        qit = N_aux
        # 123 to 129 => Assembly of the matrix Q.
        Q = np.zeros((len(self.Y_static), len(qit)))
        for i in range(0, len(self.Y_static)):
            for j in range(0, len(qit)):
                Q[i, j] = self.Y_static[i, 0]**(qit[j, 0])
                for k in range(0, self.n_inputs):
                    Q[i, j] = Q[i, j]*self.X_static[i, k]**(qit[j, 1+k])
        return Q.dot(R) 

    def static_gain(self):
        """Matrix of static regressors referring to derivative.
        
        Returns:
        --------
        (G+H).dot(R) : ndarray of floats
            Matrix of static regressors for the derivative (gain) multiplied 
            he matrix of the linear mapping R.
        """
        R, qit = self.R_qit()
        # 142 to 157 => Construction of the matrix H and G (Static gain).
        H = np.zeros((len(self.Y_static), len(qit)))
        G = np.zeros((len(self.Y_static), len(qit)))
        for i in range(0, len(self.Y_static)):
            for j in range(1, len(qit)):
                if self.Y_static[i, 0] == 0:
                    H[i, j] = 0
                else:
                    H[i, j] = self.gain[i]*qit[j, 0]*self.Y_static[i, 0]\
                        **(qit[j, 0]-1)
                for k in range(0, self.n_inputs):
                    if self.X_static[i, k] == 0:
                        G[i, j] = 0
                    else:
                        G[i, j] = qit[j, 1+k]*self.X_static[i, k]\
                            **(qit[j, 1+k]-1)
        return (G+H).dot(R)
    
    def weights(self):
        """Weights givenwith each objective.
        
        Returns:
        -------
        w : ndarray of floats
           Matrix with the weights.
        """
        w1 = np.arange(0.01, 1.00, 0.05)
        w2 = np.arange(1.00, 0.01, -0.05)
        a1 = []
        a2 = []
        a3 = []
        for i in range(0, len(w1)):
            for j in range(0, len(w2)):
                if w1[i]+w2[j] <= 1:
                    a1.append(w1[i])
                    a2.append(w2[j])
                    a3.append(1 - (w1[i]+w2[j]))
        if self.sg != False and self.sf != False:
            W = np.zeros((3, len(a1)))
            W[0, :] = a1
            W[1, :] = a2
            W[2, :] = a3
        else:
            W = np.zeros((2, len(a1)))
            W[0, :] = a2
            W[1, :] = np.ones(len(a1))-a2
        return W
    def affine_information_least_squares(self):
        """Calculation of parameters via multi-objective techniques.
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
        """
        # 206 to 210 => Checking if the weights add up to 1.
        if sum(self.W[:, 0]) != 1:
            W = self.weights()
        else:
            W = self.W
        E = np.zeros(np.shape(W)[1])
        Array_theta = np.zeros((np.shape(W)[1], np.shape(self.final_model)[0]))
        #  214 to 241 => Calculation of the Parameters as a result of the input data.
        for i in range(0, np.shape(W)[1]):
            part1 = W[0, i]*(self.psi).T.dot(self.psi)
            part2 = W[0, i]*(self.psi.T).dot(self.y_train)
            w = 1
            if self.sf == True:
                QR = self.static_function()
                part1 = W[w, i]*(QR.T).dot(QR) + part1
                part2 = part2 + (W[w, i]*(QR.T).dot(self.Y_static))\
                    .reshape(-1,1)
                w = w + 1
            if self.sg == True:
                HR = self.static_gain()
                part1 = W[w, i]*(HR.T).dot(HR) + part1
                part2 = part2 + (W[w, i]*(HR.T).dot(self.gain)).reshape(-1,1)
                w = w+1
            if i == 0:
                J = np.zeros((w, np.shape(W)[1]))
            Theta = ((np.linalg.inv(part1)).dot(part2)).reshape(-1, 1)
            Array_theta[i, :] = Theta.T
            J[0, i] = (((self.y_train)-(self.psi.dot(Theta))).T).dot((self.y_train)\
                     -(self.psi.dot(Theta)))
            w = 1
            if self.sg == True:
                J[w, i] = (((self.gain)-(HR.dot(Theta))).T).dot((self.gain)\
                          -(HR.dot(Theta)))
                w = w+1
            if self.sf == True:
                J[w, i] = (((self.Y_static)-(QR.dot(Theta))).T).dot((self.Y_static)-(QR.dot(Theta)))
        for i in range(0, np.shape(W)[1]):
            E[i] = np.linalg.norm(J[:, i]/np.max(J)) # Normalizing quadratic errors.
        return J/np.max(J), W, E, Array_theta, HR, QR