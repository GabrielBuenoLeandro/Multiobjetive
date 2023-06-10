import numpy as np
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.narmax_base import RegressorDictionary

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
    model_type : string, default='NARMAX'
        Model type.
    final_model : ndarray, default = ([[0],[0]])
        Template code.
    """
    def __init__(self,
                 sg=True,
                 sf=True,
                 model_type='NARMAX',
                 final_model=np.zeros((1, 1)),
                 ):
        self.sg = sg
        self.sf = sf
        self.n_inputs = np.max(final_model//1000)-1
        self.non_degree = np.shape(final_model)[1]
        self.model_type = model_type
        self.final_model = final_model
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
        # 60 to 66 => Construction of the generic qit matrix.
        xlag = []
        for i in range(0, self.n_inputs):
            xlag.append(1)
        object_qit = RegressorDictionary(xlag=xlag, ylag=[1])
        # With xlag and ylag equal to 1 there is no repetition of terms, being ideal for qit assembly.
        qit = object_qit.regressor_space(n_inputs=self.n_inputs)//1000
        model = self.final_model//1000
        # 68 to 73 => Construction of the generic R matrix.
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
        #  97 to 105 => Converting the qit into a matrix of exponents, where the first column indicates the output, 
        # the second column the first input, the third column the second input and so on.
        a = np.shape(qit)[0]
        N_aux = np.zeros((a, int(np.max(qit))))
        for k in range(0, int(np.max(qit))):
            for i in range(0, np.shape(qit)[0]):
                for j in range(0, np.shape(qit)[1]):
                    if k + 1 == qit[i, j]:
                        N_aux[i, k] = 1 + N_aux[i, k]
        qit = N_aux
        # 107 to 112 => Assembly of the matrix Q.
        Q = np.zeros((len(y_static), len(qit)))
        for i in range(0, len(y_static)):
            for j in range(0, len(qit)):
                Q[i, j] = y_static[i, 0]**(qit[j, 0])
                for k in range(0, self.n_inputs):
                    Q[i, j] = Q[i, j]*x_static[i, k]**(qit[j, 1+k])
        return Q.dot(R) 

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
        # 135 to 149 => Construction of the matrix H and G (Static gain).
        H = np.zeros((len(y_static), len(qit)))
        G = np.zeros((len(y_static), len(qit)))
        for i in range(0, len(y_static)):
            for j in range(1, len(qit)):
                if y_static[i, 0] == 0:
                    H[i, j] = 0
                else:
                    H[i, j] = gain[i]*qit[j, 0]*y_static[i, 0]\
                        **(qit[j, 0]-1)
                for k in range(0, self.n_inputs):
                    if x_static[i, k] == 0:
                        G[i, j] = 0
                    else:
                        G[i, j] = qit[j, 1+k]*x_static[i, k]\
                            **(qit[j, 1+k]-1)
        return (G+H).dot(R)
    
    def weights(self):
        """Weights givenwith each objective.
        
        Returns:
        -------
        w : ndarray of floats
           Matrix with the weights.
        """
        w1 = np.logspace(-0.0000001, -5, num=40, base =2.71)
        w2 = w1[::-1]
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
    def affine_information_least_squares(self,  y_static=np.zeros(1),  x_static=np.zeros(1), gain=np.zeros(1), y_train=np.zeros(1),psi=np.zeros((1,1)), W=np.zeros((1,1))):
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
        # 215 to 216 => Checking if the weights add up to 1.
        if np.round(sum(W[:, 0]), 5) != 1:
            W = self.weights()
        E = np.zeros(np.shape(W)[1])
        Array_theta = np.zeros((np.shape(W)[1], np.shape(self.final_model)[0]))
        #  220 to 247 => Calculation of the Parameters as a result of the input data.
        for i in range(0, np.shape(W)[1]):
            theta1 = W[0, i]*(psi).T.dot(psi)
            theta2 = W[0, i]*(psi.T).dot(y_train)
            w = 1
            if self.sf == True:
                if i==0:
                    QR = self.static_function(x_static, y_static)
                    QR_aux1 = (QR.T).dot(QR)
                    QR_aux2 = (QR.T).dot(y_static)
                theta1 = W[w, i]*QR_aux1 + theta1
                theta2 = theta2 + (W[w, i]*QR_aux2)\
                    .reshape(-1,1)
                w = w + 1
            if self.sg == True:
                if i==0:
                    HR = self.static_gain(x_static, y_static, gain)
                    HR_aux1 = (HR.T).dot(HR)
                    HR_aux2 = (HR.T).dot(gain)
                theta1 = W[w, i]*HR_aux1 + theta1
                theta2 = theta2 + (W[w, i]*HR_aux2).reshape(-1,1)
                w = w+1
            if i == 0:
                J = np.zeros((w, np.shape(W)[1]))
            Theta = ((np.linalg.inv(theta1)).dot(theta2)).reshape(-1, 1)
            Array_theta[i, :] = Theta.T
            J[0, i] = (((y_train)-(psi.dot(Theta))).T).dot((y_train)\
                     -(psi.dot(Theta)))
            w = 1
            if self.sg == True:
                J[w, i] = (((gain)-(HR.dot(Theta))).T).dot((gain)\
                          -(HR.dot(Theta)))
                w = w+1
            if self.sf == True:
                J[w, i] = (((y_static)-(QR.dot(Theta))).T).dot((y_static)-(QR.dot(Theta)))
        for i in range(0, np.shape(W)[1]):
            E[i] = np.sqrt(np.sum(J[:,i]**2)) # Normalizing quadratic errors.
        # Finding the smallest squared error in relation to the three objectives.
        min_value = min(E)
        position = (list(E).index(min_value))
        return J/np.max(E), W, E/np.max(E), Array_theta, HR, QR, position