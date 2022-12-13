import numpy as np
import matplotlib.pyplot as plt
import random
import math

random.seed()
def trayectory(t):
    #return np.array([np.sin(t),np.cos(t)])
    rand = random.uniform(0, 1)
    #return t**2 , 2*t*math.sin(rand) -1.5
    return t**2 , 2*t -1.5


"""
def predict():
    X = np.dot(F,X)
    P = np.dot(dot(F,P),F.T) + Q
    return X

def update(z):
    y = z - np.dot(H, X)        # Error entre medicion y predict (Z es la medida)
    S = self.R + np.dot(H, np.dot(P, H.T))
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    X_k = X + np.dot(K, y)
    #P = (I-KH)P(I-KH)' + KRK'
    I_KH = I - dot(K, H)
    P_k = np.dot(np.dot(I_KH, P), I_KH.T) + np.dot(np.dot(K, R),K.T)
    return X_k,P_k
"""
# Funcion que calcula matriz de ruido de covarianza del proceso , Q
def Q_covarianza_matrix(X_p, X_k_ant, n, sum_mean_Q, sum_Q):
    sum_mean_Q = sum_mean_Q + (X_p - X_k_ant)
    mean_Q = sum_mean_Q / n
    sum_Q = sum_Q + np.dot((X_p - X_k_ant - mean_Q),(X_p - X_k_ant - mean_Q).T)
    Q = sum_Q / (n-1)
    return Q,sum_mean_Q,sum_Q

# Funcion que calcula matriz de covarianza de la medida , R
def R_covarianza_matrix(measure, measure_ant, n, sum_mean_R, sum_R):
    sum_mean_R = sum_mean_R + (measure - measure_ant)
    mean_R = sum_mean_R / n
    sum_R = sum_R + np.dot((measure - measure_ant - mean_R),(measure - measure_ant - mean_R).T)
    R = sum_R / (n-1)
    return R, sum_mean_R, sum_R

# Funcion que añade ruido sobre el ruido en un intervalo
def ruido_intervalo(noise, signal_x, signal_y):
    for j in range(len(noise)):
        signal_x[150+j] = signal_x[150+j] + noise[j]
        signal_y[150+j] = signal_y[150+j] + noise[j]
    return signal_x, signal_y

def rmse(x,y,x_k,y_k):
    sum_x = 0
    sum_y = 0
    for i in range(len(x)):
        sum_x = sum_x + (x[i] - x_k[i])**2
        sum_y = sum_y + (y[i] - y_k[i])**2
    rmse_x = np.sqrt((1/len(x)) * sum_x)
    rmse_y = np.sqrt((1/len(y)) * sum_y)
    return rmse_x,rmse_y




signal = np.linspace(2,15,250)
x_s,y_s = trayectory(signal)

#Complicamos la trayectoria
x_s = np.append(x_s,x_s+x_s[249])
y_s = np.append(y_s,y_s+y_s[249])

# Ruido Blanco a la señal
noise = np.random.normal(0,1,500)
x_n = x_s + noise*2
y_n = y_s + noise*2

# Aumentamos el ruido en mitad de la señal
#noise2 = np.random.normal(0,1,150)
#x_n,y_n = ruido_intervalo(noise2,x_n,y_n)


dim_x = 2
dim_z = 2
X = np.zeros((dim_x, 1))        # State Matrix
P = np.eye(dim_x)               # Procces Covariance Matrix
F = np.eye(dim_x)               # x = Fx + Bu
#H = np.array([[1.,0.], [0.,1.]])
H = np.eye(dim_x)
KG = np.zeros((dim_x, dim_z)) # kalman gain
y = np.zeros((dim_z, 1))
S = np.zeros((dim_z, dim_z)) # system uncertainty
SI = np.zeros((dim_z, dim_z)) # inverse system uncertainty
I = np.eye(dim_x)
#Z = 0.01 * np.ones((dim_x,1))

Q = np.eye(dim_x)             # Noise Covariance Matrix
#R = np.zeros((dim_z,dim_z))
R = np.eye(dim_x)
#R = np.eye(dim_z)
sum_Q = np.zeros(dim_x)
sum_R = np.zeros(dim_x)
sum_mean_Q = np.zeros((dim_x,1))
sum_mean_R = np.zeros((dim_z,1))

x_kalman = []
y_kalman = []
for i in range(len(x_s)):

    measure = np.array([[x_n[i]],[y_n[i]]]) #+ Z    # Tomamos la medida

    #Prediccion X
    X = np.dot(F,X)

    if (i >= 2):
        Q,sum_mean_Q,sum_Q = Q_covarianza_matrix(X, X_p_ant, i, sum_mean_Q, sum_Q)
        R,sum_mean_R,sum_R = R_covarianza_matrix(measure, measure_ant, i, sum_mean_R, sum_R)


    #Prediccion P
    P = np.dot(np.dot(F,P),F.T) + Q

    y = np.dot(H,measure)
    y_error = y - np.dot(H, X)        # Error entre medicion y predict (Z es la medida)
    S = np.dot(H, np.dot(P, H.T)) + R

    KG = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    I_KH = I - np.dot(KG, H)
    #X_k = X_Kp + KG*Y
    X_k = X + np.dot(KG, y_error)
        #P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.
    P_k = np.dot(np.dot(I_KH, P), I_KH.T) + np.dot(np.dot(KG, R),KG.T)
    #P_k = np.dot(I_KH,P)

    #Guardamos la solucion
    x_kalman.append(X_k[0])
    y_kalman.append(X_k[1])
    #Guardamos para culacular matrix Q y R
    X_p_ant = X
    measure_ant = measure
    X = X_k
    P = P_k

rmse_x,rmse_y = rmse(x_s,y_s,x_kalman,y_kalman)
print("RMSE x fuction: ", rmse_x)
print("RMSE y fuction: ", rmse_y)

fig, ax = plt.subplots()
ax.plot(x_n, y_n, color ='r' ,label = 'Trayectory with Noise')
ax.plot(x_kalman, y_kalman, '--g', linewidth = 2, label='Kalman Filter')
ax.plot(x_s, y_s, color='b', label = 'Trayectory')
#ax.vlines([150, 300], 0, 1, transform=ax.get_xaxis_transform(), colors='y')
plt.legend()
plt.show()
