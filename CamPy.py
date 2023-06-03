import openpyxl as xls
import numpy as np
import matplotlib.pyplot as plt

err = 40  
#incrm_base = 0.0001
incrm_p = 0.0 
incrm_q = 0.00003  
fi = 30  
p_prcons = 500  # p'c
M = 6 * np.sin(np.radians(fi)) / (3 - np.sin(np.radians(fi)))  
v0 = 1.55  
lambda_nc = 0.148
kappa_rc = 0.05
v = 0.18

de_tot_p = 0.0
de_tot_q = 0.065
## p' e q iniziale
p_pr = 100
q = 0.0

dp = 0
dq = 0
de_p = 0
de_q = 0

ip = incrm_p
iq = incrm_q

valori_step = []
valori_p = []
valori_q = []
valori_pr = []
valori_de_p = []
valori_de_q = []
valori_p_prcons = []
valori_gamma = []
valori_G = []
valori_K = []
D_p_p = []
valori_H = []
valori_f = []
valori_Kp = []
valori_dp_prcons = []

### ROUTINES
i = 0
while de_q <= de_tot_q:
    valori_step.append(0)
    valori_p.append(0)
    valori_q.append(0)
    valori_pr.append(0)
    valori_de_p.append(0)
    valori_de_q.append(0)
    valori_p_prcons.append(0)
    valori_gamma.append(0)
    valori_G.append(0)
    valori_K.append(0)
    D_p_p.append(0)
    valori_H.append(0)
    valori_f.append(0)
    valori_Kp.append(0)
    valori_dp_prcons.append(0)
    

    plt.plot(valori_de_q, valori_q)
    plt.show()

    valori_step.append(i)
    K = (v0 * p_pr) / kappa_rc
    G = (3 * (1 - 2 * v) * K) / (2 * (1 + v))
    f = (q ** 2 / M ** 2) + (p_pr * (p_pr - p_prcons))
    H = (p_pr * p_prcons * (2 * p_pr - p_prcons) * (v0 / (lambda_nc - kappa_rc)))
    Kp = ((H + (K * (2 * p_pr - p_prcons) ** 2) + (3 * G * ((2 * q) / M ** 2) ** 2)))
    gamma = (((2 * p_pr - p_prcons) * K * ip) + (((2 * q) / (M ** 2)) * 3 * G * iq)) / Kp
    dp_prcons = gamma * (p_prcons * (2 * p_pr - p_prcons) * (v0 / (lambda_nc - kappa_rc)))

    if f < 0:  # elastico
        print(f"\nincremento elastico - step {i}")

        ###p'
        de_p = de_p + ip  # incremento epsilon p
        dp = K * (ip)  # incremento di tensione
        p_pr = p_pr + dp  # aggiornamento p

        ###q
        de_q = de_q + iq  # incremento epsilon q
        dq = 3 * G * (iq)  # incremento di tensione
        q = q + dq  # aggiornamento q

        # aggiornamento dominio elastico
        f = (q ** 2 / M ** 2) + (p_pr * (p_pr - p_prcons))

        
        valori_p[i]=p_pr  # memorizzazione
        valori_de_p[i]=de_p
        
        valori_q[i]=q  # memorizzazione
        valori_de_q[i]=de_q
        
        valori_p_prcons[i]=p_prcons
        
        valori_gamma[i]=gamma
       
        valori_G[i]=G
        valori_K[i]=K
       
        valori_f[i]=f
       
        valori_H[i]=H
        valori_Kp[i]=Kp
        
        valori_dp_prcons[i]=dp_prcons
        incrm_p = incrm_p * 2
        incrm_q = incrm_q * 2
        iq = incrm_q
        ip = incrm_p
        i = i + 1

        print(f"\n step numero {i}")
        print(f"la funzione snervamento è {f}")
        print(f"il valore di H è:{H}")
        print(f"il valore di Kp è {Kp}")
        print(f"la p'c è uguale a {p_prcons}")
        print(f"p' è pari a {p_pr}")
        print(f"q è pari a {q}")
        print(f" gamma è pari a {gamma}")
        print(f"valori di eps q: {de_q}")

    elif f > 0:

        
        K = (v0 * p_pr) / kappa_rc
       
        G = (3 * (1 - 2 * v) * K) / (2 * (1 + v))
        
        f = (q ** 2 / M ** 2) + (p_pr * (p_pr - p_prcons))
        H = (p_pr * p_prcons * (2 * p_pr - p_prcons) * (v0 / (lambda_nc - kappa_rc)))
        Kp = ((H + (K * (2 * p_pr - p_prcons) ** 2) + (3 * G * ((2 * q) / M ** 2) ** 2)))
        gamma = (((2 * p_pr - p_prcons) * K * ip) + (((2 * q) / (M ** 2)) * 3 * G * iq)) / Kp
        dp_prcons = gamma * (p_prcons * (2 * p_pr - p_prcons) * (v0 / (lambda_nc - kappa_rc)))

        print(f"\nincremento plastico - step {i}")
        ## calcolo incrementi tensionali dq e dp
        de_p = de_p + ip
        de_q = de_q + iq
        dpe = K * (ip)
        dqe = 3 * G * (iq)
        dpp = (-((K ** 2 * (2 * p_pr - p_prcons) ** 2) / Kp) * ip) - ((3 * G * K * (2 * p_pr - p_prcons) * ((2 * q) / M ** 2)) / Kp) * iq
        dqq = -(((3 * G * K * (2 * p_pr - p_prcons) * ((2 * q) / M ** 2)) / Kp) * ip) - ((9 * (G ** 2) * ((2 * q) / M ** 2) ** 2) / Kp) * iq
        dp = dpe + dpp
        dq = dqe + dqq
        
        p_pr = p_pr + dp
        # print(f"p' = {p_pr}")
       
        q = q + dq
        # print(f"q = {q}")
       
        K = (v0 * p_pr) / kappa_rc
        # print(f"cost el K fine calc pl: {K}")
       
        G = (3 * (1 - 2 * v) * K) / (2 * (1 + v))
        
       
        dp_prcons = gamma * (p_prcons * (2 * p_pr - p_prcons) * (v0 / (lambda_nc - kappa_rc)))
       
        p_prcons = p_prcons + dp_prcons
        print(f"p'c = {p_prcons}")
       
        gamma = (((2 * p_pr - p_prcons) * K * ip) + (((2 * q) / (M ** 2)) * 3 * G * iq)) / Kp
        print(f"gamma uguale a: {gamma}")
        f = (q ** 2 / M ** 2) + (p_pr * (p_pr - p_prcons))

        if f <= err:

            
            valori_p[i]=p_pr
            valori_de_p[i]=de_p
            valori_q[i]=q
            valori_de_q[i]=de_q
            valori_p_prcons[i]=p_prcons
            valori_gamma[i]=gamma
            valori_G[i]=G
            valori_K[i]=K

            valori_f[i]=f
            valori_H[i]=H
            valori_Kp[i]=Kp
            D_p_p[i]=dpp
            valori_dp_prcons[i]=dp_prcons
            # incrm_p = incrm_p * 1.5
            # incrm_q = incrm_q * 1.5
            # iq = incrm_q
            # ip = incrm_p
            i = i + 1

        elif f > err:
                j = 0

                while valori_f[j] <= err:
                    j = j+1
                
                de_p = valori_de_p[j-1]
                de_q = valori_de_q[j-1]
                p_pr = valori_p[j-1]  
                q = valori_q[j-1] 
                p_prcons = valori_p_prcons[j-1]
                gamma = valori_gamma[j-1]
                H = valori_H[j-1]
                Kp = valori_Kp[j-1]
                i = j -1
               

                f = (q ** 2 / M ** 2) + (p_pr * (p_pr - p_prcons))
                ####
                incrm_p = incrm_p / 100
                incrm_q = incrm_q / 100
                iq = incrm_q
                ip = incrm_p
                #i = i - 1
        print(f"\n step numero {i}")
        print(f"la funzione snervamento è {f}")
        # print(f"il valore di H è:{H}")
        # print(f"il valore di Kp è {Kp}")
        print(f"la p'c è uguale a {p_prcons}")
        print(f"p' è pari a {p_pr}")
        print(f"q è pari a {q}")
        print(f" gamma è pari a {gamma}")
        # print(f"valori di eps q: {de_q}")
        print(f"valori di iq: {iq}")
        print(f"eps s: {de_q}")


figure, axis = plt.subplots(nrows=2, ncols=2)
axis[0, 0].plot(valori_p, valori_q)
axis[0, 0].set_title('piano p-q')


axis[0, 1].plot(valori_de_q, valori_q)
axis[0, 1].set_title('piano q-eps_s')


axis[1, 0].plot(valori_de_q, valori_G)
axis[1, 0].set_title('piano G-eps_s')

axis[1, 1].plot(valori_de_q, valori_K)
axis[1, 1].set_title('piano K-eps_s')
plt.tight_layout()
plt.show()
