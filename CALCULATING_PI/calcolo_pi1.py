import matplotlib.pyplot as plt
import numpy as np
import math
import random

#--------------------------------------------------------

n_lanci = 5000 # numero di lanci

#--------------------------------------------------------

def f(x): return math.sqrt(1-x**2)

punti=0
xp = []  # ASCISSE CASUALI
yp = []  # ORDINTATE CASUALI - SOTTO IL GRAFICO
xP = []  # ASCISSE CASUALI
yP = []  # ASCISSE CASUALI - SOPRA IL GRAFICO

#--------------------------------------------------------

for p in range(n_lanci):
    x = random.uniform(0,1)   # Estraggo un numero random tra (0,1)
    y = random.uniform(0,1)   # Estraggo un numero random tra (0,1)
    if(x**2+y**2 <= 1):			# Se la distanza dal centro sta dentro la circonferenza
        punti+=1              
        xp.append(x) 
        yp.append(y)
    else:                     	# Se la distanza dal centro sta dentro la circonferenza
        xP.append(x) 
        yP.append(y)

pigreco=4*punti/n_lanci

#--------------------------------------------------------

print('Punti = %d' %n_lanci)
print('punti = %d' %punti)
print('pi = %.3f'  %pigreco)

#--------------------------------------------------------

X = np.linspace(0,1)
yf = []
for x in X:
    yf.append(f(x))
    
#--------------------------------------------------------

plt.axis('equal')
plt.grid(which='major')
plt.plot(X, yf,  color='blue',  linewidth='2')
plt.scatter(xp, yp, color='green', marker ='.')
plt.scatter(xP, yP, color='red',   marker ='.')
plt.plot([], [], color='white', marker='.',linestyle='None', label=r'pi_greco = %.4f' %pigreco)
plt.title('Metodo Monte Carlo: Metodo diretto')
plt.legend()
plt.savefig('Calcolo_pi.pdf', pi = 100)

plt.show()
