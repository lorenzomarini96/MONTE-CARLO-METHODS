import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import random
import math
from scipy.optimize import curve_fit


#====================================================================================================
# 1)  Creo un array x di N componenti sono uguali alla media di 10 numeri distribuiti random uniform.
#====================================================================================================

# Creo un array x di N componenti = la media di 10 numeri distribuiti random uniform.
Npoint = 1000           # Numero di eventi che voglio generare
x = np.zeros(Npoint) # Crea un array di soli zeri di N componenti

ipoint = 0                  # Indica la componente i-esima del vettore x
nd = 10                     # numero di addendi
while ipoint<Npoint -1:     # -1 perché parte da 0
    somma = 0                       # Inizializzo la somma a zero
    array = np.random.random(nd)    # Creo un array di 10 componenti random
    mu = np.mean(array)             # Media dei valori
    std  = np.std(array)            # Deviazione standard
    std  = std/np.sqrt(nd)          # Deviazione standard della media
    zeta = (mu - 1/2)/std           # Variabile standard
    x[ipoint] = zeta
    print('z =',zeta)
    ipoint+=1                       # Incremento di 1 la componente
   

#====================================================================================================
# 2) Stampo i risultati su terminale
#====================================================================================================

# Stampa la x
print('x =',x)

# Calcolo la media dei valori di x
mean = x.mean()
# Calolo la standard deviation dei valori di x
sigma = x.std()

# Stampa i risultati
print('mean =',mean)
print('sigma =',sigma)

#====================================================================================================
# 3) Creo l'istogramma dei valori ottenuti e confronto con i risultati attesi
#====================================================================================================

# COMANDI PER GENERARE ISTOGRAMMA

fig = plt.figure(figsize=(10,5),facecolor='white')
y_data, edges, _ = plt.hist(x, bins=50, range=(-3,3), density=False, facecolor='green', alpha=0.75, edgecolor='black', linewidth=1.2, label='')
# hist restituisce il vettore y_data contenente i counts ed edges contenente i bins. range=(x.min(), x.max()) per non avere il range normalizzato
x_data = 0.5 * (edges[1:] + edges[:-1]) # Baricentri

# FIT GAUSSIANO
def gauss_function(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))

popt, pcov = curve_fit(gauss_function, x_data, y_data)
# Definizione vettore delle x (asse x di estremi minimo e massimo di x_data)
x = np.linspace(min(x_data),max(x_data),100)

# Plot the fit results
plt.plot(x, gauss_function(x, *popt), 'r--', label=r'$p(x)$ = $A$ $e^{-\frac{{(x - \mu)}^2}{(2\sigma)^2}}$')

# Valore medio e deviazione standard da inserire nel titolo
mu    = popt[1]
sigma = popt[2]

plt.title(r"$\mathrm{Histogram}$, N = %.i, $\mathrm{\mu}$ = %.3f, $\mathrm{\sigma}$ = %.3f" %(Npoint, mu, sigma), fontsize = 15)
plt.xlabel(r'x', fontsize = 15)
#plt.xticks(np.arange(-4, 4, 1), fontsize = 10)
plt.ylabel('Probability Density', fontsize = 14)

plt.legend(fancybox=True, shadow=True, prop={"size":15}) # prop={"size":14} cambia dimensioni di legend
plt.grid()

# Salvataggio plot
plt.savefig('FIGURE/somma_n.png', pi=100)

# Show plot
plt.show()

