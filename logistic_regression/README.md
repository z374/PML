Ciao!
La regressione logistica, a dispetto del nome, non è un regressore, ma un classificatore. 

Funzionamento:
Sia p la probabilità che un certo sample appartenga ad una certa classe.
è possibile mappare lo spazio di probabilità nell'insieme dei numeri reali considerando il log(p/q). 
A questo punto, considerando l'input netto dato da w_0x_0 + w_1x_1 + ... + w_mx_m, dove w_0 rappresenta la bias unit, è possibile, tramite la corretta funzione di verosimiglianza, determinare i valori ottimali per w_0, w_1, ... w_m.
N.B.: A livello pratico, anziché massimizzare la verosimiglianza, si preferisce minimizzare il suo opposto (così da poter riciclare l'algoritmo del gradiente discendente).

NMB : La base da cui sono partito sono le note di S. Raschka
