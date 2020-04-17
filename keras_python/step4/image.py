import numpy as np


def couper_image(divisionx,divisiony,NBmorceau,tailleX,tailleY,NBbande,img) :
    tab_ech = np.zeros((NBmorceau,tailleX,tailleY,NBbande))
    n_morceau = 0
    for i in range(0 , divisionx):
        for j in range(0 , divisiony):
            tab_ech[n_morceau,:,:,:]=img[i*tailleX:(i+1)*tailleX,j*tailleY:(j+1)*tailleY,0:NBbande]  
            n_morceau=n_morceau+1
    return tab_ech