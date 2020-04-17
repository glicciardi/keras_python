import numpy as np

def couper_image(divisionx,divisiony,NBmorceau,tailleX,tailleY,img) :
    tailleimg = img.shape
    tab_ech = np.zeros((tailleX,tailleY,tailleimg[2],NBmorceau))
    morceauact = 0
    for i in range(0 , divisionx):
        for j in range(0 , divisiony):
            tab_ech[:,:,:,morceauact]=img[i*tailleX:(i+1)*tailleX,j*tailleY:(j+1)*tailleY,:]  
            morceauact=morceauact+1
    return tab_ech

def couper_image_decal(divisionx,divisiony,NBmorceau,tailleX,tailleY,img) :
    tailleimg = img.shape
    tab_ech = np.zeros((NBmorceau,tailleX,tailleY,tailleimg[2]))
    morceauact = 0
    for i in range(0 , divisionx):
        for j in range(0 , divisiony):
            tab_ech[morceauact,:,:,:]=img[i*tailleX:(i+1)*tailleX,j*tailleY:(j+1)*tailleY,:]  
            morceauact=morceauact+1
    return tab_ech