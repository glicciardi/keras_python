#autoencoders
#re-fabrication d'un NLPCA pour images HP
get_ipython().magic('reset -sf')

#importer les bibliothèques
import image

import numpy as np
from skimage import io
import matplotlib.pyplot as plt


from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Reshape

#parametre de l'image
divisionx = 16
divisiony = 16
NBmorceau = divisionx*divisiony


#lecture des images
NBimage = 2;

img1 = io.imread('modisGrenoble.tif')
plt.imshow(img1[:,:,1])
plt.show()

img2 = io.imread('modisGrenoble2.tif')
plt.imshow(img2[:,:,1])
plt.show()


#decoupage des images

tailleimg = img2.shape
tailleX = int(tailleimg[0]/divisionx)
tailleY = int(tailleimg[1]/divisiony)
NBbande = 1#tailleimg[2]
taillemorceau = tailleX*tailleY*NBbande
NBpixelTotal = NBimage*NBmorceau*tailleX*tailleY*NBbande #pixel utilisée

echantillon = np.zeros((NBimage*NBmorceau,tailleX,tailleY,NBbande))

#vecteur image couper

image1Coupe = image.couper_image(divisionx,divisiony,NBmorceau,tailleX,tailleY,NBbande,img1)
image2Coupe = image.couper_image(divisionx,divisiony,NBmorceau,tailleX,tailleY,NBbande,img2)

echantillon[0:NBmorceau] = image1Coupe
echantillon[NBmorceau:NBimage*NBmorceau] = image2Coupe

# vecteur image en ligne
image1CoupeLigne = np.reshape(image1Coupe,(NBmorceau,taillemorceau),0)
image2CoupeLigne = np.reshape(image2Coupe,(NBmorceau,taillemorceau),0)

#parametre du reseau
Layer1=taillemorceau
Layer2=int(taillemorceau)
Layer3=int(taillemorceau)
NB_dim_reduit = Layer2
print('NBnode : \nlayer1 ',Layer1,'\nlayer2 ',Layer2,'\nlayer3 ',Layer3,'\n')

#fabrication du reseau de neurone
input_img = Input(shape=(taillemorceau,))

encodeur1 = Dense(Layer1,activation = 'sigmoid',name = '1_encodeur')(input_img)
encodeur2 = Dense(Layer2,activation = 'sigmoid',name = '2_encodeur')(encodeur1)

decodeur1 = Dense(Layer2,activation = 'sigmoid',name = '1_decodeur')(encodeur2)
decodeur2 = Dense(Layer1,activation = 'sigmoid',name = '2_decodeur')(decodeur1)

autoencodeur = Model(input_img, decodeur2)
autoencodeur.summary()

#encodeur
encodeur = Model(input_img , encodeur2)
encodeur.summary()


#decodeur
input_reduit = Input(shape=(NB_dim_reduit,))
decoder_layer1 = autoencodeur.layers[-1]
decoder_layer2 = autoencodeur.layers[-2]


decoder = Model(input_reduit,decoder_layer1(decoder_layer2(input_reduit)))
decoder.summary()

#compilation
int_vitesse_grad = 0.00005 #vitesse du gradient (trop grand on approche pas trop petit on avance pas)
autoencodeur.compile(loss ='mean_squared_error', optimizer = Adam(lr = int_vitesse_grad))

#normalisation des données eventuellement
print('max des images',max(np.reshape(echantillon,(NBpixelTotal))))
print('min des images',min(np.reshape(echantillon,(NBpixelTotal))))

history = autoencodeur.fit(image1CoupeLigne,image1CoupeLigne,
                epochs=100,
                batch_size=1,#nombre de fois ou l'on calcule le gradient pour faire un moyenne avant utilisation
                shuffle=False, #normalisation des parametres avant réseau pas besoin ici
                validation_data=(image2CoupeLigne,image2CoupeLigne),
                verbose = 2)

plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.legend(('validation','entrainement'))
plt.show()
#test
#train 
encoded_image = encodeur.predict(image1CoupeLigne)
decoded_imgs = decoder.predict(encoded_image)
decodedautoencodeur = autoencodeur.predict(image1CoupeLigne)

decoded_imgs = np.reshape(decoded_imgs,(NBmorceau,tailleX,tailleY,NBbande))
decodedautoencodeur =  np.reshape(decodedautoencodeur,(NBmorceau,tailleX,tailleY,NBbande))
reference = np.reshape(image1CoupeLigne,(NBmorceau,tailleX,tailleY,NBbande))

for i in range(0 , 2):
    f, (ax1, ax2 , ax3) = plt.subplots(1, 3, sharey=True)
    ax1.set_title('train image origine')
    ax1.imshow(reference[i,:,:,0])
    ax2.set_title('train image retrouvé')
    ax2.imshow(decoded_imgs[i,:,:,0])
    ax3.set_title('autoencodeur')
    ax3.imshow(decodedautoencodeur[i,:,:,0])


#test    
encoded_image = encodeur.predict(image2CoupeLigne)
decoded_imgs = decoder.predict(encoded_image)
decodedautoencodeur = autoencodeur.predict(image2CoupeLigne)

decoded_imgs = np.reshape(decoded_imgs,(NBmorceau,tailleX,tailleY,NBbande))
decodedautoencodeur =  np.reshape(decodedautoencodeur,(NBmorceau,tailleX,tailleY,NBbande))
reference = np.reshape(image2CoupeLigne,(NBmorceau,tailleX,tailleY,NBbande))

for i in range(0 , 2):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.set_title('test image origine')
    ax1.imshow(reference[i,:,:,0])
    ax2.set_title('test image retrouvé')
    ax2.imshow(decoded_imgs[i,:,:,0])
    ax3.set_title('autoencodeur')
    ax3.imshow(decodedautoencodeur[i,:,:,0])    

    