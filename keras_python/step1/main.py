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
tailleX = 25
tailleY = 20

#lecture de l'image et divisiond des images
NBimage = 2;

img1 = io.imread('modisGrenoble.tif')
plt.imshow(img1[:,:,1])
plt.show()



img2 = io.imread('modisGrenoble2.tif')
plt.imshow(img2[:,:,1])
plt.show()
tailleimg = img2.shape
NBpixel = tailleimg[0]*tailleimg[1]*tailleimg[2]
NB_bande = tailleimg[2]
taillemorceau = tailleX*tailleY*NB_bande

echantillon = np.concatenate((image.couper_image(divisionx,divisiony,NBmorceau,tailleX,tailleY,img1)
                            ,image.couper_image(divisionx,divisiony,NBmorceau,tailleX,tailleY,img2))
                            ,3)

plt.imshow(echantillon[:,:,1,1])
plt.show()
NBmorceau=NBimage*NBmorceau
NBpixel=NBimage*NBpixel
#parametre du reseau
NBn_enc_1=100
NBn_enc_2=40
int_vitesse_grad = 0.1

#fabrication du reseau de neurone
input_img = Input(shape=(taillemorceau,))

encodeur1 = Dense(NBn_enc_1,activation = 'sigmoid',name = '1_encodeur')(input_img)
encodeur2 = Dense(NBn_enc_2,activation = 'sigmoid',name = '2_encodeur')(encodeur1)


decodeur1 = Dense(NBn_enc_2,activation = 'sigmoid',name = '1_decodeur')(encodeur2)
decodeur2 = Dense(NBn_enc_1,activation = 'sigmoid',name = '2_decodeur')(decodeur1)
decodeur3 = Dense(taillemorceau,activation = 'sigmoid',name = '3_decodeur')(decodeur2)

#autoencodeur
autoencodeur = Model(input_img, decodeur3)
autoencodeur.summary()

#encodeur
encodeur = Model(input_img , encodeur2)
#encodeur.summary()

#decodeur
input_reduit = Input(shape=(NBn_enc_2,))
decoder_layer1 = autoencodeur.layers[-1]
decoder_layer2 = autoencodeur.layers[-2]
decoder_layer3 = autoencodeur.layers[-3]

decoder = Model(input_reduit,decoder_layer1(decoder_layer2(decoder_layer3(input_reduit))))
#decoder.summary()

#compilation
autoencodeur.compile(loss ='mean_squared_error', optimizer = Adam(lr = int_vitesse_grad))

#normalisation des données eventuellement
print('max des images',max(np.reshape(echantillon,(NBpixel))))
print('min des images',min(np.reshape(echantillon,(NBpixel))))

#parametre entrainement
part_entrainement=1 #non protegé contre les divisions non entière
part_test_entrainement=0
echantillon_reshape = np.zeros((NBmorceau,taillemorceau))
for i in range(0 , NBmorceau):
    echantillon_reshape[i,:] = np.reshape(echantillon[:,:,:,i],(taillemorceau))

entrainement = echantillon_reshape[0:int(NBmorceau*part_entrainement-1),:]
validation = echantillon_reshape[int(NBmorceau*part_entrainement):int(NBmorceau*(part_entrainement+part_test_entrainement))-1,:]
#entrainement
autoencodeur.fit(entrainement,entrainement,
                epochs=150,
                batch_size=taillemorceau,
                shuffle=False,
                validation_data=(validation, validation))

#test du résultat
#image du reseau
test=image.couper_image_decal(divisionx,divisiony,int(NBmorceau/2),tailleX,tailleY,img1)

test_reshape = np.zeros((int(NBmorceau/2),taillemorceau))
for i in range(0 , int(NBmorceau/2)):
    test_reshape[i,:] = np.reshape(echantillon[:,:,:,i],(taillemorceau))

encoded_image = encodeur.predict(test_reshape)
decoded_imgs = decoder.predict(encoded_image)

decoded_imgs = np.reshape(decoded_imgs,(int(NBmorceau/2),tailleX,tailleY,NB_bande))


#affichage
for i in range(1 , 10):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.set_title('image retrouvé')
    ax1.imshow(decoded_imgs[i,:,:,2])
    ax2.set_title('image origine')
    ax2.imshow(test[i,:,:,2])


