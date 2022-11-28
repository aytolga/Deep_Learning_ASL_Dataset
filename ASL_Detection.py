from tensorflow.python.client import device_lib

#print(device_lib.list_local_devices())  #Ekran kartı uygunluğunu kontrol eder.

#Verileri Yüklemek için gereken kütüphaneler
import os
import cv2
import numpy as np

#Veri görselleştirme
import matplotlib.pyplot as plt

#Model Eğitimi
from keras import utils
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split

"""
Yazar: Tolga AY

Bu uygulamada işaret dili sınıflandırıcısı yapılmıştır.

Gerekli açıklamalar kodda yapılmıştır.

aytolga@outlook.com

"""

"""
verileri_al fonksiyonunun çalışma mantığı:

Öncelikle goruntuler ve etiketler olarak iki ayrı liste açılır.
Uygun bir veri seti üzerinden os modülü yardımıyla verilere erişilir.
Daha sonrasında goruntu değişkeni ile, veri sırası, verilerin etiketleri ve görseller alınır.
cv2 yardımıyla bu görseller işlenir ve kullanılmaya hazır hale getirilir.
"""

def verileri_al(egitim_seti):

    goruntuler = []
    etiketler = []

    dir_list = os.listdir(egitim_seti)

    for i in range(len(dir_list)):
        print("Görüntüler alınıyor...")
        for goruntu in os.listdir(egitim_seti + '/' + egitim_seti[i]):
            img = cv2.imread(egitim_seti + '/' + egitim_seti[i] + '/' + goruntu)
            img = cv2.resize(img, (32, 32))
            goruntuler.append(img)
            etiketler.append(i)

    return goruntuler, etiketler
"""
Verileri vektörleştirmek ve 0-1 araasına getirmek için kullanılan fonksiyon,
daha sonrasında eğitim,test olarak ayırdık.
"""
def veriyi_hazirla(x,y):
    x_array = np.array(x).astype("float32")

    yeni_x = x_array/255.0

    encode_y = utils.to_categorical(y) #Y verilerinin normalleştirilmesi.

    x_train, x_test, y_train, y_test = train_test_split(yeni_x, encode_y, test_size=0.1)

    return x_train, x_test, y_train, y_test
"""
matplotlib yardımıyla verileri görselleştirdik.
"""
def gorsellestir():
    plt.figure(figsize=(16,5))

    for i in range (0,29):
        plt.subplot(3,10,i+1)
        plt.xticks([])
        plt.yticks([])
        path = egitim_seti + "/{0}/{0}1.jpg".format(classes[i])
        img = plt.imread(path)
        plt.imshow(img)
        plt.xlabel(classes[i])

    plt.show()


classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',   #Harfler için sınıflar ve çeşitler.
           'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

egitim_seti = 'C:/Users/Tolga Ay/PycharmProjects/Opencvproject/tlg_alphabet_train'
test_seti= 'C:/Users/Tolga Ay/PycharmProjects/Opencvproject/tlg_alphabet_test'

gorseller, etiketler = verileri_al(egitim_seti)   #Görüntüleri ve etiketleri almak için verileri_al fonksiyonu çağrıldı.

gorsellestir()   #Görüntüleri grafikleştirdik.


x_train, x_test, y_train, y_test = veriyi_hazirla(gorseller, etiketler)

classes_for_CNN = 29
batch = 32
epochs = 15

model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())  #Merkezleri düzenler daha stabil çalışmasını sağlar

model.add(Conv2D(128, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())  #Çıkış olarak 3*3 matristen 9*1 vektör yapar.
model.add(Dropout(0.2)) #Bırakma regülarizasyon için gerekli.
model.add(Dense(1024, activation='relu'))
model.add(Dense(classes_for_CNN, activation='softmax'))

adam = Adam(lr=0.001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

history = model.fit(x_train, y_train, batch_size=batch, epochs=epochs, validation_split=0.2, shuffle = True, verbose=1)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

"""
Modeli h5 dosyası olarak kaydettik.

"""
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_for_als.h5")
print("Saved model to disk")