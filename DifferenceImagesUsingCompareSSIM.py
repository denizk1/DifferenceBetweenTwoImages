from skimage.metrics import structural_similarity as compare_ssim
import cv2
import numpy as np
import imutils

def CompareSSIM(img1,img2):
    #resim okuma
    image1 = cv2.imread(img1)
    image2 = cv2.imread(img2)


    #resimleri gri tona cevirme
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


    #iki görüntü arasındaki benzerlik icin SSIM (iki benzer goruntu arasındaki algısal fark olcumu) kullanarak iki resim arasındaki benzerligi buluyoruz
    (score, diff) = compare_ssim(gray_image1, gray_image2, full=True)

    #diff [0,1] aralıgında kayan nokta veri türü olarak temsil edilir . Bu yuzden bu farkı 8-bitlik goruntuye ceviriyoruz ve bir fark goruntusu elde ediyoruz
    diff = (diff * 255).astype("uint8")

    #Giriş olarak verilen görüntüyü ikili görüntüye çevirmek için kullanılan bir yöntem olan threshold yöntemi kullaniyoruz. Böylelikle farklı olan nesneleri belirlemek daha kolay olur.
    #Kaynak olarak alınan görüntü üzerindeki piksel,esik degeri olarak verilen değerden küçükse maksimum deger olarak verilen parametre değerine atanır
    #Böylelikle farklı olan nesneler siyah,geri kalani beyaz olarak esiklenir
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    #farkli olan nesnelerin sinirlarini belirlemek icin opencv nin FindContours metodunu kullaniyoruz
    #Böylelikle resmin sinirlarini belirleyip kare icine almak daha kolay olacaktir
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours=imutils.grab_contours(contours)

    mask = np.zeros(image1.shape, dtype='uint8')
    filled_after = image2.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(image2, (x, y), (x + w, y + h), (36, 255, 12), 2)

    return image1,image2