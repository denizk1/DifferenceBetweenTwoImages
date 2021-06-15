import cv2
import imutils

def Subtract(img1,img2):
    #resim okuma
    image1 = cv2.imread(img1)
    image2 = cv2.imread(img2)

    # iki resmi birbirinden cikartma islemi yapılarak aralarındaki farkı bir degiskene atiyoruz
    difference = cv2.subtract(image1, image2)

    #resmin pikselleri üzerinde degisikligin kolay yapilmasi icin gri tona ceviriyoruz
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)


    #Giris olarak verilen goruntuyu ikili goruntuye cevirmek icin kullanilan bir yontem olan threshold yontemi kullaniyoruz. Boylelikle farkli olan nesneleri belirlemek daha kolay olur.
    #Kaynak olarak alınan goruntu uzerindeki piksel,esik degeri olarak verilen degerden kucukse maksimum deger olarak verilen parametre degerine atanir
    #Boylelikle farkli olan nesneler siyah,geri kalani beyaz olarak esiklenir
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    #farkli olan nesnelerin sinirlarini belirlemek icin opencv nin FindContours metodunu kullaniyoruz
    #Boylelikle resmin sinirlarini belirleyip kare icine almak daha kolay olacaktir
    cnts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)


    #farklı olan nesneleri dikdörtgen icerisine aliyoruz
    #resmin kendi cerceve sinirini aldigi icin -1 e kadar olan kisim aliniyor
    for c in cnts[:-1]:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return image1,image2

