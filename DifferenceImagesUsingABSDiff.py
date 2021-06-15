import cv2
import imutils

def ABSDiff(img1,img2):
    #resim okuma
    image1 = cv2.imread(img1)
    image2 = cv2.imread(img2)

    #iki dizi arasindaki eleman basina mutlak farkı hesaplıyoruz. Fark, uçuncu bagimsiz deiskende donduruluyor.
    diff = image1.copy()
    cv2.absdiff(image1, image2, diff)

    #gri tona cevirme
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # hepsini yakalamak icin farkliliklarin boyutunu arttiriyoruz. Goruntu uzerinde parametreler ile verilen alan icerisindeki sinirlar genisletmektiliyor bu genisletme sayesinde piksel guruplari buyur ve pikseller arasi bosluklar kuculur.
    for i in range(0, 4):
        dilated = cv2.dilate(gray.copy(), None, iterations=i + 1)

    (T, thresh) = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts[:-1]:

        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image1,image2