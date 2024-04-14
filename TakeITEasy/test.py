import cv2 as cv

# this is a wrapper around command line tesseract, so tesseract needs to be installed as well:
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

def affiche(img):
    cv.namedWindow('display', cv.WINDOW_NORMAL) 
    cv.resizeWindow('display', 900, 900) 
    cv.imshow('display', img)
    cv.waitKey(0) 
    cv.destroyAllWindows() 
path = 'img/res_800/'

for i in range(1,10):
    # not sure if higher resolution 2000x2000 would be better:
    filename = path + str(i) + '.png'
    # - sign is so that the number if black with withish background:
    img = -cv.imread(filename,cv.IMREAD_GRAYSCALE) 
    # if we know the position of the image on the board, only three possibilities:
    if i in [1,5,9]:
        whitelist = '159'
    if i in [2,6,7]:
        whitelist = '267'
    if i in [3,4,8]:
        whitelist = '348'
    affiche (img)
    # --psm 10 is to indicate we're looking for a single character: 
    print('detecting', i,':', pytesseract.image_to_string(img, config='--psm 10 -c tessedit_char_whitelist=' + whitelist))

