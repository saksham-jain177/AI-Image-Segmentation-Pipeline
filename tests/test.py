from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image_path = 'E:\saksham-jain-wasserstoff-AiInternTask\data\input_images\sample1.jpg'
image = Image.open(image_path)
extracted_text = pytesseract.image_to_string(image)
print(extracted_text)


#just to check if tesseract is working or not