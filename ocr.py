#import pytesseract
#from pytesseract import Output
#from pdf2image import convert_from_path
#import os
#from PIL import Image, ImageOps
#import camelot
#
## Ensure Tesseract is installed and set the path if needed
#pytesseract.pytesseract_cmd = r"/opt/homebrew/bin/tesseract"
#
#def extract_text_images_and_tables(pdf_path, output_folder):
#    """
#    Extracts text, individual images, and tables from a scanned PDF and saves them as separate files.
#
#    Args:
#        pdf_path (str): Path to the scanned PDF.
#        output_folder (str): Folder to save the extracted content.
#    """
#    # Convert PDF to images
#    poppler_path = "/opt/homebrew/bin"
#    pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
#
#    # Define the path for the text output
#    text_document_path = os.path.join(output_folder, "extracted_text.txt")
#
#    # Write the extracted text to a .txt file
#    with open(text_document_path, "w", encoding="utf-8") as text_file:
#        for page_number, page in enumerate(pages):
#            # Save page as an image
#            image_path = os.path.join(output_folder, f"page_{page_number + 1}.png")
#            page.save(image_path, "PNG")
#
#            # OCR to extract text
#            extracted_text = pytesseract.image_to_string(Image.open(image_path))
#
#            # Write the extracted text to the document
#            text_file.write(f"--- Page {page_number + 1} ---\n")
#            text_file.write(extracted_text)
#            text_file.write("\n\n")
# 
#            # Extract individual images from the saved page image
#            page_image = Image.open(image_path)
#            grayscale_image = ImageOps.grayscale(page_image)
#            binary_image = grayscale_image.point(lambda x: 0 if x < 128 else 255, '1')
#
#            # Identify large connected regions for image extraction
#            binary_pixels = binary_image.load()
#            width, height = binary_image.size
#            visited = set()
#            region_count = 0
#
#            def flood_fill(x, y):
#                """Flood-fill algorithm to detect connected regions."""
#                stack = [(x, y)]
#                region = []
#                while stack:
#                    cx, cy = stack.pop()
#                    if (cx, cy) in visited or cx < 0 or cy < 0 or cx >= width or cy >= height:
#                        continue
#                    if binary_pixels[cx, cy] == 0:  # Black pixel
#                        visited.add((cx, cy))
#                        region.append((cx, cy))
#                        stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])
#                return region
#
#            for x in range(0,width,10):
#                for y in range(0,height,10):
#                    if binary_pixels[x, y] == 0 and (x, y) not in visited:
#                        region = flood_fill(x, y)
#                        if len(region) > 5000:  # Minimum size to consider as an image
#                            region_count += 1
#                            x_min = min(p[0] for p in region)
#                            y_min = min(p[1] for p in region)
#                            x_max = max(p[0] for p in region)
#                            y_max = max(p[1] for p in region)
#
#                            # Filter out very small regions
#                            if (x_max - x_min) > 100 and (y_max - y_min) > 100:
#                                # Crop and save the individual image
#                                cropped_image = page_image.crop((x_min, y_min, x_max, y_max))
#                                region_path = os.path.join(output_folder, f"page_{page_number + 1}_region_{region_count}.png")
#                                cropped_image.save(region_path, "PNG")
#                                print(f"Saved region image: {region_path}")
#
#    print(f"Text data saved to {text_document_path}")
#
#          
#    # Extract tables using Camelot with enhanced precision
#    try:
#        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
#        filtered_tables = [table for table in tables if table.accuracy > 90]  # Filter tables with accuracy > 90
#        for i, table in enumerate(filtered_tables):
#            table_path = os.path.join(output_folder, f"table_{i + 1}.csv")
#            table.to_csv(table_path)
#            print(f"High-precision Table {i + 1} saved to {table_path}")
#    except Exception as e:
#        print(f"No tables found or an error occurred during table extraction: {e}")
#
#    print(f"Images saved in {output_folder}")
#
#
## Example usage
#if __name__ == "__main__":
#    pdf_path = "/Users/arshia/Downloads/ocr/SIW Issue 416 28_04_2000 1.pdf"
#    # Replace with the path to your scanned PDF
#    output_folder = "/Users/arshia/Downloads/ocr"  # Replace with your desired output folder
#
#    if not os.path.exists(output_folder):
#        os.makedirs(output_folder)
#
#    extract_text_images_and_tables(pdf_path, output_folder)

#import pytesseract
#from pdf2image import convert_from_path
#import cv2
#import numpy as np
#import os
#from PIL import Image
#import camelot
#
## Ensure Tesseract is installed and set the path if needed
#pytesseract.pytesseract.tesseract_cmd = r"/opt/homebrew/bin/tesseract"
#
#def extract_text_images_and_tables(pdf_path, output_folder):
#    """
#    Extracts text, significant images, and tables from a scanned PDF and saves them as separate files.
#
#    Args:
#        pdf_path (str): Path to the scanned PDF.
#        output_folder (str): Folder to save the extracted content.
#    """
#    # Convert PDF to images
#    poppler_path = "/opt/homebrew/bin"
#    pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
#
#    # Define the path for the text output
#    text_document_path = os.path.join(output_folder, "extracted_text.txt")
#
#    # Write the extracted text to a .txt file
#    with open(text_document_path, "w", encoding="utf-8") as text_file:
#        for page_number, page in enumerate(pages):
#            # Save page as an image
#            image_path = os.path.join(output_folder, f"page_{page_number + 1}.png")
#            page.save(image_path, "PNG")
#
#            # OCR to extract text
#            extracted_text = pytesseract.image_to_string(Image.open(image_path))
#
#            # Write the extracted text to the document
#            text_file.write(f"--- Page {page_number + 1} ---\n")
#            text_file.write(extracted_text)
#            text_file.write("\n\n")
#
#            # Extract significant images using OpenCV
#            img = cv2.imread(image_path)
#            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#            # Apply binary thresholding
#            _, binary = cv2.threshold(gray, 51, 50, cv2.THRESH_BINARY_INV)
#
#            # Find contours
#            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#            region_count = 0
#            for contour in contours:
#                x, y, w, h = cv2.boundingRect(contour)
#
#                # Filter out small regions
#                if w > 100 and h > 100:  # Minimum width and height to consider
#                    region_count += 1
#
#                    # Crop and save the significant image
#                    cropped_image = img[y:y + h, x:x + w]
#                    region_path = os.path.join(output_folder, f"page_{page_number + 1}_region_{region_count}.png")
#                    cv2.imwrite(region_path, cropped_image)
#                    print(f"Saved significant region image: {region_path}")
#
#    print(f"Text data saved to {text_document_path}")
#
#    # Extract tables using Camelot
#    try:
#        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
#        for i, table in enumerate(tables):
#            table_path = os.path.join(output_folder, f"table_{i + 1}.csv")
#            table.to_csv(table_path)
#            print(f"Table {i + 1} saved to {table_path}")
#    except Exception as e:
#        print(f"No tables found or an error occurred during table extraction: {e}")
#
#    print(f"Images saved in {output_folder}")
#
## Example usage
#if __name__ == "__main__":
#    pdf_path = "/Users/arshia/Downloads/ocr/PublicWaterMassMailing.pdf"
#    # Replace with the path to your scanned PDF
#    output_folder = "/Users/arshia/Downloads/ocr"  # Replace with your desired output folder
#
#    if not os.path.exists(output_folder):
#        os.makedirs(output_folder)
#
#    extract_text_images_and_tables(pdf_path, output_folder)

#
#import pytesseract
#from pytesseract import Output
#from pdf2image import convert_from_path
#import os
#from PIL import Image, ImageOps
#import pdfplumber
#
## Ensure Tesseract is installed and set the path if needed
#pytesseract.pytesseract_cmd = r"/opt/homebrew/bin/tesseract"
#
#def extract_text_images_and_tables(pdf_path, output_folder):
#    """
#    Extracts text, individual images, and tables from a scanned PDF and saves them as separate files.
#
#    Args:
#        pdf_path (str): Path to the scanned PDF.
#        output_folder (str): Folder to save the extracted content.
#    """
#    # Convert PDF to images
#    poppler_path = "/opt/homebrew/bin"
#    pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
#
#    # Define the path for the text output
#    text_document_path = os.path.join(output_folder, "extracted_text.txt")
#
#    # Write the extracted text to a .txt file
#    with open(text_document_path, "w", encoding="utf-8") as text_file:
#        for page_number, page in enumerate(pages):
#            # Save page as an image
#            image_path = os.path.join(output_folder, f"page_{page_number + 1}.png")
#            page.save(image_path, "PNG")
#
#            # OCR to extract text
#            extracted_text = pytesseract.image_to_string(Image.open(image_path))
#
#            # Write the extracted text to the document
#            text_file.write(f"--- Page {page_number + 1} ---\n")
#            text_file.write(extracted_text)
#            text_file.write("\n\n")
#
#            # Extract individual images from the saved page image
#            page_image = Image.open(image_path)
#            grayscale_image = ImageOps.grayscale(page_image)
#            binary_image = grayscale_image.point(lambda x: 0 if x < 128 else 255, '1')
#
#            # Identify large connected regions for image extraction
#            binary_pixels = binary_image.load()
#            width, height = binary_image.size
#            visited = set()
#            region_count = 0
#
#            def flood_fill(x, y):
#                """Flood-fill algorithm to detect connected regions."""
#                stack = [(x, y)]
#                region = []
#                while stack:
#                    cx, cy = stack.pop()
#                    if (cx, cy) in visited or cx < 0 or cy < 0 or cx >= width or cy >= height:
#                        continue
#                    if binary_pixels[cx, cy] == 0:  # Black pixel
#                        visited.add((cx, cy))
#                        region.append((cx, cy))
#                        stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])
#                return region
#
#            for x in range(0,width,10):
#                for y in range(0,height,10):
#                    if binary_pixels[x, y] == 0 and (x, y) not in visited:
#                        region = flood_fill(x, y)
#                        if len(region) > 5000:  # Minimum size to consider as an image
#                            region_count += 1
#                            x_min = min(p[0] for p in region)
#                            y_min = min(p[1] for p in region)
#                            x_max = max(p[0] for p in region)
#                            y_max = max(p[1] for p in region)
#
#                            # Filter out very small regions
#                            if (x_max - x_min) > 100 and (y_max - y_min) > 100:
#                                # Crop and save the individual image
#                                cropped_image = page_image.crop((x_min, y_min, x_max, y_max))
#                                region_path = os.path.join(output_folder, f"page_{page_number + 1}_region_{region_count}.png")
#                                cropped_image.save(region_path, "PNG")
#                                print(f"Saved region image: {region_path}")
#
#    print(f"Text data saved to {text_document_path}")
#
#
#    # Extract tables using pdfplumber for better accuracy
#    try:
#        with pdfplumber.open(pdf_path) as pdf:
#            for page_number, page in enumerate(pdf.pages, start=1):
#                tables = page.extract_tables()
#                for table_index, table in enumerate(tables):
#                    table_path = os.path.join(output_folder, f"page_{page_number}_table_{table_index + 1}.csv")
#                    with open(table_path, "w", encoding="utf-8") as csv_file:
#                        for row in table:
#                            csv_file.write(",".join([str(cell) if cell is not None else "" for cell in row]) + "\n")
#                    print(f"Saved Table {table_index + 1} from Page {page_number} to {table_path}")
#    except Exception as e:
#        print(f"No tables found or an error occurred during table extraction: {e}")
#
#    print(f"Tables and images saved in {output_folder}")
#
## Example usage
#if __name__ == "__main__":
#    pdf_path = "/Users/arshia/Downloads/ocr/SIW Issue 416 28_04_2000 1.pdf"
#    # Replace with the path to your scanned PDF
#    output_folder = "/Users/arshia/Downloads/ocr"  # Replace with your desired output folder
#
#    if not os.path.exists(output_folder):
#        os.makedirs(output_folder)
#
#    extract_text_images_and_tables(pdf_path, output_folder)

import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
import os
from PIL import Image, ImageOps
import camelot
import pandas as pd

# Ensure Tesseract is installed and set the path if needed
pytesseract.pytesseract_cmd = r"/opt/homebrew/bin/tesseract"

def extract_text_images_and_tables(pdf_path, output_folder):
    """
    Extracts text, individual images, and tables from a scanned PDF and saves them as separate files.

    Args:
        pdf_path (str): Path to the scanned PDF.
        output_folder (str): Folder to save the extracted content.
    """
    # Convert PDF to images
    poppler_path = "/opt/homebrew/bin"
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)

    # Define the path for the text output
    text_document_path = os.path.join(output_folder, "extracted_text.txt")

    # Write the extracted text to a .txt file
    with open(text_document_path, "w", encoding="utf-8") as text_file:
        for page_number, page in enumerate(pages):
            # Save page as an image
            image_path = os.path.join(output_folder, f"page_{page_number + 1}.png")
            page.save(image_path, "PNG")

            # OCR to extract text
            extracted_text = pytesseract.image_to_string(Image.open(image_path), config="--psm 6")

            # Write the extracted text to the document
            text_file.write(f"--- Page {page_number + 1} ---\n")
            text_file.write(extracted_text)
            text_file.write("\n\n")

            # Extract individual images from the saved page image
            page_image = Image.open(image_path)
            grayscale_image = ImageOps.grayscale(page_image)
            binary_image = grayscale_image.point(lambda x: 0 if x < 128 else 255, '1')

            # Identify large connected regions for image extraction
            binary_pixels = binary_image.load()
            width, height = binary_image.size
            visited = set()
            region_count = 0

            def flood_fill(x, y):
                """Flood-fill algorithm to detect connected regions."""
                stack = [(x, y)]
                region = []
                while stack:
                    cx, cy = stack.pop()
                    if (cx, cy) in visited or cx < 0 or cy < 0 or cx >= width or cy >= height:
                        continue
                    if binary_pixels[cx, cy] == 0:  # Black pixel
                        visited.add((cx, cy))
                        region.append((cx, cy))
                        stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])
                return region

            for x in range(0, width, 10):
                for y in range(0, height, 10):
                    if binary_pixels[x, y] == 0 and (x, y) not in visited:
                        region = flood_fill(x, y)
                        if len(region) > 5000:  # Minimum size to consider as an image
                            region_count += 1
                            x_min = min(p[0] for p in region)
                            y_min = min(p[1] for p in region)
                            x_max = max(p[0] for p in region)
                            y_max = max(p[1] for p in region)

                            # Filter out very small regions
                            if (x_max - x_min) > 100 and (y_max - y_min) > 100:
                                # Crop and save the individual image
                                cropped_image = page_image.crop((x_min, y_min, x_max, y_max))
                                region_path = os.path.join(output_folder, f"page_{page_number + 1}_region_{region_count}.png")
                                cropped_image.save(region_path, "PNG")
                                print(f"Saved region image: {region_path}")

    print(f"Text data saved to {text_document_path}")

    # Extract tables using Camelot with enhanced formatting precision
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream", strip_text=" \n")
        filtered_tables = [table for table in tables if table.accuracy > 90]  # Filter tables with accuracy > 90
        for i, table in enumerate(filtered_tables):
            # Use Pandas to reformat table for better accuracy
            df = table.df
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # Clean whitespace
            df = df.replace('', pd.NA).dropna(how='all')  # Remove empty rows

            # Ensure consistent formatting
            df = df.fillna('').astype(str)

            # Save the reformatted table to CSV
            table_path = os.path.join(output_folder, f"table_{i + 1}.csv")
            df.to_csv(table_path, index=False, header=False, encoding="utf-8-sig")
            print(f"Formatted High-precision Table {i + 1} saved to {table_path}")
    except Exception as e:
        print(f"No tables found or an error occurred during table extraction: {e}")

    print(f"Images saved in {output_folder}")

# Example usage
if __name__ == "__main__":
    pdf_path = "/Users/arshia/Downloads/ocr/SIW Issue 416 28_04_2000 1.pdf"
    # Replace with the path to your scanned PDF
    output_folder = "/Users/arshia/Downloads/ocr"  # Replace with your desired output folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    extract_text_images_and_tables(pdf_path, output_folder)
