import PyPDF2
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTFigure
import pdfplumber
from pdf2image import convert_from_bytes

import pytesseract
import logging
import requests
from io import BytesIO

import signal
import time
import functools


import time
import signal


class ModuleException(Exception):
    """Exception to be raised in this module."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class TimeoutError(ModuleException):
    """To be raised if the function takes too much time"""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class BadUrlException(ModuleException):
    """Raised when the url is not valid"""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def timeout(seconds_before_timeout):
    """A decorator that raises an error once the function takes longer than `seconds_before_timeout` time"""

    def wrapper_wrapper(func):
        def handler(signum, frame):
            raise TimeoutError()

        @functools.wraps(func)
        def wrapper_function(*args, **kwargs):
            old = signal.signal(signal.SIGALRM, handler)
            old_time_left = signal.alarm(seconds_before_timeout)
            if (
                0 < old_time_left < second_before_timeout
            ):  # never lengthen existing timer
                signal.alarm(old_time_left)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                if old_time_left > 0:  # deduct f's run time from the saved timer
                    old_time_left -= time.time() - start_time
                signal.signal(signal.SIGALRM, old)
                signal.alarm(old_time_left)
            return result

        return wrapper_function

    return wrapper_wrapper


def get_simple_logger(name, level="info"):
    """Creates a simple loger that outputs to stdout"""
    level_to_int_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    if isinstance(level, str):
        level = level_to_int_map[level.lower()]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class PDFExtractor:
    """A class to extract pdf information. The information that are extracted from the class are:

    - Textual Information
    - Tabular Information
    - Image Information
    """

    def __init__(
        self,
        file_path,
        min_characters=5,
        maximum_pages=3,
        is_url=True,
    ) -> None:
        """A class that can be used to extract pdf information from a file path or url

        Parameters
        ----------
        file_path : str
            The path to the file or the url
        min_characters : int, optional
            The minimum number of characters that a text needs to have to be considered a relevant text, by default 5
        maximum_pages : int, optional
            The maximum number of pages that the pdf can have, by default 5
        is_url : bool, optional
            Whether the file_path is a url or not, by default True

        Raises
        ------
        ValueError
            If the url raises a status code other than 200

        Returns
        -------
        None

        Examples
        --------
        >>> extractor = PDFExtractor(file_path="file_path", min_characters=5, maximum_pages=5, is_url=False)
        >>> extractor.extract_pages()
        """
        self.file_path = file_path
        self.min_characters = min_characters
        self.maximum_pages = maximum_pages
        self.logger = get_simple_logger(name="pdf_extractor", level="info")
        try:
            self.byte_file = self._create_byte_object(self.file_path, is_url)
        except Exception as e:
            self.logger.error(e)
            raise BadUrlException(str(e))
        # PyPDF2 object
        self.pypdf_object = PyPDF2.PdfReader(self.byte_file)
        self.pdf_plumber_object = pdfplumber.open(self.byte_file)
        # Pages from pdfminer
        self.pdfminer_pages = extract_pages(self.byte_file)
        self.page_contents = []
        self.final_content = ""
        self.one_page_contents = {
            "contents": [],
            "images": [],
            "tables": [],
        }

    # @timeout(120)
    def _create_byte_object(self, path, is_url):
        """Creates the byte file based on if it is url or not

        Parameters
        ----------
        path : str
            The path to the file or the url
        is_url : bool
            Whether the path is a url or not

        Returns
        -------
        BytesIO
            The byte file object

        Raises
        ------
        BadUrlException
            If the url raises a status code other than 200
        """
        if is_url:
            res = requests.get(path)
            if res.status_code == 200:
                byte_file = BytesIO(res.content)
            else:
                raise BadUrlException(f"The url raised status code: {res.status_code}")
        else:
            byte_file = open(path, "rb")

        return byte_file

    def __check_for_relevant_text(self, text):
        """Checks if the text is a relevant text or not

        Parameters
        ----------
        text : str
            The text to be checked

        Returns
        -------
        bool
            True if the text is a relevant text, False otherwise
        """
        # Remove the line breaker from the text
        text_ = text.replace("\n", "")
        if len(text_) > self.min_characters:
            self.logger.debug(f"Text: {text_} is a relevant text.")
            return True
        self.logger.debug(f"Text: {text_} is not a relevant text.")
        return False

    def _handle_text(self, component):
        """Handles the extraction of textual information

        Parameters
        ----------
        component : pdfplumber.page.Page
            The page object to extract the text from

        Returns
        -------
        bool
            True if the text is a relevant text, False otherwise

        """
        text = component.get_text()
        if self.__check_for_relevant_text(text):
            self.one_page_contents["contents"].append(text)
            return True
        return False

    def __table_converter(self, table):
        """Converts table from the output given by pdf_plumnber to make it more readble

        Parameters
        ----------
        table : list
            The table to be converted

        Returns
        -------
        str
            The converted table as a string

        Examples
        --------
        >>> table = [["Name", "Age"], ["John", "30"], ["Jane", "25"]]
        >>> table_string = __table_converter(table)
        >>> print(table_string)
        |Name|Age|
        |John|30|
        |Jane|25|

        >>> table = [["Name", "Age"], ["John", "30"], ["Jane", "25"], ["\n", "None"]]
        >>> table_string = __table_converter(table)
        >>> print(table_string)
        |Name|Age|
        |John|30|
        |Jane|25|
        |None|None|

        >>> table = [["Name", "Age"], ["John", "30"], ["Jane", "25"], ["\n", "None"], ["\n", "None"]]
        >>> table_string = __table_converter(table)
        """
        table_string = ""
        # Iterate through each row of the table
        for row_num in range(len(table)):
            row = table[row_num]
            # Remove the line breaker from the wrapted texts
            cleaned_row = [
                item.replace("\n", " ")
                if item is not None and "\n" in item
                else "None"
                if item is None
                else item
                for item in row
            ]
            # Convert the table into a string
            table_string += "|" + "|".join(cleaned_row) + "|" + "\n"
        # Removing the last line break
        table_string = table_string[:-1]
        return table_string

    def _handle_table(self, page_number):
        """Handles the extraction of tabular information

        Parameters
        ----------
        page_number : int
            The page number to extract the table from

        Returns
        -------
        None
        """
        page = self.pdf_plumber_object.pages[page_number]
        try:
            tables = page.extract_tables()
        except:
            tables = [
                ["NA", "NA"],
            ]
        tables_final = [self.__table_converter(table) for table in tables]
        self.one_page_contents["tables"] = tables_final

    # Create a function to crop the image elements from PDFs

    def __crop_image(self, element, page_number):
        """Crops the pdf and creates a new pdf with only the cropped area. This will later be converted into image and then OCRed using pytesseract

        Parameters
        ----------
        element : pdfminer.layout.LTTextContainer
            The element to crop from the pdf
        page_number : int
            The page number to crop the element from

        Returns
        -------
        bytes
            The cropped pdf as a byte object
        """
        pypdf_page = self.pypdf_object.pages[page_number]
        # Get the coordinates to crop the image from PDF
        [image_left, image_top, image_right, image_bottom] = [
            element.x0,
            element.y0,
            element.x1,
            element.y1,
        ]
        # Crop the page using coordinates (left, bottom, right, top)
        pypdf_page.mediabox.lower_left = (image_left, image_bottom)
        pypdf_page.mediabox.upper_right = (image_right, image_top)
        # Save the cropped page to a new PDF
        cropped_pdf_writer = PyPDF2.PdfWriter()
        cropped_pdf_writer.add_page(pypdf_page)
        # convert to byte
        cropped_pdf_stream = BytesIO()
        cropped_pdf_writer.write(cropped_pdf_stream)
        byte_object = cropped_pdf_stream.getvalue()
        return byte_object

    # Create a function to convert the PDF to images
    def _convert_to_images(self, pdf_byte):
        """Converts the pdf byte object to images

        Parameters
        ----------
        pdf_byte : bytes
            The pdf byte object to be converted

        Returns
        -------
        PIL.Image
            The converted image
        """
        images = convert_from_bytes(pdf_byte)
        image = images[0]
        return image

    # @timeout(20)
    def _image_to_text(self, image):
        """Extracts text from image using pytesseract

        Parameters
        ----------
        image : PIL.Image
            The image to extract text from

        Returns
        -------
        str
            The extracted text from the image

        Examples
        --------
        >>> image = PIL.Image.open("image.jpg")
        >>> text = _image_to_text(image)
        >>> print(text)
        DUMMY TEXT
        """
        text = pytesseract.image_to_string(image)
        # text = "DUMMY TEXT"
        self.logger.debug(f"Extracted {text} from the image.")
        return text

    def _handle_image(self, element, page_number):
        """Handles the extraction of image information

        Parameters
        ----------
        element : pdfminer.layout.LTFigure
            The element to extract the image from
        page_number : int
            The page number to extract the image from

        Returns
        -------
        bool
            True if the image is a relevant image, False otherwise

        Notes
        -----
        Extract the text from the image using pytesseract. Check if the text is a relevant text or not. If the text is a relevant text, add it to the one_page_contents dictionary with the key "images"
        If the text is not a relevant text, do nothing
        Return True if the image is a relevant image, False otherwise
        If the image is a relevant image, add it to the one_page_contents dictionary with the key "images"
        If the image is not a relevant image, do nothing
        Return True if the image is a relevant image, False otherwise
        """
        cropped_pdf = self.__crop_image(element, page_number)
        image = self._convert_to_images(pdf_byte=cropped_pdf)
        extracted_text = self._image_to_text(image)
        if self.__check_for_relevant_text(extracted_text):
            self.one_page_contents["images"].append(extracted_text)
            return True
        return False

    def extract_one_page(self, page_number, pdfminer_page):
        """Extracts information from one page of the pdf

        Parameters
        ----------
        page_number : int
            The page number to extract the information from
        pdfminer_page : pdfminer.layout.LTPage
            The pdfminer page object to extract the information from

        Returns
        -------
        None
        """
        self.one_page_contents = {
            "contents": [],
            "images": [],
            "tables": [],
        }
        self._handle_table(page_number=page_number)
        max_image_per_page = 2
        image_number = 0
        for element_number, element in enumerate(pdfminer_page._objs):
            type_ = type(element)
            self.logger.debug(
                f"Handling Page: {page_number}, Element: {element_number} Type: {type_}"
            )

            if isinstance(element, LTTextContainer):
                self._handle_text(element)

            if isinstance(element, LTFigure) and image_number < max_image_per_page:
                added = self._handle_image(
                    element=element,
                    page_number=page_number,
                )
                if added:
                    image_number += 1
        self.page_contents.append(self.one_page_contents)

    def create_final_text_content(self):
        """Creates the final text using the information extracted so far. The final text has all the textual information, image information and tabular information. This text directly can be used for machine learning.

        Returns
        -------
        str
            The final text content of the pdf. This text can be directly used for machine learning.
        """
        final_content = ""
        for page_number, content in enumerate(self.page_contents):
            final_content += f"PAGE {page_number}\n"
            text_contents = content["contents"]
            final_content += "\n".join(text_contents)
            for i, image in enumerate(content["images"]):
                final_content += f"IMAGE {i}\n{image.strip()}\nIMAGE {i} ENDS\n"

            for i, table in enumerate(content["tables"]):
                final_content += f"TABLE {i}\n{table.strip()}\nTABLE {i} ENDS\n"
            final_content += f"PAGE {page_number} ENDS\n"
        self.final_content = final_content
        return final_content

    def extract_pages(self):
        """Extracts information from all the pages of the pdf. This is the final method to be used

        Parameters
        ----------
        None

        Returns
        -------
        str
            The final text content of the pdf. This text can be directly used for machine learning.
        """
        pages = extract_pages(self.byte_file)
        for page_number, page in enumerate(pages):
            self.logger.info(f"Working on the page: {page_number}")
            if page_number >= self.maximum_pages:
                self.logger.info(f"Maximum page limit reached. Breaking...")
                break
            self.extract_one_page(page_number, page)

        final_content = self.create_final_text_content()
        return final_content
