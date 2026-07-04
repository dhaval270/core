import os
import time
import logging
from dotenv import load_dotenv
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import concurrent.futures
import google.generativeai as genai
from src.utils.helper import load_template
import io
import fitz
from openai import OpenAI
from threading import Lock
import multiprocessing

EXTRACTION_PROMPT = r"core/prompts/text_extraction.jinja2"

# Configure logging
logger = logging.getLogger("core")

# Create locks for thread-safe operations
flag_lock = Lock()


class GeminiExtractor:
    def __init__(self, gemini_api_key, openai_api_key):
        """Initialize with API keys and set up the models."""
        self.gemini_api_key = gemini_api_key
        genai.configure(api_key=self.gemini_api_key)
        self.model = self._init_model()

        # Initialize OpenAI client for moderation
        self.moderation_client = OpenAI(api_key=openai_api_key)

    def _init_model(self):
        """Set up the Gemini model with desired configuration."""
        config = {
            "temperature": 0.0,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 10000,
            "response_mime_type": "text/plain",
        }
        return genai.GenerativeModel(
            model_name="gemini-2.0-flash", generation_config=config
        )

    def split_pdf_into_pages(self, pdf_path):
        """Splits PDF into individual pages in memory"""
        doc = fitz.open(pdf_path)
        pages = []

        # Process each page individually
        for page_num in range(len(doc)):
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            pdf_chunk = new_doc.write()
            pdf_chunk_bytes = io.BytesIO(pdf_chunk)
            # Just store (page_number, chunk_bytes) - simplified!
            pages.append((page_num, pdf_chunk_bytes))

        doc.close()
        return pages

    def moderate_text(self, text, threshold=0.1):
        """
        Check if text contains inappropriate content

        Args:
            text (str): Extracted text to check
            threshold (float): Threshold for flagging content

        Returns:
            tuple: (is_flagged, sexual_score)
        """
        try:
            moderation_response = self.moderation_client.moderations.create(
                model="omni-moderation-latest",
                input=[{"type": "text", "text": text}],
            )

            # Access sexual content score
            sexual_score = moderation_response.results[0].category_scores.sexual
            is_flagged = sexual_score > threshold

            if is_flagged:
                logger.warning(
                    f"Flagged content due to irrelevant content. Score: {sexual_score}"
                )

            return is_flagged, sexual_score

        except Exception as e:
            logger.error(f"Moderation failed: {str(e)}")
            return False, 0.0

    def extract_and_moderate(self, page_data, threshold=0.22, stop_event=None):
        """
        Extract text from a PDF page using the Gemini API and moderate it.
        """
        page_number, pdf_bytes = page_data

        if stop_event and stop_event.is_set():
            logger.info(f"Stopping page {page_number + 1} due to stop signal.")
            return (page_number, "", False, 0.0)

        start_time = time.time()
        logger.info("Extracting text from page %s", page_number + 1)

        prompt = load_template(EXTRACTION_PROMPT).render()

        try:
            uploaded_pdf = genai.upload_file(pdf_bytes, mime_type="application/pdf")
            response = self.model.generate_content(
                [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {
                                "file_data": {
                                    "mime_type": "application/pdf",
                                    "file_uri": uploaded_pdf.uri,
                                }
                            },
                        ],
                    }
                ]
            )
            extracted_text = response.text

            extraction_time = time.time() - start_time
            logger.info(
                "Completed extraction for page %s in %.2f seconds",
                page_number + 1,
                extraction_time,
            )

            is_flagged, sexual_score = self.moderate_text(extracted_text, threshold)

            return (page_number, extracted_text, is_flagged, sexual_score)

        except Exception as error:
            logger.error("Error processing page %s: %s", page_number + 1, error)
            return (page_number, "Error processing document", False, 0.0)

    def process_pdf(
        self, input_file_path, threshold=0.22, max_allowed_flags=5, max_workers=4
    ):
        start_time = time.time()
        pages = self.split_pdf_into_pages(input_file_path)
        logger.info(f"Split PDF into {len(pages)} pages")

        total_flagged_count = 0
        flagged_pages = []
        page_contents = {}  # Will store page text content
        early_exit = False
        should_stop = multiprocessing.Event()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.extract_and_moderate, page, threshold, should_stop
                ): page
                for page in pages
            }

            for future in concurrent.futures.as_completed(futures):
                if should_stop.is_set():
                    break

                result = future.result()
                page_number, extracted_text, is_flagged, _ = result

                if should_stop.is_set():
                    continue

                # Store with 1-indexed page number (more natural for users)
                page_contents[page_number + 1] = extracted_text

                if is_flagged:
                    with flag_lock:
                        total_flagged_count += 1
                        flagged_pages.append(
                            page_number + 1
                        )  # Store 1-indexed page number

                        if total_flagged_count >= max_allowed_flags:
                            should_stop.set()
                            logger.warning(
                                "Irrelevant content detected. Cannot upload this book."
                            )
                            early_exit = True
                            break

        # Clean up resources
        for _, pdf_bytes in pages:
            pdf_bytes.close()

        if early_exit:
            combined_text = ""
            page_contents = {}
        else:
            # Create combined text from ordered pages
            combined_text = "\n".join(
                page_contents[page_num + 1]
                for page_num in range(len(pages))
                if page_num + 1 in page_contents
            )
            logger.info(f"Completed extraction for file: {input_file_path}")

        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        ## Sort page contents by page number    
        sorted_page_contents = dict(
            sorted(page_contents.items(), key=lambda item: item[0])
        )
        return {
            "text": combined_text,
            "page_contents": sorted_page_contents,  # Dictionary mapping page numbers to text
            "flagged_pages": flagged_pages,  # List of flagged page numbers
            "count": total_flagged_count,
            "verified": not early_exit,
        }


if __name__ == "__main__":
    from src.utils.logger import logger
    import json

    # Configuration
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    extractor = GeminiExtractor(GEMINI_API_KEY, OPENAI_API_KEY)

    # Process single PDF file
    input_pdf_path = r"E:\granth-rag-final\granthrag-backend\core\data\internal\005A Kamandakya - Nitisara english translation-10-30.pdf"
    extracted_text = extractor.process_pdf(
        input_pdf_path, threshold=0.22, max_allowed_flags=5, max_workers=4
    )
    output_folder = "extra/data/processed/extracted_text_files"
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.join(
        output_folder, os.path.basename(input_pdf_path).replace(".pdf", ".txt")
    )
    text= extracted_text["text"]

    ## Store the text in text file
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(text)