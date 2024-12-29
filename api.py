from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import logging
from test import process_image  # Assuming process_image is the function in test.py that handles OCR
import psutil

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_memory_usage():
    # Get the process ID of the current process
    process = psutil.Process(os.getpid())
    # Get memory usage in MB
    memory_usage = process.memory_info().rss / (1024 * 1024)
    print(f"Memory usage: {memory_usage:.2f} MB")

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Ensure the temp directory exists
        temp_dir = "../temp"
        os.makedirs(temp_dir, exist_ok=True)

        # Save the uploaded file
        file_location = os.path.join(temp_dir, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Process the image using the existing OCR functionality
        log_memory_usage()
        result = process_image(file_location)

        # Clean up the saved file
        os.remove(file_location)

        logger.info(f"Detected text: {result['detected_text']}")

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)