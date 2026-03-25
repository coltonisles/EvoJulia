from fastapi import FastAPI, UploadFile, File
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from image_preprocessor import load_and_process

#Initialize webserver
app = FastAPI()

#api endpoint to handle the image upload and preprocessing
@app.post("/api/upload")
#file is the name of the form field in the HTML
#UploadFile is a python class that represents the file that was uploaded
#File is a decorator that tells fastAPI that the file is required
async def handle_upload(file: UploadFile = File(...)):
    try:
        #async reading of raw binary; used to build the jpg
        content = await file.read()

        temp_path = "temp_target.jpg"
        #writes the binary data to a temporary file; 'wb' = 'write binary'
        with open(temp_path, "wb") as f:
            f.write(content)

        #grey scale pixel array
        processed_arr = load_and_process(temp_path)

        #delete the temporary file
        os.remove(temp_path)

        return{
            "message": "Image successfully uploaded and precessed",
            "filename": file.filename,
            "processed_shape": processed_arr.shape
        }
    except Exception as e:
        #JSONResponse is a class that converts the python dictionary into a JSON string and assigns a adds the HTTP headers like 'status'
        return JSONResponse(status_code=500, content={"message": str(e)})



#Mount the frontend directory so the browser can load the HTML and js
#'/' is the root url, fastAPI looks for python code to run when mounting but StaticFiles() tells it to look in the '../frontend' folder for the html file
app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")
