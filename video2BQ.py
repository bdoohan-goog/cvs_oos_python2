import os
import tempfile
from google.cloud import storage
from typing import List, Dict, Union, Tuple
import cv2
import numpy as np
import pandas as pd
import pandas_gbq
import time
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
from google.cloud import bigquery
from google import genai
from google.genai import types
import base64
import datetime # Import datetime
import json # Keep import, might be useful later
from google.cloud import discoveryengine
from PIL import Image # Import the Image module from Pillow
import io # Import io to handle bytes in memory


#from publishing import sendPS

# Make sure you have the necessary libraries installed:
# pip install --upgrade google-cloud-aiplatform google-cloud-bigquery google-auth pillow
PROMPT_JSON = """ PROMPT: \"Create newlines between each product.

Please analyze each product image using the following structured approach:

Create a detailed inventory document for each product. What specific fields would you need to complete:
- Product Name: [STRING] Official name as displayed on packaging
- Brand: [STRING] Manufacturing company
- Product Category: [STRING] Type of item (food, electronics, clothing, etc.)
- Package Size/Volume: [NUMERIC] Specific measurements when visible
- Quantity: [NUMERIC] Number of units visible
- Price: [NUMERIC] Cost per item/unit as displayed
- Promotional Details: [STRING] Any visible sales, discounts, or special offers
- Distinguishing Features: [STRING] Unique characteristics of the product

Just Reply with a single unified JSON """



def _generate(file_uri="gs://public-bdoohan-bucket/Aisle with yellow tags (1).mp4"):
  #file_uri="gs://public-bdoohan-bucket/Aisle with yellow tags (1).mp4"
  bucket_name = file_uri.split("gs://")[1].split("/")[0]#"public-bdoohan-bucket"
  video_prefix = file_uri.split("gs://")[1].split("/")[1][0:15]
  print(bucket_name)
  print(video_prefix)

  #bucket_name = "public-bdoohan-bucket"
  #video_prefix = "cvs_shelfs0"):

  client = genai.Client(
      vertexai=True,
      project="kiran-cailin",
      location="us-central1",
  )

  video_file_name = file_uri.split("gs://")[1]#f"{video_prefix} (1).mp4"

  storage_client = storage.Client(project="kiran-cailin")
  bucket = storage_client.bucket(bucket_name)

  image_parts = []
  print(f"Listing blobs with prefix: {video_prefix} in bucket: {bucket_name}")
  blobs = bucket.list_blobs(prefix=video_prefix)
  image_count = 0
  for blob in blobs:
    file_extension = blob.name.split('.')[-1].lower()
    if blob.name.startswith(video_prefix) and file_extension in ['png', 'jpg', 'jpeg'] and blob.name != video_file_name:
      mime_type = f"image/{file_extension}"
      if file_extension == 'jpg': # Handle .jpg for image/jpeg mime type
          mime_type = "image/jpeg"
      
      # --- START: Image Rotation Logic ---
      print(f"Rotating image: {blob.name}")
      # Download image into memory
      image_bytes = blob.download_as_bytes()
      
      # Open the image from bytes using Pillow
      img = Image.open(io.BytesIO(image_bytes))
      
      # Rotate the image by 90 degrees clockwise (-90)
      rotated_img = img.rotate(-90, expand=True)
      
      # Save the rotated image to an in-memory byte buffer
      output_buffer = io.BytesIO()
      image_format = 'JPEG' if file_extension in ['jpg', 'jpeg'] else 'PNG'
      rotated_img.save(output_buffer, format=image_format)
      rotated_image_bytes = output_buffer.getvalue()
      # --- END: Image Rotation Logic ---

      # Add the ROTATED image data directly to the parts list
      #image_parts.append(types.Part.from_data(data=rotated_image_bytes, mime_type=mime_type))
      # The 'inline_data' parameter is used in older versions of the library.
      image_parts.append(types.Part(inline_data={'data': rotated_image_bytes, 'mime_type': mime_type}))
      print(f"Added rotated image: {blob.name} with MIME type: {mime_type}")
      image_count += 1

  print(f"Total images found and added: {image_count}")
  if image_count > 3000:
      print("WARNING: Exceeded the recommended limit of 3000 images for Gemini 2.5 Pro. This might be the cause of the INVALID_ARGUMENT error.")

  video_uri = f"gs://{video_file_name}"
  video_part = types.Part.from_uri(file_uri=video_uri, mime_type="video/mp4")
  print(f"Added video: {video_uri} with MIME type: video/mp4")

  model = "gemini-2.5-pro-preview-05-06"

  contents = [
    types.Content(
      role="user",
      parts=[
        types.Part.from_text(text=PROMPT_JSON),
        video_part,
        *image_parts # Unpack the list of image parts here
      ]
    )
  ]

  print(f"Total parts in contents list: {len(contents[0].parts)}")

  generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 1,
    seed = 0,
    max_output_tokens = 8192,
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
    response_mime_type = "application/json",
    response_schema = {"type":"OBJECT","properties":{"response":{"type":"STRING"}}},
    system_instruction=[types.Part.from_text(text="""Be helpful. Do what the user asks for. give details.Just reply in JSON.""")],
  )

  holder = []
  try:
      for chunk in client.models.generate_content_stream(
          model = model,
          contents = contents,
          config = generate_content_config,
      ):
          if chunk.text is not None:
              holder.append(chunk.text)

  except Exception as e:
      print(f"\nAn error occurred during content generation: {e}")
      return '{"response": "[]"}' 

  generated_content = "".join([str(item) for item in holder if item is not None])
  if not generated_content.strip(): 
      return '{"response": "[]"}' 

  try:
      parsed_content = json.loads(generated_content)
      return json.dumps(parsed_content)
  except json.JSONDecodeError as e:
      print(f"Error decoding final generated JSON: {e}")
      return '{"response": "[]"}'

def gemini2json(json_string=""):
  import json
  aa = json.loads(json.loads(json_string)['response'])
  return(aa)

def answer_query_sample(
    project_id: str,
    location: str,
    engine_id: str,
    search_query: str
) -> str:
    print(f"Project ID: {project_id}")
    print(f"Location: {location}")
    print(f"Engine ID: {engine_id}")
    print(f"Search Query: {search_query}")
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )
    client = discoveryengine.ConversationalSearchServiceClient(
        client_options=client_options
    )
    serving_config = f"projects/{project_id}/locations/{location}/collections/default_collection/engines/{engine_id}/servingConfigs/default_serving_config"
    query_understanding_spec = discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec(
        query_rephraser_spec=discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec.QueryRephraserSpec(
            disable=False,
            max_rephrase_steps=1,
        ),
        query_classification_spec=discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec.QueryClassificationSpec(
            types=[
                discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec.QueryClassificationSpec.Type.ADVERSARIAL_QUERY,
                discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec.QueryClassificationSpec.Type.NON_ANSWER_SEEKING_QUERY,
            ]
        ),
    )
    answer_generation_spec = discoveryengine.AnswerQueryRequest.AnswerGenerationSpec(
        ignore_adversarial_query=False,
        ignore_non_answer_seeking_query=False,
        ignore_low_relevant_content=False,
        model_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec.ModelSpec(
            model_version="gemini-2.0-flash-001/answer_gen/v1",
        ),
        prompt_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec.PromptSpec(
            preamble="give the most likely UPC based on this data. Just replay with the UPCs. If none exists, say so",
        ),
        include_citations=True,
        answer_language_code="en",
    )
    request = discoveryengine.AnswerQueryRequest(
        serving_config=serving_config,
        query=discoveryengine.Query(text=search_query),
        session=None,
        query_understanding_spec=query_understanding_spec,
        answer_generation_spec=answer_generation_spec,
    )
    response = client.answer_query(request)
    return response.answer.answer_text

def get_gemini_analysis(row):
    """Concatenates fields of a row and passes to answer_query_sample."""
    project_id = 'kiran-cailin'
    location = "global"
    engine_id = "upc-codes_1747428051210"

    query_string = " + ".join([str(col)+": "+ str(row[col]) for col in row.index if col not in ['time', 'store']])

    try:
        analysis_result = answer_query_sample(
            project_id=project_id,
            location=location,
            engine_id=engine_id,
            search_query=query_string
        )
        return analysis_result.strip()
    except Exception as e:
        print(f"Error analyzing row: {e}")
        return "Analysis failed"

def pd2bq(pandas_df):
  """Writes a pandas DataFrame to BigQuery."""
  table_id = 'spielbergWebapp.results_per_store'
  project_id = 'kiran-cailin'

  columns = ['Product Name', 'Brand', 'Product Category', 'Package Size', 'Quantity', 'Price', 'Promotional Details', 'Features', 'time', 'store', "gemini_analysis"]
  pandas_df["time"] = time.time()
  pandas_df["store"] = "88"
  pandas_df['gemini_analysis'] = pandas_df.apply(get_gemini_analysis, axis=1)
  pandas_df.columns = columns
  pandas_df.reset_index(drop=True, inplace=True)
  pandas_gbq.to_gbq(pandas_df, table_id, project_id=project_id, if_exists='append')
  print(f"DataFrame written to BigQuery table {table_id}")


if __name__ == "__main__":
    my_json = _generate()
    print(my_json)
    my_clean_json = gemini2json(my_json)
    print(my_clean_json)
    df = pd.DataFrame(my_clean_json)
    pd2bq(df)
    print("Done!")