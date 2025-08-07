from flask import Flask, render_template, request, jsonify, send_from_directory
# Assuming your video2BQ.py (like v1 you provided) has these functions
import video2BQ # Import the whole module
import pandas as pd # For creating DataFrame in main.py if needed
import json
import threading
import os
import time # For timestamp if not handled by video2BQ.pd2bq
from google.cloud import storage
from google.cloud import bigquery
import logging
import datetime

# Configure basic logging for Flask app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

GCS_BUCKET_NAME = "public-bdoohan-bucket"
BQ_PROJECT_ID = "kiran-cailin"
BQ_DATASET_ID = "spielbergWebapp"
BQ_TABLE_ID = "results_per_store"
BQ_FULL_TABLE_ID = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}"

storage_client = storage.Client()
bq_client = bigquery.Client(project=BQ_PROJECT_ID)

app = Flask(__name__, static_folder='static')

processing_status = {
    "is_processing": False,
    "result": None, # For Gemini's direct output (products or error JSON)
    "current_file": None,
    "bq_row_message": None, # For messages about BQ operations
    "summary": None, # For grounded_upc results
    "related_images": []
}

def get_filename_from_uri(uri):
    if not uri: return "Unknown File"
    try:
        if uri.startswith("gs://"):
            path_after_bucket = '/'.join(uri.split('/')[3:])
            return path_after_bucket if path_after_bucket else uri.split('/')[-1]
        return os.path.basename(uri)
    except Exception: return os.path.basename(uri)

def process_video(video_uri):
    global processing_status
    processing_status["is_processing"] = True
    processing_status["result"] = None
    processing_status["summary"] = None
    processing_status["bq_row_message"] = None
    processing_status["current_file"] = get_filename_from_uri(video_uri)
    processing_status["related_images"] = []

    app.logger.info(f"Starting video processing for: {video_uri} using individual video2BQ functions.")

    gemini_raw_json_string = "" # Initialize to ensure it's always a string

    try:
        # Step 1: Call _generate from video2BQ
        app.logger.info(f"Calling video2BQ._generate for {video_uri}")
        gemini_raw_json_string = video2BQ._generate(video_uri) # This is from video2BQ_v1.py
        app.logger.info(f"Received from video2BQ._generate (first 500 chars): {gemini_raw_json_string[:500]}")
        processing_status["result"] = {"status": "completed_gemini_call", "message": gemini_raw_json_string}

        # Step 2: Parse Gemini's response
        app.logger.info("Calling video2BQ.gemini2json")
        # gemini2json in video2BQ_v1.py does: json.loads(json.loads(json_string)['response'])
        # This assumes gemini_raw_json_string is '{"response": "[...actual_product_json_as_string...]"}'
        # And it returns the parsed list/dict of products.
        products_data = [] # Default to empty list
        parsed_outer_json_for_grounding = {} # For grounding query prep

        try:
            parsed_outer_json_for_grounding = json.loads(gemini_raw_json_string)
            products_data = video2BQ.gemini2json(gemini_raw_json_string) # Expects list or dict
        except json.JSONDecodeError as e:
            app.logger.error(f"Failed to parse JSON from _generate output with gemini2json: {e}. Raw: {gemini_raw_json_string[:200]}")
            # products_data remains empty list
            processing_status["result"]["message"] = json.dumps({"error": "Failed to parse Gemini JSON", "details": str(e), "raw_output": gemini_raw_json_string})


        # Step 3: Create DataFrame
        # products_data should be a list of dicts (products) or a single dict (if one product)
        df_for_bq = pd.DataFrame()
        if isinstance(products_data, list) and products_data:
            df_for_bq = pd.DataFrame(products_data)
        elif isinstance(products_data, dict) and products_data: # Single product
            df_for_bq = pd.DataFrame([products_data])
        
        if df_for_bq.empty:
            app.logger.warning("DataFrame is empty after gemini2json. Creating dummy DataFrame for BQ.")
            # The pd2bq in video2BQ_v1.py will fail if df_for_bq is empty because it tries to set columns.
            # It expects columns like 'Product Name', etc.
            # The `video2BQ_v1.py`'s `pd2bq` also hardcodes `columns` list and applies it.
            # So we need to provide a DataFrame that *might* match those expected columns, or let it fail if it's robust.
            # For safety, if product_data was truly empty (e.g. Gemini returned '{"response":"[]"}'), make one dummy row.
            # Note: video2BQ_v1.py's pd2bq defines its own column list and applies `get_gemini_analysis`
            # The columns for dummy data should align with what `get_gemini_analysis` might expect for query_string.
            dummy_cols = ['Product Name', 'Brand', 'Product Category', 'Package Size/Volume', 'Quantity', 'Price', 'Promotional Details', 'Distinguishing Features']
            dummy_row = {col: "N/A (from main.py)" for col in dummy_cols}
            df_for_bq = pd.DataFrame([dummy_row])
            app.logger.info("Created dummy DataFrame with N/A values.")


        # Step 4: Call pd2bq from video2BQ.
        # This pd2bq from video2BQ_v1.py is expected to:
        #   - Add 'time'
        #   - Add 'store' (hardcoded to "88" in that version)
        #   - Apply get_gemini_analysis to create 'gemini_analysis' column (NEEDS TYPO FIX)
        #   - Set column names to a predefined list
        #   - Write to BigQuery
        app.logger.info(f"Calling video2BQ.pd2bq with DataFrame of shape: {df_for_bq.shape}")
        if not df_for_bq.empty:
             video2BQ.pd2bq(df_for_bq.copy()) # Pass a copy to avoid side effects if pd2bq modifies inplace
             processing_status["bq_row_message"] = "Data processing and BigQuery insertion handled by video2BQ.pd2bq."
        else:
             processing_status["bq_row_message"] = "Skipped BigQuery insertion as DataFrame was empty before pd2bq call."
             app.logger.warning("DataFrame was unexpectedly empty before calling video2BQ.pd2bq.")


        # Step 5: Grounding for UI summary (using parsed_outer_json_for_grounding or df_for_bq)
        grounding_query_ui = ""
        if isinstance(parsed_outer_json_for_grounding, dict) and "response" in parsed_outer_json_for_grounding:
            # Use the inner product string for a more direct grounding if available
            # This matches closer to how generate() from video2BQ_v3 worked for grounding.
            inner_product_json_str_for_grounding = parsed_outer_json_for_grounding["response"]
            if inner_product_json_str_for_grounding and inner_product_json_str_for_grounding != "[]":
                grounding_query_ui = f"Product data summary: {inner_product_json_str_for_grounding}"
            else:
                grounding_query_ui = "No product details in Gemini response for UI summary."
        elif isinstance(parsed_outer_json_for_grounding, dict) and "error" in parsed_outer_json_for_grounding:
             grounding_query_ui = f"Gemini processing error occurred: {parsed_outer_json_for_grounding.get('error','Unknown Gemini error')}"
        else: # Fallback to using the raw string if parsing failed earlier for grounding structure
            grounding_query_ui = f"Raw Gemini output: {gemini_raw_json_string}" if gemini_raw_json_string else "No Gemini output for UI summary."


        if len(grounding_query_ui) > 1900: grounding_query_ui = grounding_query_ui[:1900] + "..."
        app.logger.info(f"Prepared grounding query for UI summary (first 100 chars): {grounding_query_ui[:100]}")
        
        summary_results_text = gemini_raw_json_string#video2BQ.grounded_upc(grounding_query_ui)
        app.logger.info(f"Received UI summary from video2BQ.grounded_upc (first 100): {str(summary_results_text)[:100]}")
        processing_status["summary"] = {"status": "success", "message": summary_results_text}

    except Exception as e:
        app.logger.error(f"Major error in process_video for {video_uri}: {e}", exc_info=True)
        # Ensure gemini_raw_json_string is a string even if _generate failed badly
        error_msg_for_result = gemini_raw_json_string if gemini_raw_json_string else f"Processing error: {str(e)}"
        processing_status["result"] = {"status": "error", "message": error_msg_for_result}
        processing_status["summary"] = {"status": "error", "message": f"Grounding error or skipped: {str(e)}"}
        processing_status["bq_row_message"] = "BQ insertion likely failed due to major processing error."
    finally:
        processing_status["is_processing"] = False
        app.logger.info(f"Video processing finished for: {video_uri}")


# The rest of main.py remains the same as the previous "Sunshine" version:
# / (index route)
# /test
# /static/<path:filename>
# /list_videos (renamed to list_videos_endpoint)
# /list_related_images (renamed to list_related_images_endpoint)
# /bq_preview (renamed to bq_preview_endpoint)
# /process (POST request, renamed to process_request_endpoint)
# /status (renamed to get_status_endpoint)
# if __name__ == '__main__': block

@app.route('/')
def index():
    return render_template('video_index.html')

@app.route('/test')
def test():
    return render_template('test_video.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/list_videos')
def list_videos_endpoint():
    try:
        blobs = storage_client.list_blobs(GCS_BUCKET_NAME)
        video_files = []
        for blob in blobs:
            if blob.name.lower().endswith(('.mp4', '.mov', '.avi', '.wmv', '.mkv')):
                video_files.append(f"gs://{GCS_BUCKET_NAME}/{blob.name}")
        app.logger.info(f"Listed {len(video_files)} videos from GCS bucket {GCS_BUCKET_NAME}.")
        return jsonify({"status": "success", "videos": sorted(video_files, key=lambda x: x.lower())})
    except Exception as e:
        app.logger.error(f"Error listing GCS bucket {GCS_BUCKET_NAME}: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Error listing GCS bucket: {str(e)}"}), 500

@app.route('/list_related_images')
def list_related_images_endpoint():
    video_uri = request.args.get('video_uri')
    if not video_uri or not video_uri.startswith("gs://"):
        app.logger.error(f"Invalid or missing video_uri for /list_related_images: {video_uri}")
        return jsonify({"status": "error", "message": "Invalid or missing video_uri parameter"}), 400

    app.logger.info(f"Listing related images for video: {video_uri}")
    try:
        bucket_name = video_uri.split('/')[2]
        video_path_full = '/'.join(video_uri.split('/')[3:])
        video_dir_path = os.path.dirname(video_path_full)
        video_filename_base = os.path.splitext(os.path.basename(video_path_full))[0]
        
        # Replicate prefix logic from video2BQ_v1's _generate: file_uri.split("gs://")[1].split("/")[1][0:15]
        # This seems to take the first 15 chars of the first path component after bucket name.
        # This might not be ideal if your paths are like "folder/video.mp4" vs "anotherfolder/video.mp4"
        # Let's try a more general approach: use the first part of the filename as prefix in its directory
        
        # A robust way: use the prefixing strategy from video2BQ_v1's _generate:
        # video_prefix = file_uri.split("gs://")[1].split("/")[1][0:15]
        # This takes first 15 chars of the first path component after the bucket.
        # Example: "gs://bucket/folder_long_name/video.mp4" -> prefix = "folder_long_nam"
        # Example: "gs://bucket/video.mp4" -> prefix = "video.mp4"[0:15]
        
        path_part_for_prefix = video_uri.split("gs://")[1].split("/")[1] # e.g., "folder_long_name" or "video.mp4"
        image_prefix_from_video = path_part_for_prefix[:15]

        # The search should be relative to the bucket root if the prefix is from the first path component
        prefix_to_search = image_prefix_from_video
        app.logger.info(f"Derived image prefix (first 15 chars of first path part): '{prefix_to_search}' in bucket '{bucket_name}'")

        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix_to_search)
        
        image_data = []
        for blob in blobs:
            if blob.name.lower().endswith(('.png', '.jpg', '.jpeg')) and blob.name != video_path_full:
                # Further check: ensure the found blob is "semantically" related, e.g. in same "folder"
                # if not video_dir_path or blob.name.startswith(video_dir_path): # Basic check
                public_url = f"https://storage.googleapis.com/{bucket_name}/{blob.name}"
                image_data.append({"url": public_url, "name": os.path.basename(blob.name)})
                if len(image_data) >= 12:
                    app.logger.info("Reached image limit for display (12).")
                    break
        
        app.logger.info(f"Found {len(image_data)} related images for {video_uri} using prefix '{prefix_to_search}'.")
        processing_status["related_images"] = image_data
        return jsonify({"status": "success", "images": image_data})
    except Exception as e:
        app.logger.error(f"Error listing related images for {video_uri}: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Error listing images: {str(e)}"}), 500

@app.route('/bq_preview')
def bq_preview_endpoint():
    try:
        query = f"""
            SELECT *
            FROM `{BQ_FULL_TABLE_ID}`
            ORDER BY time DESC
            LIMIT 5
        """
        app.logger.info(f"Executing BigQuery preview query: {query.strip()}")
        query_job = bq_client.query(query)
        results = query_job.result()
        rows = [dict(row_data_item) for row_data_item in results] # Renamed loop variable

        for row_dict in rows: # Renamed loop variable
            for key, value in row_dict.items():
                if isinstance(value, datetime.datetime) or isinstance(value, datetime.date):
                    row_dict[key] = value.isoformat()
                elif key == 'time' and isinstance(value, float):
                    try:
                        row_dict[key] = datetime.datetime.fromtimestamp(value).isoformat()
                    except TypeError: 
                        row_dict[key] = str(value)
        app.logger.info(f"Successfully fetched {len(rows)} rows for BigQuery preview.")
        return jsonify({"status": "success", "rows": rows})
    except Exception as e:
        app.logger.error(f"Error querying BigQuery table {BQ_FULL_TABLE_ID}: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Error querying BigQuery: {str(e)}"}), 500

@app.route('/process', methods=['POST'])
def process_request_endpoint():
    if processing_status["is_processing"]:
        app.logger.warning("Attempted to process video while another is already processing.")
        return jsonify({"status": "error", "message": "Already processing a video"}), 429

    video_uri = request.json.get('video_uri')
    if not video_uri:
        app.logger.error("No video URI provided in /process request.")
        return jsonify({"status": "error", "message": "No video URI provided"}), 400

    app.logger.info(f"Received request to process video: {video_uri}")
    thread = threading.Thread(target=process_video, args=(video_uri,))
    thread.start()
    return jsonify({"status": "processing_started", "message": f"Processing started for {get_filename_from_uri(video_uri)}"})

@app.route('/status')
def get_status_endpoint():
    status_payload = processing_status.copy()
    if status_payload.get("result") and "message" in status_payload["result"]:
        if not isinstance(status_payload["result"]["message"], str):
            try:
                status_payload["result"]["message"] = json.dumps(status_payload["result"]["message"])
            except TypeError:
                 status_payload["result"]["message"] = str(status_payload["result"]["message"])
    
    if status_payload.get("summary") and "message" in status_payload["summary"]:
        if not isinstance(status_payload["summary"]["message"], str):
            status_payload["summary"]["message"] = str(status_payload["summary"]["message"])
            
    return jsonify(status_payload)

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.logger.info("Starting Flask application with Sunshine UX enhancements...")
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))