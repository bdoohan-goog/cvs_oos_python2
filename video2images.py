# This file generates images of all out opf stock instances and saves them to GCS

import os
import tempfile
from google.cloud import storage
from typing import List, Dict, Union, Tuple
import cv2
import numpy as np
#from IPython.display import display
#from matplotlib import pyplot as plt
from google import genai
from google.genai import types
import base64
import datetime # Import datetime
import json # Keep import, might be useful later

PROMPT_JSON = """ PROMPT: \"Create newlines between each product.

Please analyze each product image using the following structured approach:

Create a detailed inventory document for each product. What specific fields would you need to complete:
- Product Name: Official name as displayed on packaging
- Brand: Manufacturing company
- Product Category: Type of item (food, electronics, clothing, etc.)
- Package Size/Volume: Specific measurements when visible
- Quantity: Number of units visible
- Price: Cost per item/unit as displayed
- Promotional Details: Any visible sales, discounts, or special offers
- Distinguishing Features: Unique characteristics of the product

Reply with a single unified JSON """

from google import genai
from google.genai import types
import base64
import datetime # Import datetime
import json # Keep import, might be useful later

PROMPT_JSON = """ PROMPT: \"Create newlines between each product.

Please analyze each product image using the following structured approach:

Create a detailed inventory document for each product. What specific fields would you need to complete:
- Product Name: Official name as displayed on packaging
- Brand: Manufacturing company
- Product Category: Type of item (food, electronics, clothing, etc.)
- Package Size/Volume: Specific measurements when visible
- Quantity: Number of units visible
- Price: Cost per item/unit as displayed
- Promotional Details: Any visible sales, discounts, or special offers
- Distinguishing Features: Unique characteristics of the product

Reply with a single unified JSON """

PROMPT_OOS = """ Just reply with a JSON file with the timestamps (number of second into the movie) when a products is Out of stock or missing. 
        Separate the timestamps by &*&. No other text.
        for example: {"timestamps":"5.0 &*& 113.0"}
        """""

from google import genai
from google.genai import types
import base64

def _generate(file_uri="gs://public-bdoohan-bucket/Aisle with yellow tags (1).mp4"):
  client = genai.Client(
      vertexai=True,
      project="kiran-cailin",
      location="global",
  )

  msg1_video1 = types.Part.from_uri(
      file_uri="gs://public-bdoohan-bucket/Aisle with yellow tags (1).mp4",
      mime_type="video/mp4",
  )

  model = "gemini-2.0-flash-001"
  contents = [
    types.Content(
      role="user",
      parts=[
        types.Part.from_text(text=PROMPT_OOS),
        msg1_video1
      ]
    ),
  ]

  generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 1,
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
  )

  holder = []

  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    print(chunk.text, end="")
    holder.append(chunk.text)
  return "".join(holder)

def list_video_files_in_gcs(gcs_uri: str) -> List[str]:
    """
    Lists all video files (.mov or .mp4) within a specified Google Cloud Storage URI.

    Args:
        gcs_uri: The Google Cloud Storage URI (e.g., 'gs://bucket_name/prefix/').
                 Can be a bucket root ('gs://bucket_name/') or a specific folder.

    Returns:
        A list of GCS URIs (e.g., 'gs://bucket_name/path/to/video.mp4')
        for all .mov or .mp4 files found.

    Raises:
        ValueError: If the gcs_uri is not a valid GCS URI.
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError("URI must be a valid GCS URI starting with 'gs://'")

    # Remove 'gs://' prefix and split into bucket and prefix/folder
    path = gcs_uri[5:]
    parts = path.split('/', 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ''

    # Ensure prefix ends with '/' if it's not empty, to treat it as a folder
    if prefix and not prefix.endswith('/'):
        prefix += '/'

    storage_client = storage.Client()
    video_files = []

    try:
        bucket = storage_client.bucket(bucket_name)

        # List blobs with the given prefix
        # The 'prefix' argument filters results to blobs whose names begin with this prefix.
        # This is efficient as it leverages GCS's indexing.
        blobs = bucket.list_blobs(prefix=prefix)

        for blob in blobs:
            # Check if the blob name ends with .mov or .mp4 (case-insensitive)
            if blob.name.lower().endswith(('.mov', '.mp4')):
                video_files.append(f"gs://{bucket_name}/{blob.name}")

    except Exception as e:
        print(f"An error occurred: {e}")
        # Depending on your error handling strategy, you might want to re-raise or return an empty list
        return []

    return video_files


def extract_frames_from_gcs_video(
    gcs_uri: str,
    timestamps: List[float],
    gcs_output_folder: str,  # New argument for GCS output folder
    display_in_jupyter: bool = True
) -> Dict[float, str]:
    """
    Extract frames from a video stored in Google Cloud Storage at specific timestamps,
    save them to a GCS folder, and optionally display them in a Jupyter notebook.

    Args:
        gcs_uri: Google Cloud Storage URI (e.g., 'gs://bucket_name/path/to/video.mp4')
        timestamps: List of timestamps in seconds where frames should be extracted
        gcs_output_folder: Google Cloud Storage folder URI to save extracted frames
                           (e.g., 'gs://your-bucket/path/to/output_folder')
        display_in_jupyter: Whether to display the extracted frames in the notebook

    Returns:
        Dictionary mapping timestamps to saved GCS image URIs

    Example:
        frame_uris = extract_frames_from_gcs_video(
            'gs://my-bucket/videos/sample.mp4',
            [10.5, 25.0, 37.2],
            'gs://my-bucket/extracted_frames/'
        )
    """
    # Parse GCS URIs
    if not gcs_uri.startswith("gs://"):
        raise ValueError("Input video URI must be a valid GCS URI starting with 'gs://'")
    if not gcs_output_folder.startswith("gs://"):
        raise ValueError("GCS output folder URI must be a valid GCS URI starting with 'gs://'")

    # Extract video filename for naming the frames
    video_filename = os.path.splitext(gcs_uri.split('/')[-1])[0]

    # Parse input video GCS URI
    input_path_parts = gcs_uri[5:].split('/')
    input_bucket_name = input_path_parts[0]
    input_blob_name = '/'.join(input_path_parts[1:])

    # Parse output folder GCS URI
    output_path_parts = gcs_output_folder[5:].split('/')
    output_bucket_name = output_path_parts[0]
    output_prefix = '/'.join(output_path_parts[1:])
    if not output_prefix.endswith('/') and output_prefix: # Ensure it ends with a slash if not empty
        output_prefix += '/'

    storage_client = storage.Client()

    # Create a temporary file to download the video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file_path = temp_file.name
    temp_file.close()

    try:
        # Download the video from GCS
        print(f"Downloading video from {gcs_uri} to {temp_file_path}...")
        input_bucket = storage_client.bucket(input_bucket_name)
        input_blob = input_bucket.blob(input_blob_name)
        input_blob.download_to_filename(temp_file_path)
        print("Download complete.")

        # Open the video file
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {temp_file_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = video_length / fps

        print(f"Video duration: {duration:.2f}s, FPS: {fps}")

        # Validate timestamps
        valid_timestamps = [t for t in timestamps if 0 <= t <= duration]
        if len(valid_timestamps) < len(timestamps):
            print(f"Warning: Some timestamps are outside the video duration (0-{duration:.2f}s)")

        # Extract frames at specific timestamps and upload them to GCS
        frame_uris = {}
        output_bucket = storage_client.bucket(output_bucket_name)

        # Temporary directory for local saving if display_in_jupyter is True
        temp_display_dir = None
        if display_in_jupyter:
            temp_display_dir = tempfile.mkdtemp()

        for timestamp in valid_timestamps:
            # Convert timestamp to frame number
            frame_number = int(timestamp * fps)

            # Set the frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # Read the frame
            ret, frame = cap.read()
            if ret:
                # Generate filename for the saved frame in GCS
                frame_filename = f"{video_filename}_frame_{timestamp:.2f}s.jpg"
                gcs_frame_blob_name = f"{output_prefix}{frame_filename}"
                gcs_frame_uri = f"gs://{output_bucket_name}/{gcs_frame_blob_name}"

                # Encode the frame to JPEG in memory
                is_success, buffer = cv2.imencode(".jpg", frame)
                if not is_success:
                    print(f"Error: Could not encode frame at timestamp {timestamp:.2f}s")
                    continue

                # Upload the frame to GCS
                blob = output_bucket.blob(gcs_frame_blob_name)
                blob.upload_from_string(buffer.tobytes(), content_type="image/jpeg")
                frame_uris[timestamp] = gcs_frame_uri
                print(f"Uploaded frame at {timestamp:.2f}s to {gcs_frame_uri}")

                # If displaying in Jupyter, save a local copy temporarily
                if display_in_jupyter and temp_display_dir:
                    local_display_path = os.path.join(temp_display_dir, frame_filename)
                    cv2.imwrite(local_display_path, frame)
                    # Update frame_uris to local paths for display_frames_in_jupyter
                    # This is a bit of a hack, but makes display_frames_in_jupyter reusable
                    frame_uris[timestamp] = local_display_path
            else:
                print(f"Warning: Could not extract frame at timestamp {timestamp:.2f}s")

        cap.release()

        # Display frames in Jupyter if requested (uses the temporarily saved local copies)
        if display_in_jupyter and frame_uris:
            # We need to filter frame_uris to only include paths that were actually saved locally
            local_frame_paths_for_display = {ts: path for ts, path in frame_uris.items() if path.startswith(temp_display_dir)}
            display_frames_in_jupyter(local_frame_paths_for_display)

        # Revert frame_uris to GCS URIs before returning if they were temporarily local for display
        final_frame_uris = {}
        for timestamp, path in frame_uris.items():
            if path.startswith(temp_display_dir):
                # Construct the GCS URI again
                frame_filename = os.path.basename(path)
                final_frame_uris[timestamp] = f"gs://{output_bucket_name}/{output_prefix}{frame_filename}"
            else:
                final_frame_uris[timestamp] = path

        return final_frame_uris

    finally:
        # Clean up temporary video file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        # Clean up temporary display directory
        if 'temp_display_dir' in locals() and temp_display_dir and os.path.exists(temp_display_dir):
            import shutil
            shutil.rmtree(temp_display_dir)


def display_frames_in_jupyter(frame_paths: Dict[float, str]):
    """
    Display extracted frames in a Jupyter notebook.
    This function still expects local paths to display images.

    Args:
        frame_paths: Dictionary mapping timestamps to local image file paths
    """
    if not frame_paths:
        print("No frames to display.")
        return

    num_frames = len(frame_paths)

    # Set up the matplotlib figure
    plt.figure(figsize=(15, 5 * ((num_frames + 2) // 3)))

    # Sort timestamps for sequential display
    sorted_timestamps = sorted(frame_paths.keys())

    for i, timestamp in enumerate(sorted_timestamps):
        frame_path = frame_paths[timestamp]
        if not os.path.exists(frame_path):
            print(f"Warning: Local file not found for display: {frame_path}")
            continue

        # Read the image using OpenCV (which reads in BGR format)
        frame = cv2.imread(frame_path)
        # Convert BGR to RGB for proper display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create subplot
        plt.subplot(((num_frames + 2) // 3), 3, i + 1)
        plt.imshow(frame_rgb)
        plt.title(f"Timestamp: {timestamp:.2f}s")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

import os
import tempfile
from google.cloud import storage
from typing import List, Dict, Union, Tuple
import cv2
import numpy as np
# from IPython.display import display
# from matplotlib import pyplot as plt


def extract_frames_from_gcs_video(
    gcs_uri: str,
    timestamps: List[Union[float, int, str]], # Updated type hint to accept str
    gcs_output_folder: str,
    display_in_jupyter: bool = True
) -> Dict[float, str]:
    """
    Extract frames from a video stored in Google Cloud Storage at specific timestamps,
    save them to a GCS folder, and optionally display them in a Jupyter notebook.

    Args:
        gcs_uri: Google Cloud Storage URI (e.g., 'gs://bucket_name/path/to/video.mp4')
        timestamps: List of timestamps in seconds where frames should be extracted.
                    Can contain floats, integers, or strings that can be converted to numbers.
        gcs_output_folder: Google Cloud Storage folder URI to save extracted frames
                           (e.g., 'gs://your-bucket/path/to/output_folder')
        display_in_jupyter: Whether to display the extracted frames in the notebook

    Returns:
        Dictionary mapping timestamps (as floats) to saved GCS image URIs

    Example:
        frame_uris = extract_frames_from_gcs_video(
            'gs://my-bucket/videos/sample.mp4',
            [10.5, "25.0", 37.2], # Example with mixed types
            'gs://my-bucket/extracted_frames/'
        )
    """
    # Parse GCS URIs
    if not gcs_uri.startswith("gs://"):
        raise ValueError("Input video URI must be a valid GCS URI starting with 'gs://'")
    if not gcs_output_folder.startswith("gs://"):
        raise ValueError("GCS output folder URI must be a valid GCS URI starting with 'gs://'")

    # Extract video filename for naming the frames
    video_filename = os.path.splitext(gcs_uri.split('/')[-1])[0]

    # Parse input video GCS URI
    input_path_parts = gcs_uri[5:].split('/')
    input_bucket_name = input_path_parts[0]
    input_blob_name = '/'.join(input_path_parts[1:])

    # Parse output folder GCS URI
    output_path_parts = gcs_output_folder[5:].split('/')
    output_bucket_name = output_path_parts[0]
    output_prefix = '/'.join(output_path_parts[1:])
    if not output_prefix.endswith('/') and output_prefix: # Ensure it ends with a slash if not empty
        output_prefix += '/'

    storage_client = storage.Client()

    # Create a temporary file to download the video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file_path = temp_file.name
    temp_file.close()

    try:
        # Download the video from GCS
        print(f"Downloading video from {gcs_uri} to {temp_file_path}...")
        input_bucket = storage_client.bucket(input_bucket_name)
        input_blob = input_bucket.blob(input_blob_name)
        input_blob.download_to_filename(temp_file_path)
        print("Download complete.")

        # Open the video file
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {temp_file_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = video_length / fps

        print(f"Video duration: {duration:.2f}s, FPS: {fps}")

        # --- FIX APPLIED HERE: Validate and convert timestamps ---
        processed_timestamps = []
        for t_val in timestamps:
            try:
                # Attempt to convert each timestamp to a float
                t = float(t_val)
                processed_timestamps.append(t)
            except (ValueError, TypeError):
                print(f"Warning: Timestamp '{t_val}' could not be converted to a number. Skipping this timestamp.")
        # --- END FIX ---

        # Filter timestamps to be within video duration
        valid_timestamps = [t for t in processed_timestamps if 0 <= t <= duration]
        if len(valid_timestamps) < len(processed_timestamps):
            print(f"Warning: Some timestamps are outside the video duration (0-{duration:.2f}s).")
        if not valid_timestamps:
            print("No valid timestamps found to extract frames.")
            return {}


        # Extract frames at specific timestamps and upload them to GCS
        frame_uris = {}
        output_bucket = storage_client.bucket(output_bucket_name)

        # Temporary directory for local saving if display_in_jupyter is True
        temp_display_dir = None
        if display_in_jupyter:
            temp_display_dir = tempfile.mkdtemp()

        for timestamp in valid_timestamps:
            # Convert timestamp to frame number
            frame_number = int(timestamp * fps)

            # Set the frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # Read the frame
            ret, frame = cap.read()
            if ret:
                # Generate filename for the saved frame in GCS
                frame_filename = f"{video_filename}_frame_{timestamp:.2f}s.jpg"
                gcs_frame_blob_name = f"{output_prefix}{frame_filename}"
                gcs_frame_uri = f"gs://{output_bucket_name}/{gcs_frame_blob_name}"

                # Encode the frame to JPEG in memory
                is_success, buffer = cv2.imencode(".jpg", frame)
                if not is_success:
                    print(f"Error: Could not encode frame at timestamp {timestamp:.2f}s. Skipping.")
                    continue

                # Upload the frame to GCS
                blob = output_bucket.blob(gcs_frame_blob_name)
                blob.upload_from_string(buffer.tobytes(), content_type="image/jpeg")
                frame_uris[timestamp] = gcs_frame_uri
                print(f"Uploaded frame at {timestamp:.2f}s to {gcs_frame_uri}")

                # If displaying in Jupyter, save a local copy temporarily
                if display_in_jupyter and temp_display_dir:
                    local_display_path = os.path.join(temp_display_dir, frame_filename)
                    cv2.imwrite(local_display_path, frame)
                    # Update frame_uris to local paths for display_frames_in_jupyter
                    # This is a hack, but makes display_frames_in_jupyter reusable
                    frame_uris[timestamp] = local_display_path
            else:
                print(f"Warning: Could not extract frame at timestamp {timestamp:.2f}s.")

        cap.release()

        # Display frames in Jupyter if requested (uses the temporarily saved local copies)
        if display_in_jupyter and frame_uris:
            # We need to filter frame_uris to only include paths that were actually saved locally
            # This handles cases where some timestamps might have failed to extract
            local_frame_paths_for_display = {ts: path for ts, path in frame_uris.items() if temp_display_dir and path.startswith(temp_display_dir)}
            if local_frame_paths_for_display:
                display_frames_in_jupyter(local_frame_paths_for_display)
            else:
                print("No frames were successfully saved locally for display.")

        # Revert frame_uris to GCS URIs before returning if they were temporarily local for display
        final_frame_uris = {}
        for timestamp, path in frame_uris.items():
            if temp_display_dir and path.startswith(temp_display_dir):
                # Construct the GCS URI again
                frame_filename = os.path.basename(path)
                final_frame_uris[timestamp] = f"gs://{output_bucket_name}/{output_prefix}{frame_filename}"
            else:
                final_frame_uris[timestamp] = path

        return final_frame_uris

    finally:
        # Clean up temporary video file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        # Clean up temporary display directory
        if 'temp_display_dir' in locals() and temp_display_dir and os.path.exists(temp_display_dir):
            import shutil
            shutil.rmtree(temp_display_dir)


def display_frames_in_jupyter(frame_paths: Dict[float, str]):
    """
    Display extracted frames in a Jupyter notebook.
    This function still expects local paths to display images.

    Args:
        frame_paths: Dictionary mapping timestamps to local image file paths
    """
    if not frame_paths:
        print("No frames to display.")
        return

    num_frames = len(frame_paths)

    # Set up the matplotlib figure
    plt.figure(figsize=(15, 5 * ((num_frames + 2) // 3)))

    # Sort timestamps for sequential display
    sorted_timestamps = sorted(frame_paths.keys())

    for i, timestamp in enumerate(sorted_timestamps):
        frame_path = frame_paths[timestamp]
        if not os.path.exists(frame_path):
            print(f"Warning: Local file not found for display: {frame_path}. Skipping.")
            continue

        # Read the image using OpenCV (which reads in BGR format)
        frame = cv2.imread(frame_path)
        # Convert BGR to RGB for proper display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create subplot
        plt.subplot(((num_frames + 2) // 3), 3, i + 1)
        plt.imshow(frame_rgb)
        plt.title(f"Timestamp: {timestamp:.2f}s")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Example usage in Jupyter notebook

# Make sure to set up authentication:
# !export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
# or in Jupyter:
# import os
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your/service-account-key.json"

# Replace with your actual GCS video URI and output folder
# video_uri = "gs://your-input-bucket/path/to/video.mp4"
# gcs_output_folder_uri = "gs://your-output-bucket/path/to/output_frames/"

# Example timestamps: Notice the "8.0" is a string to demonstrate the fix
# timestamps = [7.0, "8.0", 14.0, "invalid_time"] # Try with an invalid string too!

# Extract and display frames, saving to GCS
# frame_uris = extract_frames_from_gcs_video(
#     video_uri,
#     timestamps,
#     gcs_output_folder_uri,
#     display_in_jupyter=True
# )

# print(f"Extracted {len(frame_uris)} frames:")
# for ts, uri in frame_uris.items():
#     print(f"  Timestamp {ts:.2f}s: {uri}")




# Example usage in Jupyter notebook
if __name__ == "__main__":
    # Make sure to set up authentication:
    # !export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
    # or in Jupyter:
    # import os
    import json 
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your/service-account-key.json"

    # Replace with your actual GCS video URI and output folder
    video_uri = "gs://public-bdoohan-bucket"
    print(f"Listing video files in {video_uri}...")
    video_files = list_video_files_in_gcs(gcs_uri=video_uri)
    print(video_files)
    for file_ in video_files:
        print(file_)
        timestamps_string = _generate(file_)
        timestampsDict = json.loads(timestamps_string.replace("```json", "").replace("```", ""))
        print(timestampsDict)
        if isinstance(timestampsDict, dict):
            print("Dictionary!")
            timestamps = timestampsDict["timestamps"].split("&*&")
            print(timestamps)
        else:
            print("String!")
            timestamps = timestampsDict.split("&*&")
            print(timestamps)

        gcs_output_folder_uri = video_uri = file_
        #timestamps = [7.0, 8.0, 14.0]  # Timestamps in seconds
        # Extract and display frames, saving to GCS
        frame_uris = extract_frames_from_gcs_video(
            video_uri,
            timestamps,
            gcs_output_folder_uri,
            display_in_jupyter=False
        )

        print(f"Extracted {len(frame_uris)} frames:")
        for ts, uri in frame_uris.items():
            print(f"  Timestamp {ts:.2f}s: {uri}")