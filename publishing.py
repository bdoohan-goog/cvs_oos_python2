from google.cloud import pubsub_v1
import os
import json

def sendPS(data):

    # TODO(developer): Replace these variables with your project ID and topic ID.
    project_id = "kiran-cailin"
    topic_id = "oos-test-bdoohan"

    # If you're using a service account key, ensure GOOGLE_APPLICATION_CREDENTIALS is set
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your/keyfile.json"

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)

    try:
        # Data must be a bytestring
        #message_data_str = "Hello, Pub/Sub!"

        #JSON
        #data = {"key1": "value1", "key2": 2}
        data_string = json.dumps(data)
        data_bytes = data_string.encode("utf-8")

        future = publisher.publish(topic_path, data=data_bytes)
        # print(future.result())

        # message_data_bytes = message_data_str.encode("utf-8")

        # # Publish the message
        # future = publisher.publish(topic_path, message_data_bytes)
        # print(f"Published message ID: {future.result()}")

        # # Publish a message with attributes
        # future_with_attributes = publisher.publish(
        #     topic_path,
        #     b"Message with attributes!",
        #     origin="python-sample",
        #     username="gcp-user"
        # )
        print(f"Published message ID with attributes: {future.result()}")

    except Exception as e:
        print(f"An error occurred while publishing: {e}")
    
    return(data)

if __name__ == "__main__":
    print("Starting...")
# --- Run the function ---
    i = 10
    while i > 0:
        print(i)
        i -= 1
        data = {"Product": "Easel", 
                "Brand": "Marker", 
                "Category": "Pens", 
                "Size": "NA", 
                "Quantity": 7, 
                "Price": {8.99}, 
                "Status":"In Stock", 
                "Promotions": "None"}
        sendPS(data)
        print("Done")