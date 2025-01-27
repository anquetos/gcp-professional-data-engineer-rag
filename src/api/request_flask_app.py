import textwrap

import requests

API_ENDPOINT = "http://127.0.0.1:5000/api/query"


def request_app(query: str):
    r = requests.get(API_ENDPOINT, json={"query": query})
    r.raise_for_status()
    answer = r.json().get("response", "No answer provided.")
    return answer


if __name__ == "__main__":
    query = """
An air-quality research facility monitors the quality of the air and alerts of possible high air pollution in a region. The facility receives event data from 25,000 sensors every 60 seconds. Event data is then used for time-series analysis per region. Cloud experts suggested using BigTable for storing event data.
What will you design the row key for each even in BigTable?

A. Use event’s timestamp as row key.
B. Use combination of sensor ID with timestamp as sensorID-timestamp.
C. Use combination of sensor ID with timestamp as timestamp-sensorID.
D. Use sensor ID as row key.
"""
    answer = request_app(query)
    print(textwrap.fill(answer, 100))
