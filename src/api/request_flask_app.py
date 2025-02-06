import requests

API_ENDPOINT = "http://127.0.0.1:5000/api/query"


def request_app(query: str):
    r = requests.get(API_ENDPOINT, json={"query": query})
    r.raise_for_status()
    answer = r.json().get("response", "No answer provided.")
    return answer


if __name__ == "__main__":
    query = "Monitoring the data lake when storing the data."
    answer = request_app(query)
    print(answer)
