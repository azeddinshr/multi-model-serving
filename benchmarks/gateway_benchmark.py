import concurrent.futures
import time
import statistics
import requests

URL = "http://localhost:8080/predict"

MODEL_ID = "8ba7b810-9dad-11d1-80b4-00c04fd430c9"

PAYLOAD = {
    "model_id": MODEL_ID,
    "input_data": "This movie was absolutely fantastic."
}

TOTAL_REQUESTS = 40
CONCURRENCY = 8


def send_request(request_id):
    start = time.perf_counter()

    response = requests.post(
        URL,
        json=PAYLOAD,
        timeout=30,
    )

    response.raise_for_status()

    latency = time.perf_counter() - start

    return request_id, latency


print("Gateway benchmark")
print("Requests:", TOTAL_REQUESTS)
print("Concurrency:", CONCURRENCY)
print()

start = time.perf_counter()

latencies = []

with concurrent.futures.ThreadPoolExecutor(
    max_workers=CONCURRENCY
) as executor:

    futures = [
        executor.submit(send_request, i)
        for i in range(TOTAL_REQUESTS)
    ]

    for future in concurrent.futures.as_completed(futures):
        request_id, latency = future.result()
        latencies.append(latency)

total_time = time.perf_counter() - start

latencies_ms = [
    x * 1000
    for x in latencies
]

print(f"Total time: {total_time:.2f} s")

print(
    f"Throughput: "
    f"{TOTAL_REQUESTS / total_time:.2f} requests/s"
)

print(
    f"Average latency: "
    f"{statistics.mean(latencies_ms):.2f} ms"
)

print(
    f"P50 latency: "
    f"{statistics.median(latencies_ms):.2f} ms"
)

print(
    f"P95 latency: "
    f"{sorted(latencies_ms)[int(len(latencies_ms) * 0.95) - 1]:.2f} ms"
)
