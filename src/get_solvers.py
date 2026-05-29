import os
from dotenv import load_dotenv
from dwave.cloud import Client

load_dotenv()

token = os.getenv("API_TOKEN")
endpoint = "https://cloud.dwavesys.com/sapi"

def get_solvers():
    client = Client.from_config(token=token)
    solvers = client.get_solvers()

    return [(s.name, s.properties.get('category', 'unknown').upper()) for s in solvers]

if __name__ == "__main__":
    print(f"{'Solver Name':<60} | {'Type':<15}")
    print("-" * 60)

    for name, type in get_solvers():        
        print(f"{name:<60} | {type:<15}")