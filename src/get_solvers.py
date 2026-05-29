from modules import Qubo

if __name__ == "__main__":
    print(f"{'Solver Name':<60} | {'Type':<15}")
    print("-" * 60)

    for name, type in Qubo.get_solvers():        
        print(f"{name:<60} | {type:<15}")