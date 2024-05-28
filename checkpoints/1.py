def max_burgers_and_min_cost(N, M):
    # Calculate burgers at minimum cost (5) using friends
    minBurgersWithFriends = min(M // 5, N // 5)
    costWithFriends = minBurgersWithFriends * 5

    # Calculate burgers at original cost (10) without relying on friends
    burgersAt10 = N // 10
    costAt10 = burgersAt10 * 10

    # Determine the maximum number of burgers he can get
    if burgersAt10 > minBurgersWithFriends:
        maxBurgers = burgersAt10
        minCost = costAt10
    else:
        maxBurgers = minBurgersWithFriends
        minCost = costWithFriends

    return maxBurgers, minCost


def solve_burger_cases():
    num_cases = int(input())
    test_cases = []
    for _ in range(num_cases):
        N, M = map(int, input().split())
        test_cases.append((N, M))
    results = []
    for N, M in test_cases:
        results.append(max_burgers_and_min_cost(N, M))
    return results


# Solve the cases
results = solve_burger_cases()
for result in results:
    print(result[0], result[1])