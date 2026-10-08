records = []
all_scores = set()

if __name__ == '__main__':
    for _ in range(int(input())):
        name = input()
        score = float(input())
        all_scores.add(score)
        records.append([name, score])

    scrs_sorted = all_scores.sort()
    scnd_least = scrs_sorted[1]
    print(scnd_least)
