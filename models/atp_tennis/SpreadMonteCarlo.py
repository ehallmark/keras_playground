import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)



def simulate_match(best_of):
    def simulate_set(odds):
        s1, s2 = 0, 0
        for i in range(13):
            if s1 == 7 or s2 == 7 or (s1 == 6 and s2 <= 4) or (s2 == 6 and s1 <= 4):
                break
            odds = 1.0-odds
            if np.random.rand(1) < odds:
                s1 += 1
            else:
                s2 += 1
        return s1, s2

    m1, m2 = 0, 0
    spread = 0
    best_to = int(best_of/2)+1
    odds = 0.4 + np.random.rand(1)*0.2
    for i in range(best_of):
        if m1 >= best_to or m2 >= best_to:
            break
        if abs(spread) % 2 == 1:
            odds = 1.0-odds
        s1, s2 = simulate_set(odds)
        spread += s1 - s2
        if s1 > s2:
            m1 += 1
        else:
            m2 += 1
    return m1, m2, spread


x = []
x_3 = []
for i in range(20000):
    _, _, spread = simulate_match(5)
    x.append(spread)
    x.append(-spread)
    _, _, spread = simulate_match(3)
    x_3.append(spread)
    x_3.append(-spread)


probabilities5 = {}
probabilities3 = {}

for i in range(19):
    probabilities5[i] = 0.0
    probabilities5[-i] = 0.0

for i in range(13):
    probabilities3[i] = 0.0
    probabilities3[-i] = 0.0

for p in x:
    probabilities5[int(p)] += 1
for p in x_3:
    probabilities3[int(p)] += 1

for k in probabilities3:
    probabilities3[k] /= len(x_3)

for k in probabilities5:
    probabilities5[k] /= len(x)

probabilities5_under = {}
probabilities3_under = {}
probabilities5_over = {}
probabilities3_over = {}

for k in probabilities3:
    probabilities3_over[k] = 0.
    probabilities3_under[k] = 0.
    for j in probabilities3:
        if j <= k:
            probabilities3_under[k] += probabilities3[j]
        if j >= k:
            probabilities3_over[k] += probabilities3[j]

for k in probabilities5:
    probabilities5_over[k] = 0.
    probabilities5_under[k] = 0.
    for j in probabilities5:
        if j <= k:
            probabilities5_under[k] += probabilities5[j]
        if j >= k:
            probabilities5_over[k] += probabilities5[j]


def probability_beat(spread, grand_slam=False):
    if spread > 0:
        if grand_slam:
            return probabilities5_under[int(spread)]
        else:
            return probabilities3_under[int(spread)]
    else:
        if grand_slam:
            return probabilities5_over[round(abs(spread))]
        else:
            return probabilities3_over[round(abs(spread))]


if __name__ == '__main__':
    plt.figure(figsize=(10, 10))
    ax1 = plt.hist(x, range=(-20, 20), bins=40, label='Spread (5)',
                             histtype="step", lw=2)
    ax2 = plt.hist(x_3, range=(-20, 20), bins=40, label='Spread (3)',
                             histtype="step", lw=2)
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    print('Beat 5', probability_beat(5))

    print('Beat 3.5', probability_beat(3.5))

    print('Beat -3', probability_beat(-3))

    print('Beat -5.0', probability_beat(-5.5))
