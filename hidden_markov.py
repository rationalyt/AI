import random
import hmmlearn.hmm as hmm
import numpy as np
from scipy import stats as st

random.seed(200418887)
P = [[0 for i in range(4)] for i in range(4)]
B = [[0 for i in range(3)] for i in range(4)]
s = 0

#Task 1

for i in range(4):
    for j in range(4):
        P[i][j] = random.random()
for i in range(4):
    s = 0
    for j in range(4):
        s += P[i][j]
    for j in range(4):
        P[i][j] = P[i][j]/s

for i in range(4):
    for j in range(3):
        B[i][j] = random.random()
for i in range(4):
    s = 0
    for j in range(3): s += B[i][j]
    for j in range(3): B[i][j] = B[i][j]/s
print(P)
print("\n")
print(B)

O,S = [],[0]

while len(O) < 1000:
    s=S[-1]
    r = random.random()
    for i in range(1,4):
        if r <= sum(B[s][0:i]):
            O.append(i-1)
            break
    r = random.random()
    for i in range(1,5):
        if r <= sum(P[s][0:i]):
            S.append(i-1)
            break
print(O)
print("\n")

#Task 2,3

start_prob = np.array([1,0,0,0])
trans_prob = np.array(P)
emit_prob = np.array(B)
sequence = np.array([0,1,2])#,2,0,1,2,2,0,1,2])
obs = np.array(O)

h = hmm.MultinomialHMM(4)
h.n_features = 3
h.startprob_ = start_prob
h.transmat_ = trans_prob
h.emissionprob_ = emit_prob
print(2.7 ** h.score(sequence.reshape(-1,1)))
print(h.predict(sequence.reshape(-1,1)))


#Task 4

model = hmm.MultinomialHMM(4)
model.n_features = 3
model.fit(obs.reshape(-1,1))
print(model.emissionprob_)
print(model.transmat_)
print(model.startprob_)
print("\n Statistics")
print(st.ttest_ind(P,model.transmat_))
print(st.ttest_ind(B,model.emissionprob_))
print(st.ttest_ind(start_prob,model.startprob_))



