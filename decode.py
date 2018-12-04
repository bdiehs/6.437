import numpy as np
import random
import csv
import matplotlib.pyplot as plt


alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 
			'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 
			'w', 'x', 'y', 'z', ' ', '.']


all_indices = [i for i in range(28)]

letter_to_ind = {}

for i in range(len(alphabet)):
	letter_to_ind[alphabet[i]] = i

def matrix(el = 0):
	out=[]
	for i in all_indices:
		l=[]
		for j in all_indices:
			l.append(0)
		out.append(l)
	return out

with open('letter_probabilities.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        p = row


with open('letter_transition_matrix.csv') as csvfile:
    reader = csv.reader(csvfile)
    m=[]
    for row in reader:
        m.append(row)

P=[]


M=matrix()

logM=matrix()
for i in all_indices:
    P.append(float(p[i]))
    for j in all_indices:
        M[i][j] = float(m[i][j])
	if M[i][j]!=0:
		logM[i][j]=np.log(M[i][j])


all_diffs = matrix()
for i in all_indices:
	for j in all_indices:
		diffs=[(i, i), (j, j), (i, j), (j, i)]
		for x in all_indices:
			if (x==i or x==j):
				continue
			diffs.append((j, x))
			diffs.append((x, j))

			diffs.append((i, x))
			diffs.append((x, i))
		all_diffs[i][j] = diffs



def index_to_indices(index):
	i = int((2*index + 0.25)**0.5 + 0.5)
	j = index - (i*(i-1))//2
	return (i, j)

"""
Returns an empirical transition map 
"""
def get_emp_transition_map_and_emp_dist(ciphertext):
	emp_transition_map = matrix()
	emp_dist = [0 for i in all_indices]
	emp_dist[letter_to_ind[ciphertext[0]]] += 1
	for i in range(len(ciphertext) - 1):
		emp_dist[letter_to_ind[ciphertext[i+1]]] += 1
		emp_transition_map[letter_to_ind[ciphertext[i]]][letter_to_ind[ciphertext[i+1]]] += 1
	return emp_transition_map, emp_dist



def compute_log_likelihood(perm, ciphertext, emp_transition_map):
	for x in all_indices:
		if alphabet[perm[x]] == ciphertext[0]:
			first = x
			break
	out = np.log(P[first])
	for c in all_indices:
		for d in all_indices:
			if M[d][c] == 0:
				return False
			out += emp_transition_map[perm[c]][perm[d]] * logM[d][c]
	return out


"""
Compute the difference of log likelihood between other and perm
Returns true if perm's log likelihood is infinity, false if other's log likelihood is infinity
and otherwise, the difference
"""
def diff_log_likelihood(perm, other, a, b, ciphertext, emp_transition_map):
	for x in all_indices:
		if alphabet[perm[x]] == ciphertext[0]:
			first = x
		if alphabet[other[x]] == ciphertext[0]:
			second = x
	out = np.log(P[second]) - np.log(P[first])
	diffs = all_diffs[a][b]
	for c, d in diffs:
		if emp_transition_map[perm[c]][perm[d]] == 0:
			continue
		if M[d][c] == 0:
			#always accept since current perm has 0 likelihood
			return True
		out -= emp_transition_map[perm[c]][perm[d]] * logM[d][c]
	for c, d in diffs:
		if emp_transition_map[other[c]][other[d]] == 0:
			continue
		if M[d][c] == 0:
			#never accept since other perm has 0 likelihood
			return False
		out += emp_transition_map[other[c]][other[d]] * logM[d][c]

	return out

"""
First get the permutation such that letters appear in order according to their actual distribution
and empirical distribution
"""
def get_init_perm(emp_transition_map, emp_dist):
	sorted_emp_dist = sorted([(emp_dist[x], x) for x in all_indices])
	d = max(all_indices, key = lambda x: emp_dist[x])
	c = max(all_indices, key = lambda x: emp_transition_map[x][d] / max(float(emp_dist[x]), 1))

	sorted_emp_dist.remove((emp_dist[d], d))
	sorted_emp_dist.remove((emp_dist[c], c))

	sorted_P = sorted((P[x], x) for x in all_indices[:-2])
	perm = [0 for i in all_indices]
	perm[-2] = d
	perm[-1] = c
	#print(c, d)

	for t in range(len(all_indices)-2):
		perm[sorted_P.pop()[1]] = sorted_emp_dist.pop()[1]
	return perm


def sample(perm, ciphertext, emp_transition_map):
	t = random.randint(0, 14*27-1)
	a, b = index_to_indices(t)
	sample = perm[:]
	temp = sample[a]
	sample[a] = sample[b]
	sample[b] = temp
	diff = diff_log_likelihood(perm, sample, a, b, ciphertext, emp_transition_map)

	if diff == True:
		return sample, True
	if diff == False:
		return perm, False
	log_acc_factor = min(0, diff)
	u = random.random()
	if np.log(u) < log_acc_factor:
		return sample, True
	else: 
		return perm, False


def accuracy(x, actual, ciphertext):

	x_dict = {alphabet[i]:alphabet[x[i]] for i in all_indices}
	x_reverse = {alphabet[x[i]]:alphabet[i] for i in all_indices}
	count=0.0
	total=0.0
	for c in ciphertext:
		if c == actual[x_reverse[c]]:
			count += 1
		total += 1
	return count/total
	

def MCMC(ciphertext, emp_transition_map, emp_dist, num_iter = 40000, actual = None):
	dist = {}
	x = get_init_perm(emp_transition_map, emp_dist)
	repeated = 0
	X = [0]
	Y = [accuracy(x, actual, ciphertext)]
	for i in range(num_iter):
		X.append(i+1)
		x, success = sample(x, ciphertext, emp_transition_map)
		if success:
			repeated = 0
			Y.append(accuracy(x, actual, ciphertext))
			if actual != None:
				print("iteration " + str(i) + ": " + str(accuracy(x, actual, ciphertext)))
		else:
			Y.append(Y[-1])
			repeated += 1
			if repeated>5000:
				break
		dist[tuple(x)] = dist.get(tuple(x), 0) + 1

	plt.plot(X, Y)
	plt.show()
	return list(max(dist, key = lambda x: dist[x]))


def decode(ciphertext, output_file_name, actual = None):
	f = open(output_file_name, 'w')
	emp_transition_map, emp_dist = get_emp_transition_map_and_emp_dist(ciphertext)
	best_perm = max([MCMC(ciphertext, emp_transition_map, emp_dist, actual=actual) for i in range(1)], key = lambda x: compute_log_likelihood(x, ciphertext, emp_transition_map))
	inverse_best_perm = {alphabet[best_perm[i]]:alphabet[i] for i in all_indices}
	output = ""
	for c in ciphertext:
		output += inverse_best_perm[c]
	f.write(output)
	f.close()
	return {alphabet[i]:alphabet[best_perm[i]] for i in all_indices}

def main():
	alph_keys = alphabet[:]
	alph_values = alphabet[:]
	cipher = {}
	for i in all_indices:
		a = random.randint(0, len(alph_keys) - 1)
		b = random.randint(0, len(alph_values) - 1)
		cipher[alph_keys.pop(a)] = alph_values.pop(b)
	
	
	# cipher = {' ':'m', '.':'f', 'a':'a', 'c':'t', 'b':'d', 'e':'h', 'd':'i', 'g':'v', 'f':' ', 'i':'c', 'h':'j', 'k':'z', 'j':'w', 'm':'r', 'l':'g', 'o':'e', 'n':'x', 'q':'y', 'p':'o', 's':'u', 'r':'s', 'u':'b', 't':'k', 'w':'n', 'v':'q', 'y':'p', 'x':'l', 'z':'.'} 
	with open('ciphers_and_messages/plaintext_warandpeace.txt') as f:
	    y = list(f.read())[:1000]

	text=""
	for ch in y:
		text+=cipher[ch]
	print(cipher)
	print(decode(text, "out", cipher))


main()

