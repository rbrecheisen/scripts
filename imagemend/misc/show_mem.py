#!/usr/bin/env python

import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

def run():

	if not os.path.isfile('mem.txt'):
		print('Error: could not find mem.txt')
		sys.exit(0)

	mem_max = 0
	mem_use = []

	f = open('mem.txt', 'r')
	for line in f.readlines():
		parts = line.strip().split()
		if mem_max == 0:
			mem_max = parts[3]
		mem_use.append(float(parts[2]))
	f.close()

	plt.plot(mem_use)
	plt.title('Max. memory: {} MB\n'.format(mem_max))
	plt.show()

if __name__ == '__main__':

	run()
