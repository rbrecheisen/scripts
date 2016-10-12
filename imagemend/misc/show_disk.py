#!/usr/bin/env python

import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

def run():

	if not os.path.isfile('disk.txt'):
		print('Error: could not find disk.txt')
		sys.exit(0)

	disk_use = []

	f = open('disk.txt', 'r')
	for line in f.readlines():
		parts = line.strip().split()
		disk_use.append(float(parts[0]))
	f.close()

	plt.plot(disk_use)
	plt.title('Disk usage (MB)\n')
	plt.show()

if __name__ == '__main__':

	run()
