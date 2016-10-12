import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math


# ----------------------------------------------------------------------------------------------------------------------
def print_histograms(values, labels, bin_size):

	# Determine overall range of values
	val_min = 99999999.0
	val_max = 0.0
	for vals in values:
		v_min = min(vals)
		v_max = max(vals)
		if v_min < val_min:
			val_min = v_min
		if v_max > val_max:
			val_max = v_max
	val_min = int(math.floor(val_min))
	val_max = int(math.ceil(val_max))

	# Calculate histogram bars for each label
	nr_bins = 0
	lines_array = []
	for vals in values:
		max_length = 0
		lines = []
		for val in range(val_min, val_max + 1, bin_size):
			hits = []
			for v in vals:
				if val <= v < val + bin_size:
					hits.append(v)
			bar = len(hits) * '*'
			line = '[{} - {}] {}'.format(val, val + bin_size - 1, bar)
			if len(line) > max_length:
				max_length = len(line)
			lines.append(line)
		nr_bins = len(lines)
		lines_array.append((lines, max_length))

	# Write header labels
	header = ''
	for i in range(len(lines_array)):
		max_length = lines_array[i][1]
		filler = (max_length - len(labels[i]) + 1) * ' '
		header += labels[i] + filler
	print(header)

	# Print histogram bars
	for i in range(nr_bins):
		line = ''
		for lines, max_length in lines_array:
			if i >= len(lines):
				break
			filler = (max_length - len(lines[i]) + 1) * ' '
			line += lines[i] + filler
		print(line)
	print('\n')
