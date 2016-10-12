#!/bin/bash

dir=$1

if [ "${dir}" == "" ]; then
	echo ""
	echo "Usage: watch_perf.sh <data dir>"
	echo ""
	echo "  <data dir> : Data directory to monitor for disk usage."
	echo ""
	exit 1
fi

rm -f mem.txt
rm -f disk.txt
rm -f stop.txt

while [ ! -f stop.txt ]; do
  free -m | grep buffers/cache >> mem.txt
  du ${dir} -s -m >> disk.txt
  sleep 5
done

echo "Done"
