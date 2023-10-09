for i in {0..31}:
do
	cpufreq-set -g performance -c $i
done
