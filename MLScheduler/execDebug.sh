go build .
kubectl -n default cp Scheduler scheduler-5746578798-d9zdm:/scheduler/.
bash deleteDETFUSION.sh
