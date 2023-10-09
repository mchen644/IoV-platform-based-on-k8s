for i in (seq 1:10):
	echo %i;
	kubectl -n kube-system exec --it 0;
end

echo Start to Init cluster's service account
kubectl create sa cluster-rolebinding ca-certificate -s RSA 

echo 
