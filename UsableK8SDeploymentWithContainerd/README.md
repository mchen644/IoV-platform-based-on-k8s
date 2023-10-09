# K8SDeployment  
A usable K8S Deployment With containerd  
  
# In container making  
Edit buildContainer.sh set your buildctl's location. If you use buildkit.  
Edit buildContainer.sh to set your container's name and other metainfo.  
Put Dockerfile in the same path with buildContainer.sh  
Run bash buildContainer.sh  
Edit your username, password and any other personal info of dockerhub in pullToDockerHub.sh  
Run pullToDockerHub.sh  
  
# In K8s Deploy  
To Download necessary image(kube-system) for k8s, you can run pullImage.sh  
Note: If you use containerd, replace docker pull and docker tag with corresponding command, like ctr i pull, ctr i tag  
This command can let you download images in China Mainland  

Before running:
Edit echobackDeployment.yaml to set your image and container port(Your Http server listening port)  
Edit ingress.yaml to set your http root path in field "paths.path". / mains root is /  
Edit kubeadm-config.yaml advertiseAddress field to <your master node's IP address>. Note: serviceSubnet should not be the same subnet with your node(Virtual Subnet)  

Run kubeadm init --config kubeadm-config.yaml in master node  
Run export KUBECONFIG=/etc/kubernetes/admin.conf in master node  
Run kubectl apply -f calico.yaml in master node  
Use kubeadm join to join slave node, run this in slave node  
Use kubectl get node to see status, After all slave nodes joining  
Run bash afterAddSlaveNode.sh  
Then use curl "192.168.244.130:30080/\<Your Path\>?\<Your Args\>" to test.  
![未命名文件_1667205268898_0](https://user-images.githubusercontent.com/47754424/198976839-7344830d-6f28-4457-857a-e9712bbcccd5.png)

