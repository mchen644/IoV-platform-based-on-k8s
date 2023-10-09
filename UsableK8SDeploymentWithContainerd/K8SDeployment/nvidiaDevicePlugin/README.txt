#Note that you MUST add a label: gpu: "true" in the node with gpus and you want to use the gpu
kubectl label nodes gpu1 gpu=true

# Deploy nvidia-device-plugin
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update

# setup time-slicing virtualization
helm upgrade -i nvdp nvdp/nvidia-device-plugin \
    --version=0.14.0 \
    --namespace nvidia-device-plugin \
    --create-namespace \
    --set-file config.map.config=plugin-cm.yaml
