package worker_pool

import (
	"Scheduler/utils"
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/metrics/pkg/client/clientset/versioned"
)

var containerMap = map[string]string{
	"mcmot":  "docker.io/luoxiao23333/task_mcmot:v0",
	"slam":   "docker.io/luoxiao23333/task_slam:v0",
	"fusion": "docker.io/luoxiao23333/task_fusion:v0",
	"det":    "docker.io/luoxiao23333/task_det:v0",
}

type podInfo struct {
	TaskName string
	NodeName string
	HostName string
}

var PodsInfo = map[string]podInfo{
	"fusion-gpu1": {
		TaskName: "fusion",
		NodeName: "gpu1",
		HostName: "192.168.1.106",
	},
	"fusion-as1": {
		TaskName: "fusion",
		NodeName: "k8s-as1",
		HostName: "192.168.1.100",
	},
	"fusion-controller": {
		TaskName: "fusion",
		NodeName: "controller",
		HostName: "192.168.1.101",
	},
	"det-gpu1": {
		TaskName: "det",
		NodeName: "gpu1",
		HostName: "192.168.1.106",
	},
	// "slam-gpu1": {
	// 	TaskName: "slam",
	// 	NodeName: "gpu1",
	// 	HostName: "192.168.1.106",
	// },
	// "mcmot-controller": {
	// 	TaskName: "mcmot",
	// 	NodeName: "controller",
	// 	HostName: "192.168.1.101",
	// },
	// "slam-controller": {
	// 	TaskName: "slam",
	// 	NodeName: "controller",
	// 	HostName: "192.168.1.101",
	// },
	// "fusion-controller": {
	// 	TaskName: "fusion",
	// 	NodeName: "controller",
	// 	HostName: "192.168.1.101",
	// },
}

var clientSet *kubernetes.Clientset = nil
var clientSetLock = sync.Mutex{}

func GetClientSet() *kubernetes.Clientset {
	clientSetLock.Lock()
	if clientSet == nil {
		config, err := rest.InClusterConfig()
		if err != nil {
			log.Panic(err)
		}
		clientSet, err = kubernetes.NewForConfig(config)
		if err != nil {
			log.Panic(err)
		}
	}
	clientSetLock.Unlock()
	return clientSet
}

type MonitorInfo struct {
	HostName string
	NodeName string
	TaskType string
}

func CreateMonitor(Monitor *MonitorInfo) {
	config, err := rest.InClusterConfig()
	if err != nil {
		log.Panic(err)
	}

	clientSet, err := kubernetes.NewForConfig(config)
	if err != nil {
		log.Panic(err)
	}

	worker := addWorker(Monitor.HostName, Monitor.TaskType, Monitor.NodeName)
	workerName := worker.GetWorkerName()
	worker.PodName = workerName

	// define the container
	container := corev1.Container{
		Name:  workerName,
		Image: "docker.io/mchen644/monitor:v0",
		Resources: corev1.ResourceRequirements{
			Limits: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("0"),
				corev1.ResourceMemory: resource.MustParse("0"),
			},
			Requests: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("0"),
				corev1.ResourceMemory: resource.MustParse("0"),
			},
		},
		Env: []corev1.EnvVar{
			{
				Name:  "task_name",
				Value: worker.taskType,
			},
			{
				Name:  "port",
				Value: "8082",
			},
			{
				Name:  "GPU_CORE_UTILIZATION_POLICY",
				Value: "force",
			},
		},
	}

	// define the pod:
	var pod *corev1.Pod
	pod = &corev1.Pod{
		ObjectMeta: meta_v1.ObjectMeta{
			Name: workerName,
			Labels: map[string]string{
				"worker": worker.taskType,
			},
		},
		Spec: corev1.PodSpec{
			NodeName:    Monitor.NodeName,
			Containers:  []corev1.Container{container},
			HostNetwork: true,
		},
	}

	// create the pod
	utils.DebugWithTimeWait("Before Created")
	result, err := clientSet.CoreV1().Pods("default").Create(context.Background(), pod, meta_v1.CreateOptions{})
	if err != nil {
		log.Panic(err)
	}

	utils.DebugWithTimeWait("PodCreated")

	log.Printf("Creating pod %v in %v.\n", result.GetObjectMeta().GetName(), worker.nodeName)

	// wait until pod created
	err = wait.PollImmediate(500*time.Millisecond, 2*time.Minute, func() (bool, error) {
		podFound, err := clientSet.CoreV1().Pods("default").Get(context.Background(),
			pod.Name, meta_v1.GetOptions{})

		utils.DebugWithTimeWait(fmt.Sprintf("podFound is %v", podFound))
		if err != nil {
			return false, err
		}

		if podFound.Status.Phase == corev1.PodRunning {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		log.Panic(err)
	}
	utils.DebugWithTimeWait("PodCreated Waiting End")

	log.Printf("Pod Created!")
}

func CreateWorker(taskName, nodeName, hostname,
	cpuLimit, memLimit, gpuLimit, gpuMemory string) *Worker {
	// create the Kubernetes client
	config, err := rest.InClusterConfig()
	if err != nil {
		log.Panic(err)
	}
	clientSet, err := kubernetes.NewForConfig(config)
	if err != nil {
		log.Panic(err)
	}

	worker := addWorker(hostname, taskName, nodeName)
	log.Printf("add Worker whose taskName is %s successfully\n", taskName)
	name := worker.GetWorkerName()
	worker.PodName = name

	// define the container
	container := corev1.Container{
		Name:            name,
		Image:           containerMap[taskName],
		ImagePullPolicy: corev1.PullNever,
		Resources: corev1.ResourceRequirements{
			Limits: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse(cpuLimit),
				corev1.ResourceMemory: resource.MustParse(memLimit),
			},
			Requests: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("0"),
				corev1.ResourceMemory: resource.MustParse("0"),
			},
		},
		Env: []corev1.EnvVar{
			{
				Name:  "task_name",
				Value: taskName,
			},
			{
				Name:  "port",
				Value: worker.port,
			},
			{
				Name:  "GPU_CORE_UTILIZATION_POLICY",
				Value: "force",
			},
		},
	}
	utils.DebugWithTimeWait(fmt.Sprintf("Must Parse is [%v]", resource.MustParse(cpuLimit)))
	utils.DebugWithTimeWait(fmt.Sprintf("GPULIMIT is %v, cpu %v, mem %v", gpuLimit, cpuLimit, memLimit))
	if gpuLimit != "0" {
		log.Printf("Enable #%v gpu", gpuLimit)
		container.Resources.Limits["nvidia.com/gpucores"] = resource.MustParse(gpuLimit)
		container.Resources.Limits["nvidia.com/gpu"] = resource.MustParse("1")
		container.Resources.Limits["nvidia.com/gpumem"] = resource.MustParse(gpuMemory)
	}

	utils.DebugWithTimeWait(fmt.Sprintf("Set GPU Limit completed\n container info is %v", container))

	// define the pod
	var pod *corev1.Pod
	if gpuLimit != "0" {
		pod = &corev1.Pod{
			ObjectMeta: meta_v1.ObjectMeta{
				Name: name,
				Labels: map[string]string{
					"worker": taskName,
				},
			},
			Spec: corev1.PodSpec{
				Containers:  []corev1.Container{container},
				HostNetwork: true,
			},
		}
	} else {
		pod = &corev1.Pod{
			ObjectMeta: meta_v1.ObjectMeta{
				Name: name,
				Labels: map[string]string{
					"worker": taskName,
				},
			},
			Spec: corev1.PodSpec{
				NodeName:    nodeName,
				Containers:  []corev1.Container{container},
				HostNetwork: true,
			},
		}
	}

	if gpuLimit != "0" {
		toleration := corev1.Toleration{
			Key:      "nvidia.com/gpu",
			Operator: corev1.TolerationOpExists,
			Effect:   corev1.TaintEffectNoSchedule,
		}

		pod.Spec.Tolerations = []corev1.Toleration{toleration}
	}

	// create the pod
	utils.DebugWithTimeWait("Before Created")
	result, err := clientSet.CoreV1().Pods("default").Create(context.Background(), pod, meta_v1.CreateOptions{})
	if err != nil {
		log.Panic(err)
	}

	utils.DebugWithTimeWait("PodCreated")

	log.Printf("Creating pod %v in %v.\n", result.GetObjectMeta().GetName(), nodeName)

	// wait until pod created
	// 500s后会超时丢弃，因此找不到pod了，直接停止轮询
	err = wait.PollImmediate(500*time.Millisecond, 10*time.Minute, func() (bool, error) {
		podFound, err := clientSet.CoreV1().Pods("default").Get(context.Background(),
			pod.Name, meta_v1.GetOptions{})

		utils.DebugWithTimeWait(fmt.Sprintf("podFound is %v", podFound))
		if err != nil {
			return false, err
		}

		if podFound.Status.Phase == corev1.PodRunning {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		fmt.Printf("In creating the Worker, Pod did not reach Running state within the time limit: %v", err)
		return nil
	}
	utils.DebugWithTimeWait("Pod Created Waiting End")

	log.Printf("Pod Created!")
	return worker
}

func (w *Worker) UpdateResourceLimit(mcpu int64) {
	clientSet = GetClientSet()

	pod, err := clientSet.CoreV1().Pods("default").Get(context.Background(), w.PodName, meta_v1.GetOptions{})
	if err != nil {
		log.Panic(err)
	}

	for i, container := range pod.Spec.Containers {
		if container.Name == w.PodName {
			log.Printf("Find container %v", container.Name)
			pod.Spec.Containers[i].Resources.Limits.Cpu().SetMilli(mcpu)
		}
	}

	_, err = clientSet.CoreV1().Pods("default").Update(context.Background(), pod, meta_v1.UpdateOptions{})
	if err != nil {
		log.Panic(err)
	}

	// wait until pod updated
	err = wait.PollImmediate(500*time.Millisecond, 2*time.Minute, func() (bool, error) {
		podFound, err := clientSet.CoreV1().Pods("default").Get(context.Background(),
			pod.Name, meta_v1.GetOptions{})
		if err != nil {
			log.Panic(err)
		}

		if podFound.Status.Phase == corev1.PodRunning {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		log.Panic(err)
	}

	log.Printf("Pod %v: cpu limit has updated to %vm", pod.Name, mcpu)

}

// ResourceUsage
// Storage 和持久卷绑定，pod删除不消失
// StorageEphemeral pod删除就释放
// Measure resource in range [ CollectedTime - Window, CollectedTime ]
type ResourceUsage struct {
	CPU              int64  `json:"CPU"`
	Memory           int64  `json:"Memory"`
	Storage          int64  `json:"Storage"`
	StorageEphemeral int64  `json:"StorageEphemeral"`
	CollectedTime    string `json:"CollectedTime"`
	Window           int64  `json:"Window"`
	Available        bool   `json:"Available"`
	PodName          string `json:"PodName"`
}

func QueryResourceUsage(PodName string) *ResourceUsage {
	config, err := rest.InClusterConfig()
	if err != nil {
		log.Panic(err)
	}

	metricsClient, err := versioned.NewForConfig(config)
	if err != nil {
		log.Panic(err)
	}
	metricsInterface := metricsClient.MetricsV1beta1().PodMetricses("default")

	podMetrics, err := metricsInterface.Get(context.Background(), PodName, meta_v1.GetOptions{})
	if errors.IsNotFound(err) {
		log.Printf("Pod %v is not found", PodName)
		return &ResourceUsage{
			CPU:              0,
			Memory:           0,
			Storage:          0,
			StorageEphemeral: 0,
			CollectedTime:    "Pod Not Found",
			Window:           0,
			Available:        false,
		}
	} else if err != nil {
		log.Panic(err)
	}

	usage := podMetrics.Containers[0].Usage

	resourceUsage := &ResourceUsage{
		CPU:              usage.Cpu().MilliValue(),
		Memory:           usage.Memory().MilliValue(),
		Storage:          usage.Storage().MilliValue(),
		StorageEphemeral: usage.StorageEphemeral().MilliValue(),
		CollectedTime:    podMetrics.Timestamp.Time.String(),
		Window:           podMetrics.Window.Duration.Milliseconds(),
		Available:        true,
		PodName:          PodName,
	}

	return resourceUsage
}
