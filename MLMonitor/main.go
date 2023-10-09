package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/shirou/gopsutil/cpu"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

var MonitorPort = ":8082"

type ResourceUsageState struct {
	CPU string `json: "CPU"`
	GPU string `json: "GPU"`
}

type NumPods struct {
	Num1 int `json: "as1Pods"`
	Num2 int `json: "controllerPods"`
	Num3 int `json: "gpu1Pods"`
}

func main() {
	fmt.Printf("Monitor start at %s\n", MonitorPort)
	RunHttpServer()
}

type Router struct{}

func monitorPods(w http.ResponseWriter, req *http.Request) {
	NumPods := make(map[string]int, 3)
	// create the Kubernetes client
	config, err := rest.InClusterConfig()

	if err != nil {
		log.Panic(err)
	}
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		log.Panic(err)
	}

	// 设置要查询的命名空间
	namespace := "default"

	// 构建Nodes列表的请求
	listOptions := metav1.ListOptions{}

	// 获取所有Node上正在运行的Pod数量
	nodes, err := clientset.CoreV1().Nodes().List(context.TODO(), listOptions)
	if err != nil {
		panic(err.Error())
	}

	for _, node := range nodes.Items {
		// 获取Node的名称
		nodeName := node.Name

		// 获取Node上的Pod数量
		nodePods, err := clientset.CoreV1().Pods(namespace).List(context.TODO(), metav1.ListOptions{
			FieldSelector: fmt.Sprintf("spec.nodeName=%s", nodeName),
		})
		if err != nil {
			panic(err.Error())
		}

		// 统计包括运行、创建和删除状态的Pod数量
		runningPodsOnNode := 0
		creatingPodsOnNode := 0
		deletingPodsOnNode := 0

		for _, pod := range nodePods.Items {
			switch pod.Status.Phase {
			case corev1.PodRunning:
				runningPodsOnNode++
			case corev1.PodPending:
				creatingPodsOnNode++
			case corev1.PodSucceeded, corev1.PodFailed:
				deletingPodsOnNode++
			}
		}

		// 打印每个Node上的Pod数量（包括运行、创建和删除）
		fmt.Printf("Node: %s, 运行的Pod数量: %d, 创建中的Pod数量: %d, 正在删除的Pod数量: %d\n", nodeName, runningPodsOnNode, creatingPodsOnNode, deletingPodsOnNode)
		NumPods[nodeName] = runningPodsOnNode + creatingPodsOnNode + deletingPodsOnNode
	}

	jsonString, err := json.Marshal(NumPods)
	if err != nil {
		log.Fatal(err)
	}

	// 设置响应的Content-Type标头为application/json
	w.Header().Set("Content-Type", "application/json")

	// 发送JSON字符串作为响应
	_, err = w.Write(jsonString)
	if err != nil {
		log.Fatal(err)
	}

}

func (r *Router) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	switch req.URL.Path {
	case "/monitor":
		monitorStates(w, req)
	case "/monitorPods":
		monitorPods(w, req)
	}
}

func monitorStates(w http.ResponseWriter, req *http.Request) {
	// CPU
	cpuUsage := monitorCPU()

	// GPU
	// gpuUsage := monitorGPU()

	totalUsage := &ResourceUsageState{
		CPU: cpuUsage,
		GPU: "0",
	}

	mashal, err := json.Marshal(totalUsage)

	if err != nil {
		log.Fatal("json marshal error:", err)
	}

	_, err = w.Write(mashal)
	if err != nil {
		log.Fatal("Response write error:", err)
	}
}

func monitorCPU() string {
	// cpuCount := runtime.NumCPU()
	percent, _ := cpu.Percent(0, true)
	avg_CPU := 0.0
	for _, perCPU := range percent {
		avg_CPU += perCPU
	}
	avg_CPU = avg_CPU / float64(len(percent))

	// for i := 0; i < cpuCount; i++ {
	// 	fmt.Printf("CPU%d Average: %.2f%%\n", i, percent[i])
	// }

	fmt.Printf("CPU Average: %.2f%%\n", avg_CPU)
	str_CPU := strconv.FormatFloat(avg_CPU, 'f', -1, 64)
	return str_CPU
}

func monitorGPU() string {
	cmd := exec.Command("nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits")
	output, err := cmd.Output()
	gpuUsage := ""
	if err != nil {
		log.Fatal("Error:", err)
		gpuUsage = err.Error()
	} else {
		gpuUsage = strings.TrimSpace(string(output))
		fmt.Println("GPU utilization:", gpuUsage)
	}

	return gpuUsage
}

func RunHttpServer() {
	server := &http.Server{
		Addr:         MonitorPort,
		ReadTimeout:  1 * time.Minute,
		WriteTimeout: 1 * time.Minute,
		IdleTimeout:  1 * time.Minute,
		Handler:      &Router{},
	}

	if err := server.ListenAndServe(); err != nil {
		log.Fatal("Monitor Listen error:", err)
	}

}
