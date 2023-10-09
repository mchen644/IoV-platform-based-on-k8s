package main

import (
	"Scheduler/buffer_pool"
	"Scheduler/handler"
	"Scheduler/utils"
	"Scheduler/worker_pool"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"net/http/pprof"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

var schedulerPort = ":8081"
var completTaskLoackMap sync.Map

const completTaskImageDir = "complete_task/data/00/"
const complteTaskImageName = "input.png"

const (
	STATUS_BEGIN   = "Begin"
	STATUS_RUNNING = "Running"
	STATUS_LAST    = "Last"
)

type CompleteMetrics struct {
	FusionNodeName string  `csv:"fusion_node_name"`
	GpuLimit       int     `csv:"gpu_limit"`
	CpuLimit       int     `csv:"cpu_limit"`
	AvgLatency     float64 `csv:"avg_latency"`
}

type MetricsStat struct {
	CPULimit       string        `csv:"CPULimit"`
	GpuLimit       string        `csv:"GPULimit"`
	AvgLatency     time.Duration `csv:"AvgLatency"`
	AvgCPUUsage    float64       `csv:"AvgCPUUsage"`
	MaxCPUUsage    int64         `csv:"MaxCPUUsage"`
	AvgMemoryUsage float64       `csv:"AvgMemoryUsage"`
	MaxMemoryUsage int64         `csv:"MaxMemoryUsage"`
	HighLoadRatio  float64       `csv:"HighLoadRatio"`
	TaskNumber     int           `csv:"TaskNumber"`
}

type ResourceUsage struct {
	CPU              int64  `json:"CPU" csv:"CPU"`
	Memory           int64  `json:"Memory" csv:"Memory"`
	Storage          int64  `json:"Storage" csv:"Storage"`
	StorageEphemeral int64  `json:"StorageEphemeral" csv:"StorageEphemeral"`
	CollectedTime    string `json:"CollectedTime" csv:"CollectedTime"`
	Window           int64  `json:"Window" csv:"Window"`
	Available        bool   `json:"Available" csv:"Available"`
	PodName          string `json:"PodName" csv:"PodName"`
}

type LatencyMetrics struct {
	result             string
	slamIOLatency      time.Duration
	slamComputeLatency time.Duration
	detIOLatency       time.Duration
	detComputeLatency  time.Duration
	fusionLatency      time.Duration
	totalLatency       time.Duration
}

type router struct{}

type ResponseAction struct {
	State      []float32 `json:"State"`
	Action     []float64 `json:"Action"`
	CPU        int       `json:"CPU"`
	GPU        int       `json:"GPU"`
	Offloading string    `json:"Offloading"`
}

func (r *router) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	//log.Printf("Receive Path: [%v]", req.URL.Path)
	switch req.URL.Path {
	case "/test":
		testEOF(w, req)
	case "/Queue":
		Queue(w, req)
	case "/scheduling":
		scheduling(w, req)
	case "/receive_results":
		receiveResults(w, req)
	case "/new_task":
		newTask(w, req)
	case "/complete_task":
		completeTask(w, req)
	case "/worker_register":
		workerRegister(w, req)
	case "/query_metric":
		queryMetrics(w, req)
	case "/mcmot_finish":
		handler.GetHandler("mcmot").FinishTask(w, req)
	case "/slam_finish":
		handler.GetHandler("slam").FinishTask(w, req)
	case "/fusion_finish":
		handler.GetHandler("fusion").FinishTask(w, req)
	case "/det_finish":
		handler.GetHandler("det").FinishTask(w, req)
	case "/update_cpu":
		updateCPU(w, req)
	case "/create_workers":
		createWorkers(w, req)
	case "/create_monitors":
		createMonitors(w, req)
	case "/restart":
		restart(w, req)
	case "/debug/pprof/profile":
		pprof.Profile(w, req)
	default:
		http.NotFound(w, req)
	}
}

func testEOF(w http.ResponseWriter, req *http.Request) {
	w.Write([]byte("ok"))
}

func receiveResults(w http.ResponseWriter, r *http.Request) {
	multipartReader, err := r.MultipartReader()
	if err != nil {
		log.Panic(err)
	}

	form, err := multipartReader.ReadForm(1024 * 1024 * 2)
	if err != nil {
		log.Panic(err)
	}

	//detTaskID := form.Value["det_task_id"][0]
	fusionTaskID := form.Value["fusion_task_id"][0]
	fusionResult := form.Value["fusion_result"][0]

	notifier, ok := completTaskLoackMap.LoadOrStore(fusionTaskID, make(chan string, 1))
	if !ok {
		log.Panic("Not Load chan")
	}

	getLatency := func(name string) time.Duration {
		latency, err := time.ParseDuration(form.Value[name][0])
		if err != nil {
			log.Panic(err)
		}
		return latency
	}

	metrics := LatencyMetrics{
		result:             fusionResult,
		slamIOLatency:      getLatency("slam_io_latency"),
		slamComputeLatency: getLatency("slam_compute_latency"),
		detIOLatency:       getLatency("det_io_latency"),
		detComputeLatency:  getLatency("det_compute_latency"),
		fusionLatency:      getLatency("fusion_latency"),
		totalLatency:       getLatency("total_latency"),
	}

	notifier.(chan LatencyMetrics) <- metrics

	//log.Printf("Task id %v, result:\n%v", taskID, result)

	_, err = w.Write([]byte("OK"))
	if err != nil {
		log.Panic(err)
	}
}

func queryAction() *ResponseAction {
	responseActions, err := http.Get("http://localhost:8083/queryActions")

	if err != nil {
		log.Fatal("Scheduler request action error:", err)
	}

	content, err := io.ReadAll(responseActions.Body)
	if err != nil {
		log.Fatal("io read response error: ", err)
	}

	responseActions.Body.Close()
	var responseAction ResponseAction

	err = json.Unmarshal(content, &responseAction)

	if err != nil {
		log.Fatal("json Ummarshal error: ", err)
	}

	log.Printf("action is %v\n", responseAction)
	return &responseAction
}

func TrainAgent(Action *ResponseAction) {
	actionJSON, err := json.Marshal(Action)

	if err != nil {
		log.Fatal("Failed to marshal Action to JSON:", err)
	}

	// 发起POST请求
	_, err = http.Post("http://localhost:8083/train", "application/json", bytes.NewBuffer(actionJSON))
	if err != nil {
		log.Fatal("Scheduler send training request error:", err)
	}

}

func scheduling(w http.ResponseWriter, req *http.Request) {
	var metrics CompleteMetrics

	startTime := time.Now()
	Action := queryAction()

	cpuLimit := Action.CPU
	gpuLimit := Action.GPU
	var fusionNodeName string

	if Action.Offloading == "k8s-as1" {
		fusionNodeName = "as1"
	} else {
		fusionNodeName = Action.Offloading
	}

	log.Println("time consuming:%v\n", time.Since(startTime).Seconds())
	log.Printf("for cpu[%v] gpu[%v] fusion node[%v]",
		cpuLimit, gpuLimit, fusionNodeName)

	go func() {
		time.Sleep(35 * time.Second) // 快结束的时候，发送一个trainAgent的请求
		TrainAgent(Action)
	}()

	ctx, _ := context.WithCancel(context.Background())
	metrics = testCompleteTask(fusionNodeName, cpuLimit, gpuLimit, true, ctx, Action.Action)
	log.Println(metrics.AvgLatency)
	log.Println(metrics.CpuLimit)
	log.Println("test Done!!!!!!!!!!!!!!!!!!!!!!!!!")

	mashal, err := json.Marshal(metrics)

	if err != nil {
		log.Fatal("json mashal error:", err)
	}

	_, err = w.Write(mashal)

	if err != nil {
		log.Fatal("Response write error:", err)
	}

}

func Queue(w http.ResponseWriter, req *http.Request) {
	body := []byte{}
	resp, err := http.Post("http://localhost:8083/Queue", "application/json", bytes.NewBuffer(body))
	if err != nil {
		log.Fatalf("POST Queue request error: %v", err)
	}
	defer resp.Body.Close()
}

func createGeneralWorkers(info *CreateInfo) {
	info.BatchSize = map[string]int{
		"controller": 8,
		"as1":        4,
		"gpu1":       3,
	}

	utils.DebugWithTimeWait("Before creating workers")
	log.Printf("Creating some workers... \n%v", info)
	worker_pool.InitWorkers(info.WorkerNumbers, info.BatchSize, info.CpuLimits,
		info.GpuLimits, info.GpuMemory, info.TaskName)
	utils.DebugWithTimeWait("After creating workers")
}

func sendCompletTaskRequest(taskInfo CompleteTaskInfo, imagePath string, ctx context.Context) (string, string) {
	bufferElem := utils.GetBuffer()
	body := bufferElem.Buffer
	multipartWriter := multipart.NewWriter(body)

	writeValue := func(key, value string) {
		err := multipartWriter.WriteField(key, value)
		if err != nil {
			log.Panic(err)
		}
	}

	writeFile := func(key, filePath, fileName string) {
		filePart, err := multipartWriter.CreateFormFile(key, fileName)
		if err != nil {
			log.Panic(err)
		}
		file, err := os.Open(filePath)
		if err != nil {
			log.Panic(err)
		}
		defer func(file *os.File) {
			err = file.Close()
			if err != nil {
				log.Panic(err)
			}
		}(file)

		_, err = io.Copy(filePart, file)
		if err != nil {
			log.Panic(err)
		}
	}

	if taskInfo.Status == "Last" {
		taskInfo.DeleteDETWorker = true
		taskInfo.DeleteFusionWorker = true
		log.Printf("Task id %v, will delete", taskInfo.FusionTaskID)
	}

	jsonByte, err := json.Marshal(taskInfo)
	if err != nil {
		log.Panic(err)
	}

	writeValue("json", string(jsonByte))

	writeFile("frame", imagePath, complteTaskImageName)

	err = multipartWriter.Close()
	if err != nil {
		log.Panic(err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "http://localhost:8081/complete_task", body)
	if err != nil {
		log.Panic(err)
	}

	req.Header.Set("Content-Type", multipartWriter.FormDataContentType())
	client := &http.Client{}
	postResp, err := client.Do(req)

	if errors.Is(err, context.Canceled) {
		log.Println("http cancelled")
		utils.ReturnBuffer(bufferElem)
		return "", ""
	} else if err != nil {
		log.Panic(err)
	}

	utils.ReturnBuffer(bufferElem)

	content, err := io.ReadAll(postResp.Body)
	if err != nil {
		log.Panic(err)
	}

	postResp.Body.Close()

	results := strings.Split(string(content), ":")
	if results[0] == "Failed" {
		fmt.Println("Stuck is happening!")
		return "Failed", "Failed"
	}
	// fmt.Printf("results = %v\n", results)
	detTaskID := results[0]
	fusionTaskID := results[1]

	return detTaskID, fusionTaskID
}

func SendUpdateQueueRequest() {
	body := []byte{}
	resp, err := http.Post("http://localhost:8083/updateQueue", "application/json", bytes.NewBuffer(body))
	if err != nil {
		log.Fatalf("POST update Queue error: %v", err)
	}
	defer resp.Body.Close()
}

func testCompleteTask(fusionNodeName string, cpuLimit, gpuLimit int,
	writeResult bool, ctx context.Context, action []float64) CompleteMetrics {
	fusionResult := ""
	log.Println("Go into testCompleteTask()\n")
	detWorkerCreateInfo := &CreateInfo{
		CpuLimits: map[string]int{
			"gpu1": 0,
		},
		WorkerNumbers: map[string]int{
			"gpu1": 1,
		},
		TaskName: "det",
		GpuLimits: map[string]int{
			"gpu1": gpuLimit,
		},
		GpuMemory: map[string]int{
			"gpu1": 6000,
		},
	}

	fusionWorkerCreateInfo := &CreateInfo{
		CpuLimits: map[string]int{
			fusionNodeName: cpuLimit,
		},
		WorkerNumbers: map[string]int{
			fusionNodeName: 1,
		},
		TaskName: "fusion",
		GpuLimits: map[string]int{
			fusionNodeName: 0,
		},
		GpuMemory: map[string]int{
			fusionNodeName: 0,
		},
	}

	taskInfo := &CompleteTaskInfo{
		DETNodeName:        "gpu1",
		DETTaskID:          "-1",
		FusionNodeName:     fusionNodeName,
		FusionTaskID:       "-1",
		Status:             "Begin",
		DeleteDETWorker:    false,
		DeleteFusionWorker: false,
		CPULimit:           cpuLimit,
		GPULimit:           gpuLimit,
		Action:             action,
	}

	SendUpdateQueueRequest()
	log.Printf("Update queue successfully!")

	// Begin here
	// startPoint:
	// 	done := make(chan bool)
	// 	ctx, cancel := context.WithCancel(context.Background())
	// Begin here

	wg_worker := sync.WaitGroup{}
	wg_worker.Add(2)

	go func() {
		defer wg_worker.Done()
		createGeneralWorkers(detWorkerCreateInfo)
		log.Printf("det Worker created successfully!")
	}()

	go func() {
		defer wg_worker.Done()
		createGeneralWorkers(fusionWorkerCreateInfo)
		log.Printf("fusion Worker created successfully!")
	}()

	wg_worker.Wait()

	// time.Sleep(time.Second * 1)

	// const imageLength = 465
	const imageLength = 10
	var metricsMap = make(map[int]MetricsStat)

	detailMetricsMap := make(map[string]*ResourceUsage)

	var latencies []LatencyMetrics
	queryTime := time.Now()

	for imageIndex := 0; imageIndex < imageLength; imageIndex++ {
		select {
		case <-ctx.Done():
			return CompleteMetrics{}
		default:
			// pass
		}

		if writeResult {
			log.Printf("\rDoing image %v/%v", imageIndex+1, imageLength)
		}

		os.Stdin.Sync()
		if taskInfo.FusionTaskID == "-1" {
			taskInfo.Status = "Begin"
		} else if imageIndex == imageLength-1 {
			taskInfo.Status = "Last"
			notifier, _ := completTaskLoackMap.Load(taskInfo.FusionTaskID)
			latencyMetrics := <-notifier.(chan LatencyMetrics)
			fusionResult = fusionResult + latencyMetrics.result
			latencies = append(latencies, latencyMetrics)
		} else {
			taskInfo.Status = "Running"

			notifier, _ := completTaskLoackMap.LoadOrStore(taskInfo.FusionTaskID, make(chan LatencyMetrics, 1))

			latencyMetrics := <-notifier.(chan LatencyMetrics)

			fusionResult = fusionResult + latencyMetrics.result

			latencies = append(latencies, latencyMetrics)
		}

		imagePath := completTaskImageDir + utils.IntToNDigitsString(imageIndex, 6) + ".png"
		detTaskID, fusionTaskID := sendCompletTaskRequest(*taskInfo, imagePath, ctx)

		if detTaskID == "Failed" {
			break
		}

		if taskInfo.Status == "Begin" {
			taskInfo.DETTaskID = detTaskID
			taskInfo.FusionTaskID = fusionTaskID
		}

		//log.Printf("task info is [%v]", taskInfo)
		if taskInfo.Status == "Running" {
			queryDuration := time.Since(queryTime)
			if queryDuration.Seconds() >= 2 {
				//usage := queryMetrics(taskInfo.FusionTaskID)
				//detailMetricsMap[usage.CollectedTime] = usage
				queryTime = time.Now()
			}
		} else if taskInfo.Status == "Last" {
			// 这边可以Post一个请求给DecisionMaker，更新现有的资源
			response, err := http.Get("http://localhost:8082/monitorPods")
			if err != nil {
				log.Fatal("Get Pods number error:", err)
			}

			content, err := io.ReadAll(response.Body)

			if err != nil {
				log.Fatal("io read response error: ", err)
			}

			response.Body.Close()
			fmt.Println("v1")
			var podsNum map[string]int = make(map[string]int)
			err = json.Unmarshal(content, &podsNum)
			if err != nil {
				log.Fatal("json Unmarshal error: ", err)
			}
			log.Println("准备完成所有的task, 即将释放资源")
			log.Println(podsNum)

			notifier, _ := completTaskLoackMap.Load(taskInfo.FusionTaskID)
			latencyMetrics := <-notifier.(chan LatencyMetrics)
			fusionResult = fusionResult + latencyMetrics.result
			completTaskLoackMap.Delete(taskInfo.FusionTaskID)
		}

	}

	if writeResult {
		avgLatency := 0.
		for _, l := range latencies {
			avgLatency += float64(l.totalLatency.Milliseconds())
		}
		log.Printf("The length of latencies is %d\n", len(latencies))
		avgLatency = avgLatency / float64(len(latencies))

		log.Printf("Avg latency is %v ms for gpu limits: %v", avgLatency, gpuLimit)
		var maxCPU int64 = 0
		var maxMem int64 = 0
		var detailResults []ResourceUsage
		for _, item := range detailMetricsMap {
			if item.CPU > maxCPU {
				maxCPU = item.CPU
			}
			if item.Memory > maxMem {
				maxMem = item.Memory
			}
			detailResults = append(detailResults, *item)
		}

		file, err := os.Create(fmt.Sprintf("complete_task/detail_metrics/%v_%v_%v_%v.csv",
			"fusion", fusionNodeName, gpuLimit, cpuLimit))
		if err != nil {
			log.Panic(err)
		}

		_, err = file.Write(utils.MarshalCSV(detailResults))
		if err != nil {
			log.Panic(err)
		}

		if err = file.Close(); err != nil {
			log.Panic(err)
		}

		file, err = os.Create(fmt.Sprintf("complete_task/detail_metrics/latencies_%v_%v_%v.csv",
			fusionNodeName, gpuLimit, cpuLimit))
		if err != nil {
			log.Panic(err)
		}

		var latenciesString string
		for _, latency := range latencies {
			latenciesString = latenciesString +
				fmt.Sprintf("%v %v %v %v %v %v\n",
					latency.totalLatency.Milliseconds(),
					latency.slamIOLatency.Milliseconds(),
					latency.slamComputeLatency.Milliseconds(),
					latency.detIOLatency.Milliseconds(),
					latency.detComputeLatency.Milliseconds(),
					latency.fusionLatency.Microseconds())
		}

		_, err = file.Write([]byte(latenciesString))
		if err != nil {
			log.Panic(err)
		}

		if err = file.Close(); err != nil {
			log.Panic(err)
		}

		log.Printf("Max cpu usage %v", maxCPU)

		var avgCPU float64 = 0
		var avgMem float64 = 0
		count := 0
		for _, item := range detailMetricsMap {
			if float64(item.CPU) >= float64(maxCPU)*0.8 {
				avgCPU += float64(item.CPU)
				avgMem += float64(item.Memory)
				count += 1
			}
		}
		avgCPU /= float64(count)
		avgMem /= float64(count)

		metricsMap[gpuLimit] = MetricsStat{
			CPULimit:       "0",
			GpuLimit:       strconv.Itoa(gpuLimit),
			AvgLatency:     time.Duration(avgLatency) * time.Millisecond,
			AvgCPUUsage:    avgCPU,
			MaxCPUUsage:    maxCPU,
			AvgMemoryUsage: avgMem,
			MaxMemoryUsage: maxMem,
			HighLoadRatio:  float64(len(detailMetricsMap)) / float64(count),
			TaskNumber:     1,
		}

		var results []MetricsStat
		for _, metrics := range metricsMap {
			log.Printf("Node:%v, metrics data is %v",
				"gpu1", metrics)
			results = append(results, metrics)
		}

		file, err = os.Create(fmt.Sprintf("complete_task/fusion_result/%v_%v_%v.csv",
			fusionNodeName, gpuLimit, cpuLimit))
		if err != nil {
			log.Panic(err)
		}

		_, err = file.Write([]byte(fusionResult))
		if err != nil {
			log.Panic(err)
		}

		if err = file.Close(); err != nil {
			log.Panic(err)
		}

		return CompleteMetrics{
			FusionNodeName: fusionNodeName,
			GpuLimit:       gpuLimit,
			CpuLimit:       cpuLimit,
			AvgLatency:     avgLatency,
		}
	}

	return CompleteMetrics{}
}

// TODO
func get_action() (int, int, string) {
	var CPU int
	var GPU int
	var fusionNodeName string
	return CPU, GPU, fusionNodeName
}

func createMonitors(w http.ResponseWriter, req *http.Request) {
	wg := sync.WaitGroup{}
	wg.Add(3)
	var Monitors []*worker_pool.MonitorInfo

	Monitors = append(Monitors, &worker_pool.MonitorInfo{
		HostName: "192.168.1.101",
		NodeName: "controller",
		TaskType: "monitor",
	})

	Monitors = append(Monitors, &worker_pool.MonitorInfo{
		HostName: "192.168.1.100",
		NodeName: "k8s-as1",
		TaskType: "monitor",
	})

	Monitors = append(Monitors, &worker_pool.MonitorInfo{
		HostName: "192.168.1.106",
		NodeName: "gpu1",
		TaskType: "monitor",
	})

	for i := 0; i < len(Monitors); i++ {
		go func(i int) {
			worker_pool.CreateMonitor(Monitors[i])
			defer wg.Done()
		}(i)
	}
	wg.Wait()
	fmt.Println("Monitors are created successfully!")
}

func RunHttpServer() {
	server := &http.Server{
		Addr:         schedulerPort,
		ReadTimeout:  10 * time.Hour,
		WriteTimeout: 10 * time.Hour,
		IdleTimeout:  10 * time.Hour,
		Handler:      &router{},
	}

	if err := server.ListenAndServe(); err != nil {
		log.Panicf("listen: %s\n", err)
	}

}

type CreateInfo struct {
	CpuLimits     map[string]int `json:"cpu_limit"`
	WorkerNumbers map[string]int `json:"worker_numbers"`
	TaskName      string         `json:"task_name"`
	GpuLimits     map[string]int `json:"gpu_limit"`
	GpuMemory     map[string]int `json:"gpu_memory"`
	BatchSize     map[string]int
}

func restart(w http.ResponseWriter, r *http.Request) {
	w.Write([]byte("OK"))
	r.Body.Close()
	os.Exit(0)
}

func createWorkers(w http.ResponseWriter, r *http.Request) {
	rawInfo, err := io.ReadAll(r.Body)
	if err != nil {
		log.Panic(err)
	}

	info := &CreateInfo{}

	if err = json.Unmarshal(rawInfo, info); err != nil {
		log.Panic(err)
	}

	info.BatchSize = map[string]int{
		"controller": 8,
		"as1":        4,
		"gpu1":       3,
	}

	utils.DebugWithTimeWait("Before creating workers")
	log.Printf("Creating some workers... \n%v", info)
	worker_pool.InitWorkers(info.WorkerNumbers, info.BatchSize, info.CpuLimits,
		info.GpuLimits, info.GpuMemory, info.TaskName)
	utils.DebugWithTimeWait("After creating workers")
}

func updateCPU(w http.ResponseWriter, r *http.Request) {
	rawInfo, err := io.ReadAll(r.Body)
	if err != nil {
		log.Panic(err)
	}

	splitInfo := strings.Split(string(rawInfo), ":")
	log.Printf("receive updated requirement: %q", splitInfo)
	nodeName := splitInfo[0]
	rawCPU := splitInfo[1]

	cpuLimit, err := strconv.Atoi(rawCPU)
	if err != nil {
		log.Panic(err)
	}

	workerPool := worker_pool.GetWorkerPool("fusion")

	log.Printf("Has %v workers", len(workerPool))

	for _, taskType := range []string{"fusion"} {
		pool := worker_pool.GetWorkerPool(taskType)
		for _, worker := range pool {
			if worker.GetNodeName() == nodeName {
				worker.UpdateResourceLimit(int64(cpuLimit))
			}
		}
	}
}

// Each worker_pool node should register their IP When join the cluster
// TODO worker_pool nodes should also register their resources info
// workerRegister return back assigned port for the worker_pool
func workerRegister(w http.ResponseWriter, r *http.Request) {
	ip := strings.Split(r.RemoteAddr, ":")[0]

	bufferElem := buffer_pool.GetBuffer()
	buffer := bufferElem.Buffer
	if _, err := io.Copy(buffer, r.Body); err != nil {
		log.Panic(err)
	}

	buffer_pool.ReturnBuffer(bufferElem)

	log.Println("Worker ", ip, " Has been Registered")
}

type CompleteTaskInfo struct {
	DETNodeName        string    `json:"det_node_name"`
	DETTaskID          string    `json:"det_task_id"`
	FusionNodeName     string    `json:"fusion_node_name"`
	FusionTaskID       string    `json:"fusion_task_id"`
	Status             string    `json:"status"`
	DeleteDETWorker    bool      `json:"delete_det_worker"`
	DeleteFusionWorker bool      `json:"delete_fusion_worker"`
	CPULimit           int       `json:"cpuLimit"`
	GPULimit           int       `json:"gpuLimit"`
	Action             []float64 `json:"action"`
}

func completeTask(w http.ResponseWriter, r *http.Request) {

	reader, err := r.MultipartReader()
	if err != nil {
		log.Panic(err)
	}

	form, err := reader.ReadForm(1024 * 1024 * 2)
	if err != nil {
		log.Panic(err)
	}

	if len(form.Value["json"]) != 1 {
		log.Panicf("Expected len %v, got %v", 1, len(form.Value["json"]))
	}

	rawJson := form.Value["json"][0]
	taskInfo := &CompleteTaskInfo{}
	json.Unmarshal([]byte(rawJson), taskInfo)

	var detWorker, fusionWorker *worker_pool.Worker
	var detTaskID, fusionTaskID string

	if taskInfo.Status == STATUS_BEGIN {
		detTaskID = utils.GetUniqueID()
		fusionTaskID = utils.GetUniqueID()

		detWorker = worker_pool.OccupyWorker("det", detTaskID, taskInfo.DETNodeName)
		fusionWorker = worker_pool.OccupyWorker("fusion", fusionTaskID, taskInfo.FusionNodeName)
	} else {
		detWorker = worker_pool.GetWorkerByTaskID(taskInfo.DETTaskID)
		fusionWorker = worker_pool.GetWorkerByTaskID(taskInfo.FusionTaskID)

		detTaskID = taskInfo.DETTaskID
		fusionTaskID = taskInfo.FusionTaskID

		if detWorker == nil || fusionWorker == nil {
			log.Printf("Maybe work not complete before delete, occationally internal bugs")
			w.Write([]byte("Failed"))
			return
		}
	}

	taskHandler := handler.NewCompleteTaskHandler(
		detWorker,
		fusionWorker,
		detTaskID,
		fusionTaskID,
		form,
		taskInfo.Status,
		taskInfo.DeleteDETWorker,
		taskInfo.DeleteFusionWorker,
		r.RemoteAddr,
		taskInfo.CPULimit,
		taskInfo.GPULimit,
		taskInfo.FusionNodeName,
		taskInfo.Action)
	go taskHandler.SendTask()

	_, err = w.Write([]byte(fmt.Sprintf("%v:%v", detTaskID, fusionTaskID)))
	if err != nil {
		log.Panic(err)
	}
}

// Receive a task from devices, and submit to specific worker_pool
// Write back task id
// TODO apply and plug Scheduling and Resource Allocation Strategy
func newTask(w http.ResponseWriter, r *http.Request) {
	reader, err := r.MultipartReader()
	if err != nil {
		log.Panic(err)
	}

	form, err := reader.ReadForm(1024 * 1024 * 2)
	if err != nil {
		log.Panic(err)
	}

	taskName := form.Value["task_name"][0]
	if err != nil {
		log.Panic(err)
	}

	nodeName := form.Value["node_name"][0]
	if err != nil {
		log.Panic(err)
	}

	status := form.Value["status"][0]
	if err != nil {
		log.Panic(err)
	}

	taskID := form.Value["task_id"][0]
	if err != nil {
		log.Panic(err)
	}

	var worker *worker_pool.Worker
	var returnWorker bool

	if status == STATUS_BEGIN {
		taskID = utils.GetUniqueID()
		// TODO Make Decision Here, Apply True Resource Allocation
		// Default Round Robin and Allocate Expected Resource
		worker = worker_pool.OccupyWorker(taskName, taskID, nodeName)
		//worker := worker_pool.CreateWorker(podsInfo.TaskName, podsInfo.NodeName, podsInfo.HostName, cpuLimit)
		//worker.bindTaskID(strconv.Itoa(taskID))
		returnWorker = false
	} else if status == STATUS_RUNNING {
		worker = worker_pool.GetWorkerByTaskID(taskID)
		returnWorker = false
	} else if status == STATUS_LAST {
		worker = worker_pool.GetWorkerByTaskID(taskID)
		returnWorker = true
	}

	log.Printf("Receive task %v, assigned id %v, worker_pool %v", taskName, taskID, worker.Describe())
	if len(taskName) == 0 {
		_, err = w.Write([]byte("Un Complete Params!"))
		if err != nil {
			log.Panic(err)
		}
	}

	deleteWorker := len(form.Value["delete"]) != 0

	handlers := handler.GetHandler(taskName)
	now := time.Now()
	handlers.StartTask(worker, form, taskID)
	log.Printf("New Task start task %v", time.Since(now))
	handlers.SendBackResult(r, taskID, worker, returnWorker, deleteWorker)

	_, err = w.Write([]byte(taskID))
	if err != nil {
		log.Panic(err)
	}
}

func queryMetrics(w http.ResponseWriter, r *http.Request) {
	bufferElem := buffer_pool.GetBuffer()
	buffer := bufferElem.Buffer
	if _, err := io.Copy(buffer, r.Body); err != nil {
		log.Panic(err)
	}

	taskID := buffer.String()
	buffer_pool.ReturnBuffer(bufferElem)
	worker := worker_pool.GetWorkerByTaskID(taskID)
	var usage *worker_pool.ResourceUsage
	if worker != nil {
		usage = worker_pool.QueryResourceUsage(worker.GetPodName())
	} else {
		usage = &worker_pool.ResourceUsage{
			CPU:              0,
			Memory:           0,
			Storage:          0,
			StorageEphemeral: 0,
			CollectedTime:    "Task has been ended",
			Window:           0,
			Available:        false,
		}
	}

	marshal, err := json.Marshal(usage)

	if err != nil {
		log.Panic(err)
	}

	_, err = w.Write(marshal)
	if err != nil {
		log.Panic(err)
	}

}
