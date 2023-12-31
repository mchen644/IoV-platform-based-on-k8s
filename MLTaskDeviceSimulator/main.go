package main

import (
	"Device/utils"
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"mime/multipart"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

const MCMOTVideoPath = "MCMOT/input.avi"
const MCMOTVideoName = "input.avi"

const slamVideoPath = "slam/input.mp4"
const slamVideoName = "input.mp4"

const fusionImageDir = "fusion/data/00/"
const fusionImageName = "input.png"

// test
const detImageDir = "det/data/00/"
const detImageName = "input.png"

const slamImageDir = "slam/data/00/"
const slamImageName = "input.png"

const completTaskImageDir = "complete_task/data/00/"
const complteTaskImageName = "input.png"

const clusterURL = "http://192.168.1.101:8081"
const newTaskURL = clusterURL + "/new_task"
const completeTaskURL = clusterURL + "/complete_task"

const receivePort = ":8080"

const trainLength = 270
const episodes = 40

// map[taskID]chan bool
var fusionLockMap sync.Map
var detLockMap sync.Map
var completTaskLoackMap sync.Map
var slamTaskLockMap sync.Map

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

var waitGroup sync.WaitGroup

type CreateInfo struct {
	CpuLimits     map[string]int `json:"cpu_limit"`
	WorkerNumbers map[string]int `json:"worker_numbers"`
	TaskName      string         `json:"task_name"`
	GpuLimits     map[string]int `json:"gpu_limit"`
	GpuMemory     map[string]int `json:"gpu_memory"`
}

type router struct{}

type monitorStates struct {
	CPU string `json:"CPU"`
	GPU string `json:"GPU"`
}

func (r *router) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	switch req.URL.Path {
	case "/mcmot":
		MCMOTFinished(w, req)
	case "/slam":
		slamFinished(w, req)
	case "/fusion":
		fusionFinished(w, req)
	case "/det":
		detFinished(w, req)
	case "/complete_task":
		complteTaskFinish(w, req)
	// other router func
	default:
		http.NotFound(w, req)
	}
}

func main() {
	go func() {
		/*
			http.HandleFunc("/mcmot", MCMOTFinished)
			http.HandleFunc("/slam", slamFinished)
			http.HandleFunc("/fusion", fusionFinished)
			err := http.ListenAndServe(receivePort, nil)
			if err != nil {
				log.Panic(err)
			}*/

		server := &http.Server{
			Addr:         receivePort,
			ReadTimeout:  10 * time.Hour,
			WriteTimeout: 10 * time.Hour,
			IdleTimeout:  10 * time.Hour,
			Handler:      &router{},
		}

		if err := server.ListenAndServe(); err != nil {
			log.Panicf("listen: %s\n", err)
		}
	}()

	/*
		var nodeList = []string{"controller", "as1"}
		coreMap := map[string]float64{
			"controller": 32,
			"as1":        16,
		}
	*/

	// test SLAM
	/*
		restartScheduler()

		controllerRange := createRange(1000, 5000, 200)
		nodeList := []string{}
		taskNumber := []int{}
		for range controllerRange {
			nodeList = append(nodeList, "controller")
			expectedTask := 3
			taskNumber = append(taskNumber, expectedTask)
		}
		testSLAM(nodeList, controllerRange, taskNumber)
		as1Range := createRange(400, 3000, 200)
		nodeList = []string{}
		taskNumber = []int{}
		for range as1Range {
			nodeList = append(nodeList, "as1")
			expectedTask := 3
			taskNumber = append(taskNumber, expectedTask)
		}
		testSLAM(nodeList, as1Range, taskNumber)
	*/

	// Tesk Det
	/*
		restartScheduler()
		warmUp()
		var nodeList = []string{"gpu1", "gpu1", "gpu1"}
		var gpuLimits = []int{33, 50, 100}
		var taskNumbers = []int{3, 2, 1}
		testDET(nodeList, gpuLimits, taskNumbers)
	*/

	// query Monitor
	// queryMonitors()
	// fmt.Println("Before creating monitors!!!!!\n")
	// SendCreateMonitorsRequests()
	// fmt.Println("After creating monitors!!!!!\n")
	fileName := "arrival_data.txt"
	arrival_data := readData(fileName)

	wg := sync.WaitGroup{}
	fmt.Println(arrival_data)

	arraySum := 0

	for i := 0; i < trainLength; i++ {
		arraySum += arrival_data[i]
	}

	fmt.Println(arraySum)
	fmt.Println(trainLength)

	queueWg := sync.WaitGroup{}

	totalCount := 0
	for episode := 0; episode < episodes; episode++ {
		count := 0
		for i := 0; i < trainLength; i++ {

			for j := 0; j < arrival_data[i]; j++ {
				queueWg.Add(1)
				go func(j int) {
					defer queueWg.Done()
					SendQueueRequest()
				}(j)
			}

			queueWg.Wait()

			for j := 0; j < arrival_data[i]; j++ {
				wg.Add(1)

				go func(j int) {
					defer wg.Done()
					count += 1
					startTime := time.Now()
					SimulateVoT(count, startTime)
					totalCount += 1
					fmt.Printf("Total steps finished:%v\n", totalCount)
					// duration := time.Since(startTime)
					// log.Printf("Duration of request %v for response Received from the beginning: %v\n", count, duration.Seconds())
				}(j)
				// time.Sleep(time.Hour)
			}
			time.Sleep(40 * time.Second)
		}

		// reset environment
		// 因为device不在k8s系统中，我得发送reset环境的请求给scheduler，
		// 让scheduler在最后一个请求处理完毕之后，(因为要更新记录在decision Maker上的资源)
		// reset环境(不用delete，应该是等待直到只剩下scheduler一个pod)
		fmt.Printf("%vth episode is going to end, waiting for all requests being completed...\n", episode+1)
		wg.Wait()
		log.Printf("%vth episode ends successfully!!!!!!!!, waiting for 5s to start the next episode!\n", episode+1)
		time.Sleep(time.Second * 5)
	}

	// testCompleteTaskForAllConfig()
	for {
		time.Sleep(time.Second)
	}

	// Test if it can receive the CPU and GPU usage of the nodes
	testMonitor()
}

func SendQueueRequest() {
	body := []byte{}
	resp, err := http.Post("http://192.168.1.101:8081/Queue", "application/json", bytes.NewBuffer(body))
	if err != nil {
		log.Fatalf("POST Queue request error: %v", err)
	}
	defer resp.Body.Close()
}

func readData(fileName string) []int {
	// 打开文本文件
	file, err := os.Open(fileName)
	if err != nil {
		log.Fatal("Can not open arrival_data.txt:", err)
		return nil
	}
	defer file.Close()

	data := []int{}

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()

		parts := strings.Fields(line)
		if len(parts) >= 2 {
			value := parts[1]
			if intValue, err := strconv.Atoi(value); err == nil {
				data = append(data, intValue)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Println("Read Error:", err)
		return nil
	}

	return data
}

func SimulateVoT(count int, startTime time.Time) {
	// Restart All
	// restartScheduler()

	// create monitors on each node
	// fmt.Println("Before creating monitors!!!!!\n")
	// SendCreateMonitorsRequests()
	// fmt.Println("After creating monitors!!!!!\n")

	fmt.Printf("request %v is sending scheduling\n", count)
	metrics := SendScheduling()
	//fmt.Printf("request %v completed successfully, FusionNodeName: %s, CPULimit: %d, GPULimit: %d, AvgLatency: %f, total duration is: %v\n", count, metrics.FusionNodeName, metrics.CpuLimit, metrics.GpuLimit, metrics.AvgLatency, time.Since(startTime).Seconds())
	fmt.Printf("request %v completed successfully, FusionNodeName: %s, CPULimit: %d, GPULimit: %d, total duration is: %v\n", count, metrics.FusionNodeName, metrics.CpuLimit, metrics.GpuLimit, time.Since(startTime).Seconds())
}

func SendScheduling() *CompleteMetrics {
	bufferElem := utils.GetBuffer()
	body := bufferElem.Buffer
	multipartWriter := multipart.NewWriter(body)

	writeValue := func(key, value string) {
		err := multipartWriter.WriteField(key, value)
		if err != nil {
			log.Fatal("multipartWriter write error: ", err)
		}
	}

	writeValue("Test", "Monitor")

	err := multipartWriter.Close()
	if err != nil {
		log.Fatal("multipartWriter.Close() : ", err)
	}

	responseScheduling, err := http.Post("http://192.168.1.101:8081/scheduling", "application/json", body)

	if err != nil {
		log.Fatal("http Post scheduling error:", err)
	}

	utils.ReturnBuffer(bufferElem)

	content, err := io.ReadAll(responseScheduling.Body)
	if err != nil {
		log.Fatal("io read response error: ", err)
	}

	responseScheduling.Body.Close()

	metrics := &CompleteMetrics{}

	err = json.Unmarshal(content, metrics)

	if err != nil {
		log.Fatal("json Ummarshal error: ", err)
	}

	// fmt.Printf("NodeName: %s, CPULimit: %d, GPULimit: %d, AvgLatency: %f\n", metrics.FusionNodeName, metrics.CpuLimit,
	// 	metrics.GpuLimit, metrics.AvgLatency)
	return metrics
}

func SendCreateMonitorsRequests() {
	_, err := http.Post(clusterURL+"/create_monitors", "application/json", nil)
	if err != nil {
		log.Panic(err)
	}
}

func testMonitor() {
	bufferElem := utils.GetBuffer()
	body := bufferElem.Buffer
	multipartWriter := multipart.NewWriter(body)

	writeValue := func(key, value string) {
		err := multipartWriter.WriteField(key, value)
		if err != nil {
			log.Fatal("multipartWriter write error: ", err)
		}
	}
	writeValue("Test", "Monitor")

	err := multipartWriter.Close()
	if err != nil {
		log.Fatal("multipartWriter.Close() : ", err)
	}

	responseMonitor, err := http.Post("http://localhost:8082/monitor", "application/json", body)

	if err != nil {
		log.Fatal("http Post monitor error:", err)
	}

	utils.ReturnBuffer(bufferElem)

	content, err := io.ReadAll(responseMonitor.Body)
	if err != nil {
		log.Fatal("io read response error: ", err)
	}

	responseMonitor.Body.Close()

	resourceUsage := &monitorStates{}

	err = json.Unmarshal(content, resourceUsage)

	if err != nil {
		log.Fatal("json Ummarshal error: ", err)
	}

	fmt.Printf("CPU usage: %s, GPU usage: %s\n", resourceUsage.CPU, resourceUsage.GPU)
}

func testCompleteTaskForAllConfig() {
	var sumMetrics []CompleteMetrics

	restartScheduler()
	// time.Sleep(10 * time.Second)

	// warmUp()
	// restartScheduler()

	// create monitors on each node
	fmt.Println("Before creating monitors!!!!!\n")
	SendCreateMonitorsRequests()
	fmt.Println("After creating monitors!!!!!\n")
	time.Sleep(10 * time.Second)

	enableTestDifferentTaskNumber := false

	if enableTestDifferentTaskNumber {
		wg := sync.WaitGroup{}
		wg.Add(1)
		go func() {
			testCompleteTask("controller", 0, 50, false, context.Background())
			wg.Done()
		}()
		testCompleteTask("controller", 0, 50, true, context.Background())
		wg.Wait()

		wg = sync.WaitGroup{}
		wg.Add(2)
		for i := 0; i < 2; i++ {
			go func() {
				testCompleteTask("controller", 0, 25, false, context.Background())
				wg.Done()
			}()
		}
		testCompleteTask("controller", 0, 50, true, context.Background())
		wg.Wait()
	}

	cpuLimits := map[string][]int{
		"controller": createRange(1000, 6000, 200),
		"as1":        createRange(400, 3000, 100),
	}

	workerNumbers := map[int]int{
		25: 4, 33: 3, 50: 2, 100: 1,
	}

	for _, fusionNodeName := range []string{"as1", "controller"} {
		for _, gpuLimit := range []int{33, 50, 100} {
			for _, cpuLimit := range cpuLimits[fusionNodeName] {

				log.Printf("for cpu[%v] gpu[%v] fusion node[%v]",
					cpuLimit, gpuLimit, fusionNodeName)

			startPoint:
				done := make(chan bool)
				ctx, cancel := context.WithCancel(context.Background())

				wg := sync.WaitGroup{}
				workerNumber := workerNumbers[gpuLimit]
				wg.Add(workerNumber)
				for i := 0; i < workerNumber-1; i++ {
					go func(ctx context.Context) {
						testCompleteTask(fusionNodeName, cpuLimit, gpuLimit, false, ctx)
						wg.Done()
					}(ctx)
				}

				go func(ctx context.Context) {
					metrics := testCompleteTask(fusionNodeName, cpuLimit, gpuLimit, true, ctx)
					sumMetrics = append(sumMetrics, metrics)
					wg.Done()
				}(ctx)

				go func() {
					wg.Wait()
					done <- true
				}()

				select {
				case <-done:
					log.Println("Task Done")
				case <-time.After(500 * time.Second):
					log.Printf("Timeout! Restart and Warm Up Again")
					cancel()
					restartScheduler()
					warmUp()
					restartScheduler()
					goto startPoint
				}

				time.Sleep(5 * time.Second)
			}
			restartScheduler()
			time.Sleep(20 * time.Second)
		}

		file, err := os.Create(fmt.Sprintf("complete_task/sum_metrics_%v.csv", fusionNodeName))
		if err != nil {
			log.Panic(err)
		}

		_, err = file.Write(utils.MarshalCSV(sumMetrics))
		if err != nil {
			log.Panic(err)
		}

		if err = file.Close(); err != nil {
			log.Panic(err)
		}
	}

	/*
		for _, fusionNodeName := range []string{"as1"} {
			for cpuLimit := 5000; cpuLimit <= 5000; cpuLimit += 100 {
				for _, gpuLimit := range []int{16, 20, 25, 33, 50, 100} {
					log.Printf("for cpu[%v] gpu[%v] fusion node[%v]",
						cpuLimit, gpuLimit, fusionNodeName)
					metrics := testCompleteTask(fusionNodeName, cpuLimit, gpuLimit)
					sumMetrics = append(sumMetrics, metrics)
					time.Sleep(10 * time.Second)
				}
			}
		}
	*/
}

func createRange(lower int, upper int, step int) []int {
	upper += step
	nums := make([]int, (upper-lower)/step)
	for i := range nums {
		nums[i] = lower + step*i
	}
	return nums
}

func warmUp() {
	for warmUpIter := 0; warmUpIter < 2; warmUpIter++ {
	warmUpPoint:
		fmt.Printf("\r warm up %v/%v \n", warmUpIter+1, 2)
		wg := sync.WaitGroup{}
		wg.Add(3)
		ctx, cancel := context.WithCancel(context.Background())
		for i := 0; i < 3; i++ {
			go func(ctx context.Context) {
				testCompleteTask("controller", 0, 33, false, ctx)
				wg.Done()
			}(ctx)
		}
		done := make(chan bool, 1)
		go func() {
			wg.Wait()
			done <- true
		}()

		select {
		case <-done:
			continue
		case <-time.After(300 * time.Second):
			cancel()
			log.Printf("Timeout in warm up! Restart and Warm Up Again")
			restartScheduler()
			goto warmUpPoint
		}
	}
	fmt.Println()
}

func restartScheduler() {
	// For testing if the monitors can be created successfully
	// time.Sleep(100 * time.Second)

	cmd := exec.Command("kubectl", "delete", "pod", "--all")

	// set env
	env := []string{"KUBECONFIG=/etc/kubernetes/admin.conf"}
	cmd.Env = append(cmd.Env, env...)

	var out bytes.Buffer
	cmd.Stdout = &out
	err := cmd.Run()
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	fmt.Println(out.String())
	log.Println("Restart Done!. Wait for 20 secs")
	completTaskLoackMap = sync.Map{}
	time.Sleep(20 * time.Second)
}

func testDET(nodeList []string, gpuLimits, taskNumbers []int) {

	nodeName := "gpu1"

	const imageLength = 465
	var metricsMap = make(map[int]MetricsStat)

	for index, nodeName := range nodeList {
		gpuLimit := gpuLimits[index]
		taskNumber := taskNumbers[index]
		detailMetricsMap := make(map[string]*ResourceUsage)
		log.Printf("creating pod... wait for det start up\n")
		createGPUWorkers(gpuLimit, nodeName, taskNumber, "det")
		waitTime := int(time.Second) * 15
		log.Printf("Wait for extra %v", time.Duration(waitTime))
		time.Sleep(time.Duration(waitTime))
		taskWG := sync.WaitGroup{}
		taskWG.Add(taskNumber)
		taskIDs := make([]string, taskNumber)
		for i := range taskIDs {
			taskIDs[i] = "-1"
		}
		var latencies []time.Duration
		var detResult string
		for i := 0; i < taskNumber; i++ {
			go func(i int) {
				startTime := time.Now()
				queryTime := time.Now()
				eachFrameTime := time.Now()
				for imageIndex := 0; imageIndex < imageLength; imageIndex++ {
					if i == taskNumber-1 {
						fmt.Printf("\rDoing Image %v/%v", imageIndex+1, imageLength)
					}
					var status string
					if taskIDs[i] == "-1" {
						status = "Begin"
					} else if imageIndex == imageLength-1 {
						status = "Last"
						notifier, _ := detLockMap.Load(taskIDs[i])
						result := <-notifier.(chan string)
						if i == taskNumber-1 {
							detResult = detResult + result
							latencies = append(latencies, time.Since(eachFrameTime))
						}
					} else {
						status = "Running"
						notifier, _ := detLockMap.LoadOrStore(taskIDs[i], make(chan string))
						result := <-notifier.(chan string)
						if i == taskNumber-1 {
							detResult = detResult + result
							latencies = append(latencies, time.Since(eachFrameTime))
						}
					}

					duration := time.Since(startTime)
					needToWait := time.Duration(int64(float64(imageIndex)*(1000./30.))) * time.Millisecond
					if duration < needToWait {
						diff := needToWait - duration
						log.Printf("Wait %v", time.Duration(diff))
						time.Sleep(time.Duration(diff))
					}

					imagePath := detImageDir + utils.IntToNDigitsString(imageIndex, 6) + ".png"
					taskIDs[i] = sendDETRequest(nodeName, imagePath, status, taskIDs[i])
					eachFrameTime = time.Now()

					if status == "Running" {
						queryDuration := time.Since(queryTime)
						if i == taskNumber-1 && queryDuration.Seconds() >= 2 {
							//usage := queryMetrics(taskIDs[i])
							//detailMetricsMap[usage.CollectedTime] = usage
							queryTime = time.Now()
						}
					} else if status == "Last" {
						notifier, _ := detLockMap.Load(taskIDs[i])
						result := <-notifier.(chan string)
						if i == taskNumber-1 {
							detResult = detResult + result
						}
						detLockMap.Delete(taskIDs[i])
					}

				}
				taskWG.Done()
			}(i)
		}
		taskWG.Wait()

		avgLatency := 0.
		for _, l := range latencies {
			avgLatency += float64(l.Milliseconds())
		}
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

		file, err := os.Create(fmt.Sprintf("det/result/%v_%v.csv",
			nodeName, gpuLimit))
		if err != nil {
			log.Panic(err)
		}

		_, err = file.Write([]byte(detResult))
		if err != nil {
			log.Panic(err)
		}

		if err = file.Close(); err != nil {
			log.Panic(err)
		}

		file, err = os.Create(fmt.Sprintf("det/detail_metrics/%v_%v_%v.csv",
			"det", nodeName, gpuLimit))
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

		file, err = os.Create(fmt.Sprintf("det/detail_metrics/latencies_%v_%v_%v.csv",
			"det", nodeName, gpuLimit))
		if err != nil {
			log.Panic(err)
		}

		var latenciesString string
		for _, latency := range latencies {
			latenciesString = latenciesString + fmt.Sprintf("%v\n", latency.Milliseconds())
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
			TaskNumber:     taskNumber,
		}
	}

	var results []MetricsStat
	for _, metrics := range metricsMap {
		log.Printf("Node:%v, metrics data is %v",
			"gpu1", metrics)
		results = append(results, metrics)
	}

	file, err := os.Create("det/detMetrics_" + nodeName + ".csv")
	if err != nil {
		panic(err)
	}
	defer func(file *os.File) {
		err = file.Close()
		if err != nil {
			log.Panic(err)
		}
	}(file)

	marshalData := utils.MarshalCSV(results)

	_, err = file.Write(marshalData)
	if err != nil {
		log.Panic(err)
	}
}

func testFusion(nodeList []string, coreMap map[string]float64) {
	detectResult := utils.SplitDetectResult(fusionImageDir + "detect_result.txt")

	nodeWG := sync.WaitGroup{}
	nodeWG.Add(2)
	log.Printf("Totally got %v detect result", len(detectResult))
	for _, nodeName := range nodeList {
		var metricsMap = make(map[int]MetricsStat)
		go func(cores float64, nodeName string) {
			for cpuLimit := 1000; cpuLimit <= 6500; cpuLimit += 100 {
				detailMetricsMap := make(map[string]*ResourceUsage)
				taskNumber := int(math.Floor((cores * 1000 * 0.8) / float64(cpuLimit)))
				taskNumber = 1
				log.Printf("creating pod... wait for SLAM start up\n")
				createWorkers(cpuLimit, nodeName, taskNumber, "fusion")
				waitTime := taskNumber * int(time.Second) * 60
				log.Printf("Wait for extra %v", time.Duration(waitTime))
				time.Sleep(time.Duration(waitTime))
				taskWG := sync.WaitGroup{}
				taskWG.Add(taskNumber)
				taskIDs := make([]string, taskNumber)
				for i := range taskIDs {
					taskIDs[i] = "-1"
				}
				var latencies []time.Duration
				for i := 0; i < taskNumber; i++ {
					go func(i int) {
						startTime := time.Now()
						queryTime := time.Now()
						eachFrameTime := time.Now()
						for imageIndex := 0; imageIndex < len(detectResult); imageIndex++ {
							var status string
							if taskIDs[i] == "-1" {
								status = "Begin"
							} else if imageIndex == len(detectResult)-1 {
								status = "Last"
								notifier, _ := fusionLockMap.Load(taskIDs[i])
								<-notifier.(chan bool)
								if i == taskNumber-1 {
									latencies = append(latencies, time.Since(eachFrameTime))
								}
							} else {
								status = "Running"
								notifier, _ := fusionLockMap.LoadOrStore(taskIDs[i], make(chan bool))
								<-notifier.(chan bool)
								if i == taskNumber-1 {
									latencies = append(latencies, time.Since(eachFrameTime))
								}
							}

							duration := time.Since(startTime)
							needToWait := time.Duration(int64(float64(imageIndex)*(1000./30.))) * time.Millisecond
							if duration < needToWait {
								diff := needToWait - duration
								log.Printf("Wait %v", time.Duration(diff))
								time.Sleep(time.Duration(diff))
							}

							imagePath := fusionImageDir + utils.IntToNDigitsString(imageIndex, 6) + ".png"
							taskIDs[i] = sendFusionRequest(nodeName, imagePath, detectResult[imageIndex],
								status, taskIDs[i])
							eachFrameTime = time.Now()

							if status == "Running" {
								queryDuration := time.Since(queryTime)
								if i == taskNumber-1 && queryDuration.Seconds() >= 2 {
									usage := queryMetrics(taskIDs[i])
									detailMetricsMap[usage.CollectedTime] = usage
									queryTime = time.Now()
								}
							} else if status == "Last" {
								notifier, _ := fusionLockMap.Load(taskIDs[i])
								<-notifier.(chan bool)
								fusionLockMap.Delete(taskIDs[i])
							}

						}
						taskWG.Done()
					}(i)
				}
				taskWG.Wait()

				avgLatency := 0.
				for _, l := range latencies {
					avgLatency += float64(l.Milliseconds())
				}
				avgLatency = avgLatency / float64(len(latencies))

				log.Printf("Avg latency is %v ms for cpu limits: %v", avgLatency, cpuLimit)
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

				file, err := os.Create(fmt.Sprintf("fusion/detail_metrics/%v_%v_%v.csv",
					"fusion", nodeName, cpuLimit))
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

				metricsMap[cpuLimit] = MetricsStat{
					CPULimit:       strconv.Itoa(cpuLimit),
					AvgLatency:     time.Duration(avgLatency) * time.Millisecond,
					AvgCPUUsage:    avgCPU,
					MaxCPUUsage:    maxCPU,
					AvgMemoryUsage: avgMem,
					MaxMemoryUsage: maxMem,
					HighLoadRatio:  float64(len(detailMetricsMap)) / float64(count),
					TaskNumber:     taskNumber,
				}

			}

			var results []MetricsStat
			for cpu := 1000; cpu <= 6500; cpu += 100 {
				metrics := metricsMap[cpu]
				log.Printf("Node:%v, metrics data is %v",
					nodeName, metrics)
				results = append(results, metrics)
			}

			file, err := os.Create("fusion/fusionMetrics_" + nodeName + ".csv")
			if err != nil {
				panic(err)
			}
			defer func(file *os.File) {
				err = file.Close()
				if err != nil {
					log.Panic(err)
				}
			}(file)

			marshalData := utils.MarshalCSV(results)

			_, err = file.Write(marshalData)
			if err != nil {
				log.Panic(err)
			}
			nodeWG.Done()
		}(coreMap[nodeName], nodeName)
	}
	nodeWG.Wait()
}

func testSLAM(nodeList []string, cpuLimits, taskNumbers []int) {
	const imageLength = 465
	var metricsResult []MetricsStat

	for index, nodeName := range nodeList {
		cpuLimit := cpuLimits[index]
		taskNumber := taskNumbers[index]
		detailMetricsMap := make(map[string]*ResourceUsage)
		log.Printf("creating pod... wait for slam start up\n")
		createWorkers(cpuLimit, nodeName, taskNumber, "slam")
		waitTime := int(time.Second) * 10
		log.Printf("Wait for extra %v", time.Duration(waitTime))
		time.Sleep(time.Duration(waitTime))
		taskWG := sync.WaitGroup{}
		taskWG.Add(taskNumber)
		taskIDs := make([]string, taskNumber)
		slamResult := ""
		for i := range taskIDs {
			taskIDs[i] = "-1"
		}
		var latencies []time.Duration
		for i := 0; i < taskNumber; i++ {
			go func(i int) {
				startTime := time.Now()
				queryTime := time.Now()
				eachFrameTime := time.Now()
				for imageIndex := 0; imageIndex < imageLength; imageIndex++ {
					if i == taskNumber-1 {
						fmt.Printf("\r Doing Image %v/%v", imageIndex+1, imageLength)
					}
					var status string
					if taskIDs[i] == "-1" {
						status = "Begin"
					} else if imageIndex == imageLength-1 {
						status = "Last"
						notifier, _ := slamTaskLockMap.Load(taskIDs[i])
						result := <-notifier.(chan string)
						slamResult = slamResult + result
						if i == taskNumber-1 {
							latencies = append(latencies, time.Since(eachFrameTime))
						}
					} else {
						status = "Running"
						notifier, _ := slamTaskLockMap.LoadOrStore(taskIDs[i], make(chan string))
						result := <-notifier.(chan string)
						slamResult = slamResult + result
						if i == taskNumber-1 {
							latencies = append(latencies, time.Since(eachFrameTime))
						}
					}

					duration := time.Since(startTime)
					needToWait := time.Duration(int64(float64(imageIndex)*(1000./30.))) * time.Millisecond
					if duration < needToWait {
						diff := needToWait - duration
						log.Printf("Wait %v", time.Duration(diff))
						time.Sleep(time.Duration(diff))
					}

					imagePath := slamImageDir + utils.IntToNDigitsString(imageIndex, 6) + ".png"
					taskIDs[i] = sendSlamRequest(nodeName, imagePath, status, taskIDs[i])
					eachFrameTime = time.Now()

					if status == "Running" {
						queryDuration := time.Since(queryTime)
						if i == taskNumber-1 && queryDuration.Seconds() >= 2 {
							//usage := queryMetrics(taskIDs[i])
							//detailMetricsMap[usage.CollectedTime] = usage
							queryTime = time.Now()
						}
					} else if status == "Last" {
						notifier, _ := slamTaskLockMap.Load(taskIDs[i])
						result := <-notifier.(chan string)
						slamResult = slamResult + result
						slamTaskLockMap.Delete(taskIDs[i])
					}

				}
				taskWG.Done()
			}(i)
		}
		taskWG.Wait()

		avgLatency := 0.
		for _, l := range latencies {
			avgLatency += float64(l.Milliseconds())
		}
		avgLatency = avgLatency / float64(len(latencies))

		log.Printf("Avg latency is %v ms for cpu limits: %v", avgLatency, cpuLimit)
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

		file, err := os.Create(fmt.Sprintf("slam/detail_metrics/%v_%v_%v.csv",
			"slam", nodeName, cpuLimit))
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

		file, err = os.Create(fmt.Sprintf("slam/detail_metrics/latencies_%v_%v_%v.csv",
			"slam", nodeName, cpuLimit))
		if err != nil {
			log.Panic(err)
		}

		var latenciesString string
		for _, latency := range latencies {
			latenciesString = latenciesString + fmt.Sprintf("%v\n", latency.Milliseconds())
		}

		_, err = file.Write([]byte(latenciesString))
		if err != nil {
			log.Panic(err)
		}

		if err = file.Close(); err != nil {
			log.Panic(err)
		}

		log.Printf("Max cpu usage %v", maxCPU)

		file, err = os.Create(fmt.Sprintf("slam/result/%v_%v.csv",
			nodeName, cpuLimit))
		if err != nil {
			log.Panic(err)
		}

		file.Write([]byte(slamResult))
		if err = file.Close(); err != nil {
			log.Panic(err)
		}

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

		metricsResult = append(metricsResult, MetricsStat{
			CPULimit:       "0",
			AvgLatency:     time.Duration(avgLatency) * time.Millisecond,
			AvgCPUUsage:    avgCPU,
			MaxCPUUsage:    maxCPU,
			AvgMemoryUsage: avgMem,
			MaxMemoryUsage: maxMem,
			HighLoadRatio:  float64(len(detailMetricsMap)) / float64(count),
			TaskNumber:     taskNumber,
		})

		file, err = os.Create("slam/detail_metrics/" + nodeName + ".csv")
		if err != nil {
			panic(err)
		}
		defer func(file *os.File) {
			err = file.Close()
			if err != nil {
				log.Panic(err)
			}
		}(file)

		marshalData := utils.MarshalCSV(metricsResult)

		_, err = file.Write(marshalData)
		if err != nil {
			log.Panic(err)
		}
	}
}

func testSingleVideo(nodeList []string, coreMap map[string]float64) {
	waitGroup.Add(2)

	for _, nodeName := range nodeList {
		go func(nodeName string, cores float64) {
			// test from 1000 to 7900 and unlimit
			var results []MetricsStat
			// map[int]MetricsStat
			var metricsMap = make(map[int]MetricsStat)
			for cpu := 1000; cpu <= 6000; cpu += 100 {
				var cpuLimit string
				if cpu == 6000 {
					cpuLimit = "0"
				} else {
					cpuLimit = strconv.Itoa(cpu) + "m"
				}
				var slamID string
				var slamIDs []string
				wg := sync.WaitGroup{}
				lock := sync.Mutex{}
				taskNumber := int(math.Floor((cores * 1000 * 0.8) / float64(cpu)))
				//taskNumber = 1
				wg.Add(taskNumber)
				for i := 0; i < taskNumber; i++ {
					go func() {
						//slamID = sendSlamRequest(cpuLimit, nodeName)
						lock.Lock()
						slamIDs = append(slamIDs, slamID)
						lock.Unlock()
						wg.Done()
					}()
				}
				wg.Wait()
				slamID = slamIDs[len(slamIDs)-1]
				start := time.Now()
				dataMap := make(map[string]*ResourceUsage)
				var end time.Duration
				for {
					usage := queryMetrics(slamID)
					dataMap[usage.CollectedTime] = usage
					if usage.Available == false && usage.CollectedTime == "Task has been ended" {
						end = time.Since(start)
						for {
							done := true
							for _, id := range slamIDs {
								if id != slamID {
									check := queryMetrics(slamID)
									if !(check.Available == false && check.CollectedTime == "Task has been ended") {
										done = false
									}
								}
							}
							if done {
								goto collectedEnd
							}
						}
					}
					time.Sleep(1 * time.Second)
				}
			collectedEnd:
				log.Printf("Execution time is %v for cpu limits: %v", end, cpu)
				log.Printf("Wait for 1 min to let all pod deleted")
				time.Sleep(1 * time.Minute)
				var maxCPU int64 = 0
				var maxMem int64 = 0
				var detailResults []ResourceUsage
				for _, item := range dataMap {
					if item.CPU > maxCPU {
						maxCPU = item.CPU
					}
					if item.Memory > maxMem {
						maxMem = item.Memory
					}
					detailResults = append(detailResults, *item)
				}

				file, err := os.Create(fmt.Sprintf("slam/detail_metrics/%v_%v_%v.csv",
					"slam", nodeName, cpuLimit))
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

				log.Printf("Max cpu usage %v", maxCPU)

				var avgCPU float64 = 0
				var avgMem float64 = 0
				count := 0
				for _, item := range dataMap {
					if float64(item.CPU) >= float64(maxCPU)*0.8 {
						avgCPU += float64(item.CPU)
						avgMem += float64(item.Memory)
						count += 1
					}
				}
				avgCPU /= float64(count)
				avgMem /= float64(count)

				metricsMap[cpu] = MetricsStat{
					CPULimit:       cpuLimit,
					AvgLatency:     end,
					AvgCPUUsage:    avgCPU,
					MaxCPUUsage:    maxCPU,
					AvgMemoryUsage: avgMem,
					MaxMemoryUsage: maxMem,
					HighLoadRatio:  float64(len(dataMap)) / float64(count),
					TaskNumber:     taskNumber,
				}
			}

			for cpu := 1000; cpu <= 6000; cpu += 100 {
				metrics, _ := metricsMap[cpu]
				log.Printf("Node:%v, metrics data is %v",
					nodeName, metrics)
				results = append(results, metrics)
			}

			file, err := os.Create("slam/slamMetrics_" + nodeName + ".csv")
			if err != nil {
				panic(err)
			}
			defer func(file *os.File) {
				err = file.Close()
				if err != nil {
					log.Panic(err)
				}
			}(file)

			marshalData := utils.MarshalCSV(results)

			_, err = file.Write(marshalData)
			if err != nil {
				log.Panic(err)
			}

			waitGroup.Done()
		}(nodeName, coreMap[nodeName])
	}

	waitGroup.Wait()
}

// ResourceUsage
// Storage 和持久卷绑定，pod删除不消失
// StorageEphemeral pod删除就释放
// Measure resource in range [ CollectedTime - Window, CollectedTime ]
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

type CompleteTaskInfo struct {
	DETNodeName        string `json:"det_node_name"`
	DETTaskID          string `json:"det_task_id"`
	FusionNodeName     string `json:"fusion_node_name"`
	FusionTaskID       string `json:"fusion_task_id"`
	Status             string `json:"status"`
	DeleteDETWorker    bool   `json:"delete_det_worker"`
	DeleteFusionWorker bool   `json:"delete_fusion_worker"`
}

type CompleteMetrics struct {
	FusionNodeName string  `csv:"fusion_node_name"`
	GpuLimit       int     `csv:"gpu_limit"`
	CpuLimit       int     `csv:"cpu_limit"`
	AvgLatency     float64 `csv:"avg_latency"`
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

func testCompleteTask(fusionNodeName string, cpuLimit, gpuLimit int,
	writeResult bool, ctx context.Context) CompleteMetrics {
	fusionResult := ""

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
			"gpu1": 4000,
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
	}

	//log.Printf("creating pod... wait for det start up\n")
	createGeneralWorkers(detWorkerCreateInfo)
	createGeneralWorkers(fusionWorkerCreateInfo)

	// const imageLength = 465
	const imageLength = 465
	var metricsMap = make(map[int]MetricsStat)

	detailMetricsMap := make(map[string]*ResourceUsage)

	waitTime := int(time.Second) * 30
	if writeResult {
		log.Printf("Wait for extra %v", time.Duration(waitTime))
	}
	time.Sleep(time.Duration(waitTime))

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
			fmt.Printf("\rDoing image %v/%v", imageIndex+1, imageLength)
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
		fmt.Printf("The length of latencies is %d\n", len(latencies))
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

func createGeneralWorkers(info *CreateInfo) {
	rawInfo, err := json.Marshal(info)
	if err != nil {
		log.Panic(err)
	}

	buffer := bytes.NewBuffer(rawInfo)

	//log.Printf("sent info is %v", string(rawInfo))

	_, err = http.Post(clusterURL+"/create_workers", "application/json", buffer)
	if err != nil {
		log.Panic(err)
	}
}

func createGPUWorkers(gpu int, nodeName string, taskNumber int, taskName string) {
	info := CreateInfo{
		CpuLimits: map[string]int{
			nodeName: 0,
		},
		WorkerNumbers: map[string]int{
			nodeName: taskNumber,
		},
		TaskName: taskName,
		GpuLimits: map[string]int{
			nodeName: gpu,
		},
		GpuMemory: map[string]int{
			nodeName: 3000,
		},
	}

	rawInfo, err := json.Marshal(&info)
	if err != nil {
		log.Panic(err)
	}

	buffer := bytes.NewBuffer(rawInfo)

	log.Printf("sent info is %v", string(rawInfo))

	_, err = http.Post(clusterURL+"/create_workers", "application/json", buffer)
	if err != nil {
		log.Panic(err)
	}
}

func createWorkers(cpu int, nodeName string, taskNumber int, taskName string) {
	info := CreateInfo{
		CpuLimits: map[string]int{
			nodeName: cpu,
		},
		WorkerNumbers: map[string]int{
			nodeName: taskNumber,
		},
		TaskName: taskName,
		GpuLimits: map[string]int{
			nodeName: 0,
		},
		GpuMemory: map[string]int{
			nodeName: 0,
		},
	}

	rawInfo, err := json.Marshal(&info)
	if err != nil {
		log.Panic(err)
	}

	buffer := bytes.NewBuffer(rawInfo)

	log.Printf("sent info is %v", string(rawInfo))

	_, err = http.Post(clusterURL+"/create_workers", "application/json", buffer)
	if err != nil {
		log.Panic(err)
	}
}

func updateCPU(cpu int, nodeName string) {
	updateInfo := fmt.Sprintf("%v:%v", nodeName, cpu)
	_, err := http.Post(clusterURL+"/update_cpu", "text/plain", strings.NewReader(updateInfo))
	if err != nil {
		log.Panic(err)
	}
}

func queryMetrics(taskID string) *ResourceUsage {
	bufferElem := utils.GetBuffer()
	buffer := bufferElem.Buffer
	_, err := buffer.Write([]byte(taskID))
	if err != nil {
		log.Panic(err)
	}
	response, err := http.Post(clusterURL+"/query_metric", "application/json", buffer)
	if err != nil {
		log.Panic(err)
	}

	data, err := io.ReadAll(response.Body)
	if err != nil {
		log.Panic(err)
	}
	usage := &ResourceUsage{}
	err = json.Unmarshal(data, usage)
	if err != nil {
		log.Panic(err)
	}

	response.Body.Close()
	utils.ReturnBuffer(bufferElem)

	return usage
}

func fusionFinished(w http.ResponseWriter, r *http.Request) {
	multipartReader, err := r.MultipartReader()
	if err != nil {
		log.Panic(err)
	}

	form, err := multipartReader.ReadForm(1024 * 1024 * 100)
	if err != nil {
		log.Panic(err)
	}

	taskID := form.Value["task_id"][0]
	//_ = form.Value["fusion_result"][0]

	notifier, ok := fusionLockMap.LoadOrStore(taskID, make(chan bool, 1))
	if !ok {
		log.Panic("Not Load chan")
	}

	notifier.(chan bool) <- true

	//log.Printf("Task id %v, result:\n%v", taskID, result)

	_, err = w.Write([]byte("OK"))
	if err != nil {
		log.Panic(err)
	}
}

func detFinished(w http.ResponseWriter, r *http.Request) {
	multipartReader, err := r.MultipartReader()
	if err != nil {
		log.Panic(err)
	}

	form, err := multipartReader.ReadForm(1024 * 1024 * 100)
	if err != nil {
		log.Panic(err)
	}

	taskID := form.Value["task_id"][0]

	notifier, ok := detLockMap.LoadOrStore(taskID, make(chan string, 1))
	if !ok {
		log.Panic("Not Load chan")
	}

	detResult := form.Value["det_result"][0]

	notifier.(chan string) <- detResult

	//log.Printf("Task id %v, result:\n%v", taskID, result)

	_, err = w.Write([]byte("OK"))
	if err != nil {
		log.Panic(err)
	}
}

func complteTaskFinish(w http.ResponseWriter, r *http.Request) {
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

func slamFinished(w http.ResponseWriter, r *http.Request) {
	multipartReader, err := r.MultipartReader()
	if err != nil {
		log.Panic(err)
	}

	form, err := multipartReader.ReadForm(1024 * 1024 * 2)
	if err != nil {
		log.Panic(err)
	}

	taskID := form.Value["task_id"][0]

	notifier, ok := slamTaskLockMap.LoadOrStore(taskID, make(chan string, 1))
	if !ok {
		log.Panic("Not Load chan")
	}

	result := form.Value["slam_result"][0]
	notifier.(chan string) <- result

	//log.Printf("Task id %v, result:\n%v", taskID, result)

	_, err = w.Write([]byte("OK"))
	if err != nil {
		log.Panic(err)
	}
}

func MCMOTFinished(w http.ResponseWriter, r *http.Request) {
	multipartReader, err := r.MultipartReader()
	if err != nil {
		log.Panic(err)
	}

	form, err := multipartReader.ReadForm(1024 * 1024 * 100)
	if err != nil {
		log.Panic(err)
	}

	log.Println("Task completed!")
	saveFile("video", "MCMOT/output.mp4", form)
	saveFile("bbox_txt", "MCMOT/output.txt", form)
	saveFile("bbox_xlsx", "MCMOT/output.xlsx", form)

	log.Println(form.Value["container_output"][0])

	_, err = w.Write([]byte("OK"))
	if err != nil {
		log.Panic(err)
	}
}

func sendODRequest() {
	body := &bytes.Buffer{}
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

	writeValue("task_name", "mcmot")

	writeFile("video", MCMOTVideoPath, MCMOTVideoName)

	err := multipartWriter.Close()
	if err != nil {
		log.Panic(err)
	}

	postResp, err := http.Post(newTaskURL, multipartWriter.FormDataContentType(), body)
	if err != nil {
		log.Panic(err)
	}

	_, err = io.ReadAll(postResp.Body)
	if err != nil {
		log.Panic(err)
	}

	//log.Println(string(content))

}

func sendSlamRequest(nodeName, imagePath, status, taskID string) string {
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

	writeValue("task_name", "slam")
	writeValue("node_name", nodeName)
	writeValue("status", status)
	writeValue("task_id", taskID)
	if status == "Last" {
		writeValue("delete", "True")
	}

	writeFile("frame", imagePath, fusionImageName)

	err := multipartWriter.Close()
	if err != nil {
		log.Panic(err)
	}

	postResp, err := http.Post(newTaskURL, multipartWriter.FormDataContentType(), body)
	if err != nil {
		log.Panic(err)
	}

	utils.ReturnBuffer(bufferElem)

	content, err := io.ReadAll(postResp.Body)
	if err != nil {
		log.Panic(err)
	}

	postResp.Body.Close()

	//log.Println(string(content))

	return string(content)

}

func sendFusionRequest(nodeName, imagePath, detectResult, status, taskID string) string {
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

	writeValue("task_name", "fusion")
	writeValue("node_name", nodeName)
	writeValue("detect_result", detectResult)
	writeValue("status", status)
	writeValue("task_id", taskID)
	if status == "Last" {
		writeValue("delete", "True")
	}

	writeFile("frame", imagePath, fusionImageName)

	err := multipartWriter.Close()
	if err != nil {
		log.Panic(err)
	}

	postResp, err := http.Post(newTaskURL, multipartWriter.FormDataContentType(), body)
	if err != nil {
		log.Panic(err)
	}

	utils.ReturnBuffer(bufferElem)

	content, err := io.ReadAll(postResp.Body)
	if err != nil {
		log.Panic(err)
	}

	postResp.Body.Close()

	//log.Println(string(content))

	return string(content)

}

func sendDETRequest(nodeName, imagePath, status, taskID string) string {
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

	writeValue("task_name", "det")
	writeValue("node_name", nodeName)
	writeValue("status", status)
	writeValue("task_id", taskID)
	if status == "Last" {
		writeValue("delete", "True")
	}

	writeFile("frame", imagePath, fusionImageName)

	err := multipartWriter.Close()
	if err != nil {
		log.Panic(err)
	}

	postResp, err := http.Post(newTaskURL, multipartWriter.FormDataContentType(), body)
	if err != nil {
		log.Panic(err)
	}

	utils.ReturnBuffer(bufferElem)

	content, err := io.ReadAll(postResp.Body)
	if err != nil {
		log.Panic(err)
	}

	postResp.Body.Close()

	//log.Println(string(content))

	return string(content)

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

	writeFile("frame", imagePath, fusionImageName)

	err = multipartWriter.Close()
	if err != nil {
		log.Panic(err)
	}

	/*
		postResp, err := http.Post(completeTaskURL, multipartWriter.FormDataContentType(), body)
		if err != nil {
			log.Panic(err)
		}*/

	req, err := http.NewRequestWithContext(ctx, "POST", completeTaskURL, body)
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

	detTaskID := results[0]
	fusionTaskID := results[1]

	return detTaskID, fusionTaskID
}

func saveFile(fieldName, fileName string, form *multipart.Form) {
	file, err := form.File[fieldName][0].Open()
	if err != nil {
		log.Panic(err)
	}

	newFile, err := os.Create(fileName)
	if err != nil {
		log.Panic(err)
	}
	_, err = io.Copy(newFile, file)
	if err != nil {
		log.Panic(err)
	}

	err = file.Close()
	if err != nil {
		log.Panic(err)
	}
	err = newFile.Close()
	if err != nil {
		log.Panic(err)
	}
}
