package handler

import (
	"Scheduler/buffer_pool"
	"Scheduler/worker_pool"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"strings"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const STATUS_LAST = "Last"

type UpdateResource struct {
	CPU        float64 `json:"CPU"`
	GPU        float64 `json:"GPU"`
	Offloading string  `json:"Offloading"`
	TaskType   string  `json:"TaskType"`
}

type CompleteTaskHandler struct {
	detWorker    *worker_pool.Worker
	fusionWorker *worker_pool.Worker

	detTaskID    string
	fusionTaskID string

	form   *multipart.Form
	status string

	deleteDETWorker    bool
	deleteFusionWorker bool

	clientAddress string

	slamIOLatency      time.Duration
	slamComputeLatency time.Duration
	detIOLatency       time.Duration
	detComputeLatency  time.Duration
	fusionLatency      time.Duration
	totalLatency       time.Duration

	cpuLimit       int
	gpuLimit       int
	fusionNodeName string

	Action []float64
}

func NewCompleteTaskHandler(
	detWorker *worker_pool.Worker,
	fusionWorker *worker_pool.Worker,
	detTaskID string,
	fusionTaskID string,
	form *multipart.Form,
	status string,
	deleteDETWorker bool,
	deleteFusionWorker bool,
	clientAddress string,
	cpuLimit int,
	gpuLimit int,
	fusionNodeName string,
	Action []float64) *CompleteTaskHandler {
	return &CompleteTaskHandler{
		detWorker:          detWorker,
		fusionWorker:       fusionWorker,
		detTaskID:          detTaskID,
		fusionTaskID:       fusionTaskID,
		form:               form,
		status:             status,
		deleteDETWorker:    deleteDETWorker,
		deleteFusionWorker: deleteFusionWorker,
		clientAddress:      clientAddress,
		cpuLimit:           cpuLimit,
		gpuLimit:           gpuLimit,
		fusionNodeName:     fusionNodeName,
		Action:             Action,
	}
}

func (handler *CompleteTaskHandler) DealWithStuck() {
	time.Sleep(time.Second * 500)

	allrunning := handler.checkPodRunning(true, true)

	if !allrunning {
		// fmt.Println("1")
		// fmt.Println("No stuck! Everything is ok")
		return
	}

	delete_wg := sync.WaitGroup{}
	delete_wg.Add(2)
	fmt.Println("Something bad is stuck, Delete Workers anyway...")
	go func() {
		defer delete_wg.Done()
		handler.detWorker.ReturnToPool(handler.detTaskID)
		handler.detWorker.DeleteWorker()
		handler.SendUpdateResourceRequest("det")
		log.Printf("det worker deleted")
	}()

	go func() {
		defer delete_wg.Done()
		handler.fusionWorker.ReturnToPool(handler.fusionTaskID)
		handler.fusionWorker.DeleteWorker()
		handler.SendUpdateResourceRequest("fusion")
		log.Printf("fusion worker deleted")
	}()

	delete_wg.Wait()
	fmt.Println("Delete Works because of stucking successfully")
	handler.sendBackResults("Something bad is stuck, Delete Workers anyway...")
}

func (handler *CompleteTaskHandler) NormalSendTask(ctx context.Context) {
	wg := sync.WaitGroup{}
	wg.Add(2)
	totalTick := time.Now()
	var detResult string
	// time.Sleep(time.Second * 30000) //Simulate stuck
	go func() {
		// start det and get det result
		detResult = handler.sendToDET()
		wg.Done()
	}()

	go func() {
		now := time.Now()
		// trigger localization
		handler.startLocalization()
		handler.slamIOLatency = time.Since(now)
		wg.Done()
	}()

	wg.Wait()
	// send det result to the fusion, fusion worker will complete
	// localization first, then do fusion, then sendback result
	fusionResult := handler.sendDETResultToFusion(detResult)
	handler.totalLatency = time.Since(totalTick)

	handler.sendBackResults(fusionResult)
}

func (handler *CompleteTaskHandler) SendTask() {

	// done := make(chan bool)
	finish := make(chan bool)
	ctx, cancel := context.WithCancel(context.Background())

	// 处理卡住的情况
	go func() {
		handler.DealWithStuck()
		cancel()
	}()

	// 处理正常情况
	go func() {
		handler.NormalSendTask(ctx)
		finish <- true
	}()

	select {
	case <-ctx.Done():
		return
	case <-finish:
		return
	}

}

func (handler *CompleteTaskHandler) checkPodRunning(checkDet bool, checkFusion bool) bool {
	detPodName := handler.detWorker.PodName
	fusionPodName := handler.fusionWorker.PodName
	podNames := []string{detPodName, fusionPodName}

	config, err := rest.InClusterConfig()
	if err != nil {
		log.Panic(err)
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		log.Panic(err)
	}

	// allRunning := false

	// for !allRunning {
	allRunning := true // assume all are running until found otherwise
	for i, podName := range podNames {
		if i == 0 && !checkDet {
			continue
		}

		if i == 1 && !checkFusion {
			continue
		}

		pod, err := clientset.CoreV1().Pods("default").Get(context.TODO(), podName, meta_v1.GetOptions{})
		if err != nil {
			// fmt.Printf("Failed to get pod: %v\n", err)
			// os.Exit(1)
			allRunning = false
		}

		switch pod.Status.Phase {
		case corev1.PodRunning:
		case corev1.PodFailed, corev1.PodSucceeded:
			// fmt.Printf("Pod %s has terminated\n", podName)
			allRunning = false
		default:
			// fmt.Printf("Pod %s is in phase: %s\n", podName, pod.Status.Phase)
			allRunning = false
		}
	}
	// if !allRunning {
	// 	fmt.Println("Not all pods are running, waiting...")
	// 	time.Sleep(time.Second * 1)
	// }
	return allRunning
}

func (handler *CompleteTaskHandler) startLocalization() {
	//log.Printf("submit to %v", handler.fusionWorker.GetIP())
	workerURL := handler.fusionWorker.GetURL("run_task")

	bufferElem := buffer_pool.GetBuffer()
	postBody := bufferElem.Buffer

	multipartWriter := multipart.NewWriter(postBody)

	if err := multipartWriter.WriteField("cmd", "slam"); err != nil {
		log.Panic(err)
	}

	if err := multipartWriter.WriteField("task_name", "fusion"); err != nil {
		log.Panic(err)
	}

	if err := multipartWriter.WriteField("task_id", handler.fusionTaskID); err != nil {
		log.Panic(err)
	}

	if err := multipartWriter.WriteField("reset", "False"); err != nil {
		log.Panic(err)
	}

	fileHeader := handler.form.File["frame"][0]
	writer, err := multipartWriter.CreateFormFile("frame", "input.png")
	if err != nil {
		log.Panic(err)
	}
	file, err := fileHeader.Open()
	if err != nil {
		log.Panic(err)
	}

	frameBytes, err := io.ReadAll(file)
	//log.Printf("Receive bytes %v", len(frameBytes))
	if err != nil {
		log.Panic(err)
	}

	_, err = writer.Write(frameBytes)
	//log.Printf("Write %v bytes. Buffer size is %v", n, postBody.Cap())
	if err != nil {
		log.Panic(err)
	}

	err = file.Close()
	if err != nil {
		log.Panic(err)
	}

	err = multipartWriter.Close()
	if err != nil {
		log.Panic(err)
	}

	allrunning := false
	for {
		for !allrunning {
			allrunning = handler.checkPodRunning(false, true)
			if !allrunning {
				fmt.Println("3")
				fmt.Println("fusion pod isn't running, wait for 1s then recheck!")
				time.Sleep(time.Second)
			}
		}

		_, err = http.Post(workerURL, multipartWriter.FormDataContentType(), postBody)
		if err != nil {
			fmt.Println(err)
		} else {
			break
		}
	}

	buffer_pool.ReturnBuffer(bufferElem)
}

func (handler *CompleteTaskHandler) sendToDET() string {
	now := time.Now()
	detHandler := GetHandler("det")

	allrunning := false
	for !allrunning {
		allrunning = handler.checkPodRunning(true, false)
		if !allrunning {
			fmt.Println("4")
			fmt.Println("det pod isn't running, wait for 1s then recheck!")
			time.Sleep(time.Second)
		}
	}

	detHandler.StartTask(handler.detWorker, handler.form, handler.detTaskID)
	handler.detIOLatency = time.Since(now)

	notifier, _ := taskFinishNotifier.Load(handler.detTaskID)

	if notifier == nil {
		// if handler.deleteDETWorker {
		// 	handler.detWorker.DeleteWorker()
		// 	// Send Request to DecisionMaker to Update the resources when pods are deleted
		// 	handler.SendUpdateResourceRequest("det")
		// 	log.Printf("det worker deleted")
		// }
		fmt.Println("Something bad is happening, Delete Workers anyway...")
		for {
			time.Sleep(time.Hour)
		}

		return "Something bad is happening, Delete Workers anyway..."
	}

	finishForm := <-notifier.(chan *multipart.Form)
	taskFinishNotifier.Delete(handler.detTaskID)

	if handler.status == STATUS_LAST {
		workerURL := handler.detWorker.GetURL("run_task")

		resetBufferElem := buffer_pool.GetBuffer()
		postBody := resetBufferElem.Buffer
		multipartWriter := multipart.NewWriter(postBody)

		if err := multipartWriter.WriteField("reset", "True"); err != nil {
			log.Panic(err)
		}

		if err := multipartWriter.WriteField("task_name", "det"); err != nil {
			log.Panic(err)
		}

		if err := multipartWriter.WriteField("task_id", handler.detTaskID); err != nil {
			log.Panic(err)
		}

		err := multipartWriter.Close()
		if err != nil {
			log.Panic(err)
		}

		allrunning := false
		for {
			for !allrunning {
				allrunning = handler.checkPodRunning(true, false)
				if !allrunning {
					fmt.Println("4")
					fmt.Println("det pod isn't running, wait for 1s then recheck!")
					time.Sleep(time.Second)
				}
			}

			_, err = http.Post(workerURL, multipartWriter.FormDataContentType(), postBody)
			if err != nil {
				fmt.Println(err)
			} else {
				break
			}
		}

		buffer_pool.ReturnBuffer(resetBufferElem)

		handler.detWorker.ReturnToPool(handler.detTaskID)
	}

	if handler.deleteDETWorker {
		handler.detWorker.DeleteWorker()
		// Send Request to DecisionMaker to Update the resources when pods are deleted
		handler.SendUpdateResourceRequest("det")
		log.Printf("det worker deleted")
	}

	var err error
	handler.detComputeLatency, err = time.ParseDuration(finishForm.Value["det_latency"][0])
	if err != nil {
		log.Panic(err)
	}

	//log.Printf("receive result of task id: %v.task id is %v, det_result is %v",
	//handler.detTaskID, len(finishForm.Value["task_id"]), len(finishForm.Value["det_result"]))

	return finishForm.Value["det_result"][0]
}

func (handler *CompleteTaskHandler) SendUpdateResourceRequest(taskType string) {
	updateResource := UpdateResource{
		CPU:        handler.Action[0],
		GPU:        handler.Action[1],
		Offloading: handler.fusionNodeName,
		TaskType:   taskType,
	}

	updateResourceJSON, err := json.Marshal(updateResource)

	if err != nil {
		log.Fatal("Failed to marshal updateResource to JSON:", err)
	}

	// 发起POST请求
	_, err = http.Post("http://localhost:8083/updateResource", "application/json", bytes.NewBuffer(updateResourceJSON))
	if err != nil {
		log.Fatalf("Scheduler send %v update resource request error:%v", updateResource.TaskType, err)
	}
}

func (handler *CompleteTaskHandler) sendDETResultToFusion(detResult string) string {
	// submit fusion task to the worker_pool
	//log.Printf("submit to %v", handler.fusionWorker.GetIP())

	workerURL := handler.fusionWorker.GetURL("run_task")

	bufferElem := buffer_pool.GetBuffer()
	postBody := bufferElem.Buffer

	multipartWriter := multipart.NewWriter(postBody)

	if err := multipartWriter.WriteField("cmd", "fusion"); err != nil {
		log.Panic(err)
	}

	if err := multipartWriter.WriteField("task_name", "fusion"); err != nil {
		log.Panic(err)
	}

	if err := multipartWriter.WriteField("task_id", handler.fusionTaskID); err != nil {
		log.Panic(err)
	}

	if err := multipartWriter.WriteField("detect_result", detResult); err != nil {
		log.Panic(err)
	}

	if err := multipartWriter.WriteField("reset", "False"); err != nil {
		log.Panic(err)
	}

	err := multipartWriter.Close()
	if err != nil {
		log.Panic(err)
	}

	notifier := make(chan *multipart.Form)
	taskFinishNotifier.Store(handler.fusionTaskID, notifier)

	allrunning := false
	for {
		for !allrunning {
			allrunning = handler.checkPodRunning(false, true)
			if !allrunning {
				fmt.Println("5")
				fmt.Println("fusion pod isn't running, wait for 1s then recheck!")
				time.Sleep(time.Second)
			}
		}

		_, err = http.Post(workerURL, multipartWriter.FormDataContentType(), postBody)
		if err != nil {
			fmt.Println(err)
		} else {
			break
		}
	}

	buffer_pool.ReturnBuffer(bufferElem)

	finishForm := <-notifier
	if len(finishForm.Value["fusion_result"]) != 1 {
		log.Panicf("len of fusion result is %v", len(finishForm.Value["fusion_result"]))
	}

	log.Println("Fusion Notified!")

	handler.slamComputeLatency, err = time.ParseDuration(finishForm.Value["slam_latency"][0])
	if err != nil {
		log.Panic(err)
	}

	handler.fusionLatency, err = time.ParseDuration(finishForm.Value["fusion_latency"][0])
	if err != nil {
		log.Panic(err)
	}

	if handler.status == STATUS_LAST {
		resetBufferElem := buffer_pool.GetBuffer()
		postBody := resetBufferElem.Buffer
		multipartWriter := multipart.NewWriter(postBody)

		if err := multipartWriter.WriteField("reset", "True"); err != nil {
			log.Panic(err)
		}

		if err := multipartWriter.WriteField("task_name", "fusion"); err != nil {
			log.Panic(err)
		}

		if err := multipartWriter.WriteField("task_id", handler.fusionTaskID); err != nil {
			log.Panic(err)
		}

		err := multipartWriter.Close()
		if err != nil {
			log.Panic(err)
		}

		allrunning := false
		for {
			for !allrunning {
				allrunning = handler.checkPodRunning(false, true)
				if !allrunning {
					fmt.Println("6")
					fmt.Println("fusion pod isn't running, wait for 1s then recheck!")
					time.Sleep(time.Second)
				}
			}

			_, err = http.Post(workerURL, multipartWriter.FormDataContentType(), postBody)
			if err != nil {
				fmt.Println(err)
			} else {
				break
			}
		}

		buffer_pool.ReturnBuffer(resetBufferElem)

		handler.fusionWorker.ReturnToPool(handler.fusionTaskID)
	}

	if handler.deleteFusionWorker {
		handler.fusionWorker.DeleteWorker()
		log.Println("Before Sending Update FFFFFusion")
		handler.SendUpdateResourceRequest("fusion")
		log.Printf("fusion worker deleted")
	}

	return finishForm.Value["fusion_result"][0]
}

func (handler *CompleteTaskHandler) sendBackResults(fusionResult string) {
	sendBackBufferElem := buffer_pool.GetBuffer()
	buffer := sendBackBufferElem.Buffer
	multipartWriter := multipart.NewWriter(buffer)

	if err := multipartWriter.WriteField("det_task_id", handler.detTaskID); err != nil {
		log.Panic(err)
	}

	if err := multipartWriter.WriteField("fusion_task_id", handler.fusionTaskID); err != nil {
		log.Panic(err)
	}

	if err := multipartWriter.WriteField("fusion_result", fusionResult); err != nil {
		log.Panic(err)
	}

	writeLatency := func(fieldName string, latency time.Duration) {
		if err := multipartWriter.WriteField(fieldName, latency.String()); err != nil {
			log.Panic(err)
		}
	}

	writeLatency("slam_compute_latency", handler.slamComputeLatency)
	writeLatency("slam_io_latency", handler.slamIOLatency)
	writeLatency("det_compute_latency", handler.detComputeLatency)
	writeLatency("det_io_latency", handler.detIOLatency)
	writeLatency("fusion_latency", handler.fusionLatency)
	writeLatency("total_latency", handler.totalLatency)

	err := multipartWriter.Close()
	if err != nil {
		log.Panic(err)
	}

	// split ip and port
	// clientIP := "http://" + strings.Split(handler.clientAddress, ":")[0]
	resultAddress := "http://localhost:8081/receive_results"

	//log.Printf("result send back to %v", resultAddress)

	_, err = http.Post(resultAddress, multipartWriter.FormDataContentType(), buffer)

	if err != nil {
		log.Panic(err)
	} else {
		log.Printf("result send back to %v", resultAddress)
	}

	buffer_pool.ReturnBuffer(sendBackBufferElem)
}

func (handler *CompleteTaskHandler) sendBackToClient(fusionResult string) {

	sendBackBufferElem := buffer_pool.GetBuffer()
	buffer := sendBackBufferElem.Buffer
	multipartWriter := multipart.NewWriter(buffer)

	if err := multipartWriter.WriteField("det_task_id", handler.detTaskID); err != nil {
		log.Panic(err)
	}

	if err := multipartWriter.WriteField("fusion_task_id", handler.fusionTaskID); err != nil {
		log.Panic(err)
	}

	if err := multipartWriter.WriteField("fusion_result", fusionResult); err != nil {
		log.Panic(err)
	}

	writeLatency := func(fieldName string, latency time.Duration) {
		if err := multipartWriter.WriteField(fieldName, latency.String()); err != nil {
			log.Panic(err)
		}
	}

	writeLatency("slam_compute_latency", handler.slamComputeLatency)
	writeLatency("slam_io_latency", handler.slamIOLatency)
	writeLatency("det_compute_latency", handler.detComputeLatency)
	writeLatency("det_io_latency", handler.detIOLatency)
	writeLatency("fusion_latency", handler.fusionLatency)
	writeLatency("total_latency", handler.totalLatency)

	err := multipartWriter.Close()
	if err != nil {
		log.Panic(err)
	}

	// split ip and port
	clientIP := "http://" + strings.Split(handler.clientAddress, ":")[0]
	resultAddress := clientIP + ":8080/complete_task"

	//log.Printf("result send back to %v", resultAddress)

	_, err = http.Post(resultAddress, multipartWriter.FormDataContentType(), buffer)
	if err != nil {
		log.Panic(err)
	}

	buffer_pool.ReturnBuffer(sendBackBufferElem)
}
