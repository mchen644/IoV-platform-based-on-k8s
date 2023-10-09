package scheduler_comm

import (
	"bytes"
	"log"
	"mime/multipart"
	"net/http"
	"time"

	"github.com/luoxiao23333/ProcessResourceLimiter/config"
	"github.com/luoxiao23333/ProcessResourceLimiter/task"
)

type Handler func(*multipart.Form, string)

var taskRunnerMap *map[string]Handler = nil

func GetHandler(taskName string) Handler {
	if taskRunnerMap == nil {
		taskRunnerMap = &map[string]Handler{
			"slam":   doSlam,
			"mcmot":  doMCMOT,
			"fusion": doFusion,
			"det":    doDET,
		}
	}

	if taskName != config.GetConfig().TaskName {
		log.Panicf("Not support %v, only support %v", taskName, config.GetConfig().TaskName)
	}

	handler, ok := (*taskRunnerMap)[taskName]
	if !ok {
		log.Printf("only support %v", taskRunnerMap)
		log.Panicf("Task name %v is not support!", taskName)
	}

	return handler
}

func doSlam(form *multipart.Form, taskID string) {
	slamTask := task.GetSlamTask()

	body := &bytes.Buffer{}
	multipartWriter := multipart.NewWriter(body)

	if form.Value["reset"][0] == "True" {
		log.Println("Reset")
		slamTask.ResetSLAM()
		return
	}

	log.Printf("Receive slam trigger, start slam")
	slamTask.Localize(form)

	slamResult := <-slamTask.OutputChan
	log.Printf("Get fusion Result: %v", slamResult)

	err := multipartWriter.WriteField("slam_result", slamResult)
	if err != nil {
		log.Panic(err)
	}

	err = multipartWriter.WriteField("task_id", taskID)
	if err != nil {
		log.Panic(err)
	}

	err = multipartWriter.Close()
	if err != nil {
		log.Panic(err)
	}

	_, err = http.Post(schedulerURL+"/slam_finish", multipartWriter.FormDataContentType(), body)
	if err != nil {
		log.Panic(err)
	}

	log.Println("Slam Task has been send back")
}

func doMCMOT(form *multipart.Form, taskID string) {
	// Complete object detection
	mcmotTask := task.NewMCMOTTask()
	imgRoad, info := mcmotTask.BlockedRun(form)

	body := &bytes.Buffer{}
	multipartWriter := multipart.NewWriter(body)
	addFileToMultipart(imgRoad+"/output.mp4",
		"video", "output.mp4", multipartWriter)
	addFileToMultipart(imgRoad+"/output.txt",
		"bbox_txt", "output.txt", multipartWriter)
	addFileToMultipart(imgRoad+"/output.xlsx",
		"bbox_xlsx", "output.xlsx", multipartWriter)

	err := multipartWriter.WriteField("container_output", info)
	if err != nil {
		log.Panic(err)
	}

	err = multipartWriter.WriteField("task_id", taskID)
	if err != nil {
		log.Panic(err)
	}

	err = multipartWriter.Close()
	if err != nil {
		log.Panic(err)
	}

	_, err = http.Post(schedulerURL+"/mcmot_finish", multipartWriter.FormDataContentType(), body)
	if err != nil {
		log.Panic(err)
	}
}

func doFusion(form *multipart.Form, taskID string) {
	fusionTask := task.GetFusionTask()

	if form.Value["reset"][0] == "True" {
		log.Println("Reset")
		fusionTask.ResetSLAM()
		return
	}

	if len(form.Value["cmd"]) != 1 {
		log.Panicf("len of cmd is %v, expected %v", len(form.Value["cmd"]))
	}

	cmd := form.Value["cmd"][0]

	if cmd == "slam" {
		log.Printf("Receive slam trigger, start slam")
		fusionTask.WriteFusionInputFile(form)
		go fusionTask.Localize(form)
	} else if cmd == "fusion" {
		detResult := fusionTask.FormattedDETResult(form.Value["detect_result"][0])
		log.Printf("Receive and format det result : [%v]", detResult)
		// wait for slam is done
		<-fusionTask.LocalizeChan
		fusionTick := time.Now()
		fusionTask.InputChan <- detResult
		fusionResult := <-fusionTask.OutputChan
		fusionLatency := time.Since(fusionTick)
		log.Printf("Get fusion Result: %v", fusionResult)

		// Write result in a thread avoid deadlock
		go func() {
			body := new(bytes.Buffer)
			multipartWriter := multipart.NewWriter(body)

			err := multipartWriter.WriteField("fusion_result", fusionResult)
			if err != nil {
				log.Panic(err)
			}

			log.Printf("Write Fusion Result")

			err = multipartWriter.WriteField("task_id", taskID)
			if err != nil {
				log.Panic(err)
			}

			err = multipartWriter.WriteField("slam_latency", fusionTask.LocalizationLatency.String())
			if err != nil {
				log.Panic(err)
			}

			err = multipartWriter.WriteField("fusion_latency", fusionLatency.String())
			if err != nil {
				log.Panic(err)
			}

			err = multipartWriter.Close()
			if err != nil {
				log.Panic(err)
			}

			_, err = http.Post(schedulerURL+"/fusion_finish", multipartWriter.FormDataContentType(), body)
			if err != nil {
				log.Panic(err)
			}
			log.Println("Result has been sent back")
		}()

	}
}

func doDET(form *multipart.Form, taskID string) {
	detTask := task.GetDetTask()

	if form.Value["reset"][0] == "True" {
		log.Println("Reset")
		detTask.ResetModel()
		return
	}

	detTask.DoDET(form)
	go func() {
		now := time.Now()
		detResult := <-detTask.OutputChan
		detLatency := time.Since(now)

		log.Printf("Get DET Result: %v", detResult)

		body := new(bytes.Buffer)
		multipartWriter := multipart.NewWriter(body)

		err := multipartWriter.WriteField("det_result", detResult)
		if err != nil {
			log.Panic(err)
		}

		log.Printf("Write DET Result")

		err = multipartWriter.WriteField("task_id", taskID)
		if err != nil {
			log.Panic(err)
		}

		err = multipartWriter.WriteField("det_latency", detLatency.String())
		if err != nil {
			log.Panic(err)
		}

		err = multipartWriter.Close()
		if err != nil {
			log.Panic(err)
		}

		_, err = http.Post(schedulerURL+"/det_finish", multipartWriter.FormDataContentType(), body)
		if err != nil {
			log.Panic(err)
		}
		log.Println("Result has been sent back")
	}()

}
