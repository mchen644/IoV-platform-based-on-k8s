package task

import (
	"io"
	"log"
	"mime/multipart"
	"os"
	"strings"
	"sync"
	"time"
)

// 1. start slam
// 2. when a frame in, put in the specific path
// 3. wait for output

var (
	fusionCMD  = "/ORB_SLAM2/Examples/Monocular/fusion/fusion"
	fusionArgs = []string{
		"/ORB_SLAM2/Examples/Monocular/KITTI00-02.yaml",
		"/ORB_SLAM2/Vocabulary/ORBvoc.txt",
		"/ORB_SLAM2/input.png",
	}
	fusionRoad             = "/ORB_SLAM2/input.png"
	fusionTask *FusionTask = nil
	fusionLock             = sync.Mutex{}
)

type FusionTask struct {
	*CMDTask
	InputChan           chan string
	OutputChan          chan string
	LocalizeChan        chan bool
	LocalizationLatency time.Duration
}

func newFusionTask() *FusionTask {
	task := &FusionTask{
		CMDTask:      NewCMDTask(fusionCMD),
		InputChan:    make(chan string),
		LocalizeChan: make(chan bool, 1),
	}
	return task
}

func GetFusionTask() *FusionTask {
	fusionLock.Lock()
	if fusionTask == nil {
		fusionTask = newFusionTask()
		fusionTask.OutputChan = fusionTask.nonBlockedRun(fusionTask.InputChan)
	}
	fusionLock.Unlock()
	return fusionTask
}

func (task *FusionTask) ResetSLAM() {
	task.InputChan <- "Reset\n"
	log.Println("Reset!")
}

func (task *FusionTask) GetImagePath() string {
	return fusionRoad
}

func (task *FusionTask) nonBlockedRun(inputChan chan string) chan string {
	outputChan := make(chan string)

	go func() {
		stdOutChan := make(chan string)
		task.CMDTask.RunWithStd(inputChan, stdOutChan, fusionArgs...)
		for {
			fusionResults := ""
			for {
				line := <-stdOutChan + "\n"

				if line == "Done\n" {
					outputChan <- "Done\n"
					break
				}

				if line == "NULL\n" {
					outputChan <- "NULL\n"
					break
				}

				if line == "-1\n" {
					outputChan <- fusionResults
					break
				}

				fusionResults = fusionResults + line
				log.Println("Receive new line: " + line)
			}
		}
	}()

	return outputChan
}

func (task *FusionTask) WriteFusionInputFile(form *multipart.Form) {
	file, err := form.File["frame"][0].Open()
	if err != nil {
		log.Panic(err)
	}
	defer func(file multipart.File) {
		err = file.Close()
		if err != nil {
			log.Panic(err)
		}
	}(file)

	create, err := os.OpenFile(fusionRoad, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		log.Panic(err)
	}

	_, err = io.Copy(create, file)
	if err != nil {
		log.Panic(err)
	}

	err = create.Close()
	if err != nil {
		log.Panic(err)
	}

	log.Println(" Write file ", "input.png")
}

func (task *FusionTask) Localize(form *multipart.Form) {
	now := time.Now()
	fusionTask.InputChan <- "Localize\n"
	<-fusionTask.OutputChan
	task.LocalizeChan <- true
	task.LocalizationLatency = time.Since(now)
}

func (task *FusionTask) FormattedDETResult(detResult string) string {
	if strings.Contains(detResult, "NULL") {
		return "-1\n"
	}
	detResult = strings.ReplaceAll(detResult, ",", " ")
	return detResult + "-1\n"
}
