package task

import (
	"io"
	"log"
	"mime/multipart"
	"os"
	"sync"
)

// DET: DEtection and Tracking
var (
	detCMD  = "python"
	detArgs = []string{
		"MCMOT/src/track.py",
		"--test_containerd", "True",
		"--data_dir", "./",
		"--load_model", "./MCMOT/model/mcmot_last_track_hrnet_18_deconv.pth",
		"--reid_cls_ids", "0,1,2,3,4",
	}
	detRoad          = "/workspace/input/input.png"
	detTask *DETTask = nil
	detLock          = sync.Mutex{}
)

type DETTask struct {
	*CMDTask
	InputChan  chan string
	OutputChan chan string
}

func newDETTask() *DETTask {
	task := &DETTask{
		CMDTask:   NewCMDTask(detCMD),
		InputChan: make(chan string),
	}
	return task
}

func GetDetTask() *DETTask {
	detLock.Lock()
	if detTask == nil {
		detTask = newDETTask()
		detTask.OutputChan = detTask.nonBlockedRun(detTask.InputChan)
	}
	detLock.Unlock()
	return detTask
}

func (task *DETTask) ResetModel() {
	task.InputChan <- "Reset\n"
	log.Println("Reset!")
}

func (task *DETTask) GetImagePath() string {
	return detRoad
}

func (task *DETTask) nonBlockedRun(inputChan chan string) chan string {
	outputChan := make(chan string)

	go func() {
		stdOutChan := make(chan string)
		task.CMDTask.RunWithStd(inputChan, stdOutChan, detArgs...)
		for {
			results := ""
			for {
				line := <-stdOutChan + "\n"

				if line == "NULL\n" {
					outputChan <- "NULL\n"
					break
				}

				if line == "-1\n" {
					outputChan <- results
					break
				}

				results = results + line
				log.Println("Receive new line: " + line)
			}
		}
	}()

	return outputChan
}

func (task *DETTask) DoDET(form *multipart.Form) {
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

	create, err := os.OpenFile(detRoad, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
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

	task.InputChan <- detRoad + "\n"
}
