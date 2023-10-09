package task

import (
	"io"
	"log"
	"mime/multipart"
	"os"
	"sync"
)

var (
	slamCMD  = "/ORB_SLAM2/Examples/Monocular/fusion/fusion"
	slamArgs = []string{
		"/ORB_SLAM2/Examples/Monocular/KITTI00-02.yaml",
		"/ORB_SLAM2/Vocabulary/ORBvoc.txt",
		"/ORB_SLAM2/input.png",
	}
	slamRoad           = "/ORB_SLAM2/input.png"
	slamTask *SlamTask = nil
	slamLock           = sync.Mutex{}
)

type SlamTask struct {
	*CMDTask
	InputChan  chan string
	OutputChan chan string
}

func newSlamTask() *SlamTask {
	task := &SlamTask{
		CMDTask:   NewCMDTask(slamCMD),
		InputChan: make(chan string),
	}
	return task
}

func GetSlamTask() *SlamTask {
	slamLock.Lock()
	if slamTask == nil {
		slamTask = newSlamTask()
		slamTask.OutputChan = slamTask.nonBlockedRun(slamTask.InputChan)
	}
	slamLock.Unlock()
	return slamTask
}

func (task *SlamTask) ResetSLAM() {
	task.InputChan <- "Reset\n"
	log.Println("Reset!")
}

func (task *SlamTask) GetImagePath() string {
	return slamRoad
}

func (task *SlamTask) nonBlockedRun(inputChan chan string) chan string {
	outputChan := make(chan string)

	go func() {
		stdOutChan := make(chan string)
		task.CMDTask.RunWithStd(inputChan, stdOutChan, slamArgs...)
		for {
			slamResults := ""
			for {
				line := <-stdOutChan + "\n"

				if line == "NULL\n" {
					outputChan <- "NULL\n"
					break
				}

				if line == "-1\n" {
					outputChan <- slamResults
					break
				}

				slamResults = slamResults + line
				log.Println("Receive new line: " + line)
			}
		}
	}()

	return outputChan
}

func (task *SlamTask) WriteSlamInputFile(form *multipart.Form) {
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

	create, err := os.OpenFile(slamRoad, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
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

func (task *SlamTask) Localize(form *multipart.Form) {
	task.WriteSlamInputFile(form)
	slamTask.InputChan <- "Localize\n"
}
