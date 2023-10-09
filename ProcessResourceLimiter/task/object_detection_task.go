package task

import (
	"io"
	"log"
	"mime/multipart"
	"os"
)

var (
	objectDetectionCMD        = "bash"
	objectDetectionArgs       = []string{"run.sh"}
	objectDetectionImagesRoad = "output/"
)

type MCMOTTask struct {
	*CMDTask
}

func NewMCMOTTask() *MCMOTTask {
	task := &MCMOTTask{
		CMDTask: NewCMDTask(objectDetectionCMD),
	}
	return task
}

func (task *MCMOTTask) BlockedRun(form *multipart.Form) (
	imageRoad, objectDetectionInfo string) {
	writeFile(form)
	exitChan := task.CMDTask.Run(objectDetectionArgs...)

	select {
	case exitInfo, _ := <-exitChan:
		imageRoad = objectDetectionImagesRoad
		objectDetectionInfo = exitInfo.Output
		return objectDetectionImagesRoad, objectDetectionInfo
	}
}

func writeFile(form *multipart.Form) {
	file, err := form.File["video"][0].Open()
	if err != nil {
		log.Panic(err)
	}
	defer func(file multipart.File) {
		err = file.Close()
		if err != nil {
			log.Panic(err)
		}
	}(file)

	create, err := os.Create("input/input.avi")
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

	log.Println(" Write file ", "input.avi")
}
