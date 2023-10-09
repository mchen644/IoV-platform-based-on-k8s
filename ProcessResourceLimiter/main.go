package main

import (
	"bytes"
	"log"
	"os"
	"os/exec"
	"runtime"
	"strings"

	"github.com/luoxiao23333/ProcessResourceLimiter/config"
	"github.com/luoxiao23333/ProcessResourceLimiter/scheduler_comm"
	"github.com/luoxiao23333/ProcessResourceLimiter/task"
)

const (
	LogFileName = "task_runner.log"
	PermUserRW  = 0644 // PermUserRW -rw-r--r--
)

func main() {
	initLog()

	//if task.IsRunCgroupV2() == false {
	//	log.Panic("Warning! This system is not running cgroups v2!\n" +
	//		"Recommand Ubuntu 22.04+ with Linux kernal 5.8+")
	//}

	log.Println("Start Task Runner")
	log.Println("CPU cores number is ", runtime.NumCPU(), getMemAvailable())

	scheduler_comm.RunHttpServer()
}

func getMemAvailable() string {
	cmd := exec.Command("cat", "/proc/meminfo")
	output := bytes.Buffer{}
	cmd.Stdout = &output
	err := cmd.Run()
	if err != nil {
		log.Panic(err)
	}

	memInfoList := strings.Split(output.String(), "\n")
	for i := 0; i < len(memInfoList); i++ {
		if strings.Contains(memInfoList[i], "MemAvailable") {
			return memInfoList[i]
		}
	}

	return "Available Meminfo Read Failed"
}

func initLog() {
	logFile, err := os.OpenFile(LogFileName, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, PermUserRW)
	if err != nil {
		panic(err)
	}
	log.SetOutput(logFile)

	log.Printf("Start %v task", config.GetConfig().TaskName)

	taskName := config.GetConfig().TaskName
	if taskName == "fusion" {
		log.Println("Start fusion task forever")
		task.GetFusionTask()
	} else if taskName == "det" {
		log.Println("Start det task forever")
		task.GetDetTask()
	} else if taskName == "slam" {
		log.Println("Start slam task forever")
		task.GetSlamTask()
	}
}
