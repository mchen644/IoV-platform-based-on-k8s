package task

import (
	"log"
)

// RunTask
// An example method guide you how to use resource limiter to start a new process
// This example method exec "python test.py" in a resource limit manner
func RunTask(cpu uint64, mem int64) {
	var cmdTask = NewCMDTask("python")
	exitChan := cmdTask.Run("test.py")

	for {
		select {
		case exitInfo := <-exitChan:
			log.Printf("From PID %v, exited with Signal:{%v}, Code:{%v}, Output{%v}\n",
				cmdTask.GetProcessID(),
				exitInfo.Signal, exitInfo.Code, exitInfo.Output)
			return
		}
	}

}
