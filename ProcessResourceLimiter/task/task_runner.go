package task

import (
	"github.com/luoxiao23333/ProcessResourceLimiter/resources"
)

type Runner struct {
	resourceLimit resources.Profile
	task          Task
}

func NewTaskRunner(resourcesLimit resources.Profile, task Task) *Runner {
	return &Runner{
		resourceLimit: resourcesLimit,
		task:          task,
	}
}

func (t *Runner) RunTask(args ...string) (string, ExitStatus, error) {
	return t.task.Run(args...)
}
