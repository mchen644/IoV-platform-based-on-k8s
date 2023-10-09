package config

import (
	"gopkg.in/yaml.v3"
	"log"
	"os"
)

type Config struct {
	TaskName string `yaml:"taskName"`
}

var config *Config = nil

func GetConfig() *Config {
	if config == nil {
		configFile, err := os.ReadFile("/task_runner_config.yaml")
		if err != nil {
			log.Panic(err)
		}

		config = new(Config)

		if err = yaml.Unmarshal(configFile, config); err != nil {
			log.Panic(err)
		}
	}

	return config
}
