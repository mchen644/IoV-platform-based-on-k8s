package utils

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

var uniqueID int = 0
var idLock sync.Mutex

func GetUniqueID() string {
	var id int
	idLock.Lock()
	uniqueID++
	id = uniqueID
	idLock.Unlock()
	return strconv.Itoa(id)
}

func DebugWithTimeWait(message string) {
	if os.Getenv("Debug") == "False" {
		return
	}

	_, callerFile, callerLine, ok := runtime.Caller(1)
	if !ok {
		log.Println("Impossible for recovery debug info!")
	}

	log.Printf("\n----------------\nDebug Message: \n%v\nIn file: [%v], line: [%v]\n----------------",
		message, callerFile, callerLine)

	time.Sleep(1 * time.Second)
}

var bufferLock *sync.Mutex = &sync.Mutex{}
var bufferPool *[]*BufferElem = nil
var buferAvalable []bool

const bufferSize = 100

type BufferElem struct {
	Buffer *bytes.Buffer
	index  int
}

func GetBuffer() *BufferElem {
	bufferLock.Lock()
	if bufferPool == nil {
		bufferPool = &[]*BufferElem{}
		for i := 0; i < bufferSize; i++ {
			*bufferPool = append(*bufferPool, &BufferElem{
				Buffer: &bytes.Buffer{},
				index:  i,
			})
			buferAvalable = append(buferAvalable, true)
		}
	}

	var chooseBuffer *BufferElem = nil
	for {
		for i := 0; i < bufferSize; i++ {
			if buferAvalable[i] {
				buferAvalable[i] = false
				chooseBuffer = (*bufferPool)[i]
				break
			}
		}
		if chooseBuffer != nil {
			break
		}
		bufferLock.Unlock()
		log.Printf("No avalable buffer, wait for 5 second")
		time.Sleep(5 * time.Second)
		bufferLock.Lock()
	}

	bufferLock.Unlock()
	return chooseBuffer
}

func ReturnBuffer(bufferElem *BufferElem) {
	bufferLock.Lock()
	buferAvalable[bufferElem.index] = true
	bufferElem.Buffer.Reset()
	bufferLock.Unlock()
}

func IntToNDigitsString(n int, digits int) string {
	s := strconv.Itoa(n)
	for len(s) < digits {
		s = "0" + s
	}
	return s
}

func MarshalCSV(data interface{}) []byte {
	var lines [][]string

	rv := reflect.ValueOf(data)
	if rv.Kind() != reflect.Slice {
		log.Panic(fmt.Errorf("not a slice"))
	}

	rt := rv.Type().Elem()
	var fields []string
	for i := 0; i < rt.NumField(); i++ {
		field := rt.Field(i)
		tag := field.Tag.Get("csv")
		if tag != "" {
			fields = append(fields, tag)
		} else {
			fields = append(fields, field.Name)
		}
	}
	lines = append(lines, fields)

	for i := 0; i < rv.Len(); i++ {
		var row []string
		for j := 0; j < rt.NumField(); j++ {
			field := rt.Field(j)
			if tag := field.Tag.Get("csv"); tag != "" {
				row = append(row, fmt.Sprintf("%v", rv.Index(i).Field(j)))
			} else {
				row = append(row, fmt.Sprintf("%v", rv.Index(i).Field(j)))
			}
		}
		lines = append(lines, row)
	}

	var sb strings.Builder
	writer := csv.NewWriter(&sb)
	err := writer.WriteAll(lines)
	if err != nil {
		log.Panic(err)
	}
	if err = writer.Error(); err != nil {
		log.Panic(err)
	}

	return []byte(sb.String())
}
