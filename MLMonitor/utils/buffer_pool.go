package utils

import (
	"bytes"
	"log"
	"sync"
	"time"
)

var bufferLock = &sync.Mutex{}
var bufferSize = 10

type BufferElem struct {
	Buffer *bytes.Buffer
	Index  int
}

var bufferPool *[]*BufferElem = nil
var isBufferOccupy = make([]bool, bufferSize)

func GetBuffer() *BufferElem {
	bufferLock.Lock()
	defer bufferLock.Unlock()

	// Initialize the bufferPool if bufferPool is nil.
	if bufferPool == nil {
		bufferPool = &[]*BufferElem{}
		for i := 0; i < bufferSize; i++ {
			*bufferPool = append(*bufferPool, &BufferElem{
				Buffer: &bytes.Buffer{},
				Index:  i,
			})
		}
	}

	var chosenBuffer *BufferElem = nil

	for {

		for i := 0; i < bufferSize; i++ {
			if isBufferOccupy[i] == false {
				isBufferOccupy[i] = true
				chosenBuffer = (*bufferPool)[i]
				break
			}
		}

		if chosenBuffer != nil {
			break
		}
		bufferLock.Unlock()
		log.Fatal("No available buffer in the buffer pool")
		time.Sleep(5 * time.Second)
		bufferLock.Lock()
	}

	return chosenBuffer
}

func ReturnBuffer(bufferElem *BufferElem) {
	bufferLock.Lock()
	defer bufferLock.Unlock()
	isBufferOccupy[bufferElem.Index] = false
	bufferElem.Buffer.Reset()
}
