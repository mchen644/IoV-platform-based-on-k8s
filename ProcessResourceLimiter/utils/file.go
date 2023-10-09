package utils

import (
	"archive/zip"
	"io"
	"log"
	"os"
	"path"
	"path/filepath"
	"strings"
)

func Zip(srcDir string, zipFileName string) {

	// 预防：旧文件无法覆盖
	err := os.RemoveAll(zipFileName)
	if err != nil {
		return
	}

	// 创建：zip文件
	zipFile, _ := os.Create(zipFileName)
	defer func(zipFile *os.File) {
		err = zipFile.Close()
		if err != nil {
			log.Panic(err)
		}
	}(zipFile)

	archive := zip.NewWriter(zipFile)
	defer func(archive *zip.Writer) {
		err = archive.Close()
		if err != nil {
			log.Panic(err)
		}
	}(archive)

	err = filepath.Walk(srcDir, func(path string, info os.FileInfo, err error) error {

		if path == srcDir {
			return nil
		}

		if err != nil {
			log.Panic(err)
		}

		// read file head
		header, _ := zip.FileInfoHeader(info)
		header.Name = strings.TrimPrefix(path, srcDir+`/`)

		if info.IsDir() {
			header.Name += `/`
		} else {
			// Set compress
			header.Method = zip.Deflate
		}

		writer, _ := archive.CreateHeader(header)
		if !info.IsDir() {
			file, _ := os.Open(path)
			defer func(file *os.File) {
				err = file.Close()
				if err != nil {
					log.Panic(err)
				}
			}(file)
			_, err = io.Copy(writer, file)
			if err != nil {
				log.Panic(err)
			}
		}
		return nil
	})
	if err != nil {
		log.Panic(err)
	}
}

func RemoveDirContent(dir string) {
	dirs, err := os.ReadDir(dir)
	if err != nil {
		log.Panic(err)
	}

	for _, d := range dirs {
		err = os.RemoveAll(path.Join([]string{"tmp", d.Name()}...))
		if err != nil {
			log.Panic(err)
		}
	}
}

func UnZIP(zipFile, destDir string) {
	// remove all old content
	RemoveDirContent(destDir)

	reader, err := zip.OpenReader(zipFile)
	defer func(reader *zip.ReadCloser) {
		err = reader.Close()
		if err != nil {
			log.Panic(err)
		}
	}(reader)
	if err != nil {
		log.Panic(err)
	}

	log.Printf("unzip %v to %v", zipFile, destDir)

	for _, item := range reader.File {
		log.Print(destDir + item.Name)
		if item.FileInfo().IsDir() {
			err = os.Mkdir(destDir+item.Name, 0777)
			if err != nil {
				log.Panic(err)
			}
		}
		rc, err := item.Open()
		dst, err := createFile(destDir + item.Name)
		_, err = io.Copy(dst, rc)
		if err != nil {
			log.Print(err)
		}
	}
}

func createFile(name string) (*os.File, error) {
	err := os.MkdirAll(string([]rune(name)[0:strings.LastIndex(name, "/")]), 0755)
	if err != nil {
		log.Panic(err)
	}
	return os.Create(name)
}
