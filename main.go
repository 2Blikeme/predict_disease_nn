package main

import (
	"encoding/csv"
	"fmt"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	nn "lab2/network"
	"log"
	"os"
	"strconv"
	"time"
)

func main() {

	start := time.Now()

	d, err := os.Open("resources/symptom_Description.csv")
	if err != nil {
		return
	}
	defer d.Close()

	readerDisease := csv.NewReader(d)
	readerDisease.FieldsPerRecord = 2
	rawDiseaseData, err := readerDisease.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	diseases := make(map[string]int)
	for idx, record := range rawDiseaseData {
		if idx == 0 {
			continue
		}
		diseases[record[0]] = idx
	}

	f, err := os.Open("resources/Training.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	reader := csv.NewReader(f)

	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	inputsDataTrain := make([]float64, 132*len(rawCSVData))
	labelsDataTrain := make([]float64, len(diseases)*len(rawCSVData))

	var inputDataIndex int
	var labelDataIndex int

	for idx, record := range rawCSVData {
		if idx == 0 {
			continue
		}
		for i, value := range record {
			if i == 132 {
				for k, _ := range diseases {
					if k == value {
						labelsDataTrain[labelDataIndex] = 1
					} else {
						labelsDataTrain[labelDataIndex] = 0
					}
					labelDataIndex++
				}
				continue
			}
			inputsDataTrain[inputDataIndex], _ = strconv.ParseFloat(value, 64)
			inputDataIndex++
		}
	}

	// Form the matrices.
	inputs := mat.NewDense(len(rawCSVData), 132, inputsDataTrain)
	labels := mat.NewDense(len(rawCSVData), len(diseases), labelsDataTrain)

	// Define our network architecture and
	// learning parameters.
	config := nn.NeuralNetConfig{
		InputNeurons:  132,
		OutputNeurons: len(diseases),
		HiddenNeurons: 64,
		NumEpochs:     10000,
		LearningRate:  0.3,
	}

	// Train the neural network.
	network := nn.NewNetwork(config)
	if err := network.Train(inputs, labels); err != nil {
		log.Fatal(err)
	}

	ft, err := os.Open("resources/Testing.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer ft.Close()

	readerTest := csv.NewReader(ft)

	rawCSVTestData, err := readerTest.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	inputsDataTest := make([]float64, len(rawCSVTestData)*132)
	labelsDataTest := make([]float64, len(diseases)*len(rawCSVTestData))

	var inputDataTestIndex int
	var labelDataTestIndex int

	for idx, record := range rawCSVTestData {
		if idx == 0 {
			continue
		}
		for i, value := range record {
			if i == 132 {
				for k, _ := range diseases {
					if k == value {
						labelsDataTest[labelDataTestIndex] = 1
					} else {
						labelsDataTest[labelDataTestIndex] = 0
					}
					labelDataTestIndex++
				}
				continue
			}
			inputsDataTest[inputDataTestIndex], _ = strconv.ParseFloat(value, 64)
			inputDataTestIndex++
		}
	}

	testInputs := mat.NewDense(len(rawCSVTestData), 132, inputsDataTest)
	testLabels := mat.NewDense(42, len(diseases), labelsDataTest)

	// Делаем предсказание с помощью обученной модели.
	predictions, err := network.Predict(testInputs)
	if err != nil {
		log.Fatal(err)
	}

	// Рассчитываем точность модели.
	var truePosNeg int
	numPreds, _ := predictions.Dims()
	for i := 0; i < numPreds; i++ {

		// Получаем вид.
		labelRow := mat.Row(nil, i, testLabels)
		var species int
		for idx, label := range labelRow {
			if label == 1.0 {
				species = idx
				break
			}
		}

		// Считаем количество верных предсказаний.
		if predictions.At(i, species) == floats.Max(mat.Row(nil, i, predictions)) {
			truePosNeg++
		}
	}

	// Подсчитываем точность предсказаний.
	accuracy := float64(truePosNeg) / float64(numPreds)

	// Выводим точность.
	fmt.Printf("\nAccuracy = %0.2f\n\n", accuracy)

	elapsed := time.Since(start)
	log.Printf("Execution time %s", elapsed)
}
