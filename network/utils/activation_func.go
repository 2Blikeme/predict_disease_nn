package utils

import "math"

// Sigmoid sigmoid является реализацией сигмоиды,
// используемой для активации.
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// SigmoidPrime sigmoidPrime является реализацией производной
// сигмоиды для обратного распространения.
func SigmoidPrime(x float64) float64 {
	return x * (1.0 - x)
}
