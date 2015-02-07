package gosvm

/*
#include "wrap.h"
*/
import "C"

// Parameters for training an SVM.
type Parameters struct {
	SVMType     SVMType
	Kernel      Kernel
	CacheSize   float64 // Cache size in MB
	Epsilon     float64 // Stopping criterium
	Shrinking   bool    // Apply shrinking
	Probability bool    // Provide probability estimates
}

// Default training parameters: C-SVM classification with a constraint
// violation cost of 1, a linear kernel, a cache size of 1 MB,
// epsilon=0.001 as the stopping criterium, and no shrinking.
func DefaultParameters() Parameters {
	return Parameters{
		NewCSVC(1),
		NewLinearKernel(),
		1,
		0.001,
		false,
		false}
}

// This type represents a kernel.
type Kernel struct {
	kernelType C.int
	gamma      C.double
	coef0      C.double
	degree     C.int
}

// Create a linear kernel.
func NewLinearKernel() Kernel {
	return Kernel{C.LINEAR, 0, 0, 0}
}

// Create a polynomial kernel.
func NewPolynomialKernel(gamma, coef0 float64, degree int) Kernel {
	return Kernel{C.POLY, C.double(gamma), C.double(coef0), C.int(degree)}
}

// Create a Radial Basis Function (RBF) kernel.
func NewRBFKernel(gamma float64) Kernel {
	return Kernel{C.RBF, C.double(gamma), 0, 0}
}

// Create a sigmoid kernel.
func NewSigmoidKernel(gamma, coef0 float64) Kernel {
	return Kernel{C.SIGMOID, C.double(gamma), C.double(coef0), 0}
}

// Support vector machine type configuration
type SVMType struct {
	svmType C.int
	cost    C.double
	nu      C.double
	epsilon C.double
}

// C-Support Vector Classification (C-SVC)
func NewCSVC(cost float64) SVMType {
	return SVMType{C.C_SVC, C.double(cost), 0, 0}
}

// Nu-Support Vector Classification (nu-SVC).
func NewNuSVC(cost, nu float64) SVMType {
	return SVMType{C.NU_SVC, C.double(cost), C.double(nu), 0}
}

// One-class SVM.
func NewOneClass(nu float64) SVMType {
	return SVMType{C.ONE_CLASS, 0, C.double(nu), 0}
}

// Epsilon support vector regression (epsilon-SVR).
func NewEpsilonSVR(cost, epsilon float64) SVMType {
	return SVMType{C.EPSILON_SVR, C.double(cost), 0, C.double(epsilon)}
}

// Nu-support vector regression (nu-SVR).
func NewNuSVR(cost, nu float64) SVMType {
	return SVMType{C.NU_SVR, C.double(cost), C.double(nu), 0}
}

func toCParameter(param Parameters) *C.svm_parameter_t {
	cParam := C.gosvm_parameter_new()

	// SVM type parameters
	cParam.svm_type = param.SVMType.svmType
	cParam.C = param.SVMType.cost
	cParam.nu = param.SVMType.nu
	cParam.p = param.SVMType.epsilon

	// Kernel type parameters
	cParam.kernel_type = param.Kernel.kernelType
	cParam.gamma = param.Kernel.gamma
	cParam.coef0 = param.Kernel.coef0
	cParam.degree = param.Kernel.degree

	cParam.cache_size = C.double(param.CacheSize)
	cParam.eps = C.double(param.Epsilon)

	if param.Shrinking {
		cParam.shrinking = 1
	}

	if param.Probability {
		cParam.probability = 1
	}

	return cParam
}
