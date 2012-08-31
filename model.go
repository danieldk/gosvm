package gosvm

/*
#cgo LDFLAGS: -lsvm
#include <stdlib.h>

#include "wrap.h"
*/
import "C"

import (
	"errors"
	"runtime"
	"unsafe"
)

// A model contains the trained SVM and can be used to predict the
// class of a seen or unseen instance.
type Model struct {
	model *C.svm_model_t
	// Keep a pointer to the problem, since C model depends on it.
	problem *Problem
}

// Load a previously saved model.
func LoadModel(filename string) (*Model, error) {
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	model := &Model{C.svm_load_model_wrap(cFilename), nil}

	if model.model == nil {
		return nil, errors.New("Cannot read model: " + filename)
	}

	runtime.SetFinalizer(model, finalizeModel)

	return model, nil
}

// Train an SVM using the given parameters and problem.
func TrainModel(param Parameters, problem *Problem) (*Model, error) {
	cParam := toCParameter(param)
	defer C.free(unsafe.Pointer(cParam))

	// Check validity of the parameters.
	r := C.svm_check_parameter_wrap(problem.problem, cParam)
	if r != nil {
		msg := C.GoString(r)
		return nil, errors.New(msg)
	}

	cmodel := C.svm_train_wrap(problem.problem, cParam)
	model := &Model{cmodel, problem}
	runtime.SetFinalizer(model, finalizeModel)
	return model, nil
}

// Predict the label of an instance using the given model.
func (model *Model) Predict(nodes []FeatureValue) float64 {
	cn := cNodes(nodes)
	defer C.nodes_free(cNodes(nodes))
	return float64(C.svm_predict_wrap(model.model, cn))
}

// Save the SVM model to a file.
func (model *Model) Save(filename string) error {
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))
	result := C.svm_save_model_wrap(model.model, cFilename)

	if result == -1 {
		return errors.New("Could not save model to file: " + filename)
	}

	return nil
}

func finalizeModel(model *Model) {
	C.svm_free_and_destroy_model_wrap(model.model)
	model.problem = nil
}
