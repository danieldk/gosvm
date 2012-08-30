package libsvm

/*
#cgo LDFLAGS: -lsvm
#include "wrap.h"
*/
import "C"

import (
	"runtime"
)

// A model contains the trained SVM and can be used to predict the
// class of a seen or unseen instance.
type Model struct {
	model *C.svm_model_t
	// Keep a pointer to the problem, since C model depends on it.
	problem *Problem
}

// Train an SVM using the given problem.
func TrainModel(problem *Problem) *Model {
	param := C.parameter_new()
	cmodel := C.svm_train_wrap(problem.problem, param)
	model := &Model{cmodel, problem}
	runtime.SetFinalizer(model, finalizeModel)
	return model
}

// Save the SVM model to a file.
func (model *Model) Save() {
	C.svm_save_model_wrap(model.model)
}

func finalizeModel(model *Model) {
	C.svm_free_and_destroy_model_wrap(model.model)
	model.problem = nil
}
